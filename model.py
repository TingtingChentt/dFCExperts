import torch
import torch.nn.functional as F
import torch.nn as nn
import util
from einops import rearrange
from state import StateEx

class LayerGIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, epsilon=True):
        super().__init__()
        if epsilon: self.epsilon = nn.Parameter(torch.Tensor([[0.0]])) # assumes that the adjacency matrix includes self-loop
        else: self.epsilon = 0.0
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), 
                                 nn.BatchNorm1d(hidden_dim), 
                                 nn.ReLU(), 
                                 nn.Linear(hidden_dim, hidden_dim), 
                                 nn.BatchNorm1d(hidden_dim), 
                                 nn.ReLU())

    def forward(self, v, a):
        v_aggregate = torch.sparse.mm(a, v)
        v_aggregate += (1 + self.epsilon) * v # assumes that the adjacency matrix includes self-loop
        v_combine = self.mlp(v_aggregate)
        return v_combine


class LayerGIN_MoE(nn.Module):
    def __init__(self, num_features, hidden_dim, num_experts, s_coef, b_coef):
        super().__init__()
        self.num_experts = num_experts
        self.sparse_loss_coef = s_coef
        self.balance_loss_coef = b_coef
        self.gate = nn.Linear(num_features, self.num_experts)

        self.gin_experts = nn.ModuleDict()
        for idx in range(num_experts):
            self.gin_experts[f"gin_expert_{idx}"] = LayerGIN(num_features, hidden_dim)
    
    def gating(self, v):
        #shape b,t,n,c
        logits = self.gate(v)
        probs = F.softmax(logits, dim=-1)
        values, indexes = torch.topk(probs, k=1)
        zeros = torch.zeros_like(probs, requires_grad=True)
        gates = zeros.scatter(-1, indexes, values)  # shape b,t,n,num_experts
        return gates, probs

    def forward(self, v, a, b, t, n):
        # v shape: (btn)*c
        # first get the gating value
        x = rearrange(v, '(b t n) c -> b t n c', b=b, t=t, n=n)
        gates, probs = self.gating(x)

        bal_loss = self.balance_loss_coef * util.loss.node_balance_loss(probs)
        spa_loss = - self.sparse_loss_coef * util.loss.sparse_loss(probs)

        gins_out = []
        for _, gin in enumerate(self.gin_experts.values()):
            gins_out.append(gin(v, a))
        
        # shape (btn)*c --> (btn)*e*c
        gins_out = torch.stack(gins_out, dim=-2)
        gates = rearrange(gates, 'b t n e -> (b t n) e')
        gins_out = torch.sum(gins_out * gates.unsqueeze(-1), dim=-2)

        return gins_out, (bal_loss, spa_loss)


class ModularityEx(nn.Module):
    def __init__(self, gin_type, num_features, gnn_hidden, fc_hidden, num_layers, sparsity, drop_ratio, graph_pooling, num_experts=7, s_coef=1, b_coef=1):
        super().__init__()
        self.gin_type = gin_type  # gin or moe_gin
        self.sparsity = sparsity
        self.drop_ratio = drop_ratio
        self.num_layers = num_layers
        self.percentile = Percentile()
        self.gnn_layers = nn.ModuleList()

        if graph_pooling=='sum': self.pool = lambda x: x.sum(-2)
        elif graph_pooling=='mean': self.pool = lambda x: x.mean(-2)
        elif graph_pooling=='max': self.pool = lambda x: x.max(-2)
        else: raise

        if num_layers>0:
            if gin_type == 'gin':
                self.gnn_layers.append(LayerGIN(num_features, gnn_hidden))
            elif gin_type == 'moe_gin':
                self.gnn_layers.append(LayerGIN_MoE(num_features, gnn_hidden, num_experts, s_coef, b_coef))

            for _ in range(0, num_layers - 1):
                if gin_type == 'gin':
                    self.gnn_layers.append(LayerGIN(gnn_hidden, gnn_hidden))
                elif gin_type == 'moe_gin':
                    self.gnn_layers.append(LayerGIN_MoE(gnn_hidden, gnn_hidden, num_experts, s_coef, b_coef))
        

        input_dim1 = int(((num_features * num_features)/2)- (num_features/2) + (gnn_hidden * num_layers))
        input_dim = int(((num_features * num_features)/2)- (num_features/2))
        self.bn = nn.BatchNorm1d(input_dim)
        self.bnh = nn.BatchNorm1d(gnn_hidden*num_layers)
        self.fc = nn.Sequential(
            nn.Linear(input_dim1, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def _collate_adjacency(self, a, sparsity, sparse=True):
        i_list = []
        v_list = []

        for sample, _dyn_a in enumerate(a):
            for timepoint, _a in enumerate(_dyn_a):
                thresholded_a = (_a > self.percentile(_a, 100-sparsity))
                _i = thresholded_a.nonzero(as_tuple=False)
                _v = torch.ones(len(_i))
                _i += sample * a.shape[1] * a.shape[2] + timepoint * a.shape[2]
                i_list.append(_i)
                v_list.append(_v)
        _i = torch.cat(i_list).T.to(a.device)
        _v = torch.cat(v_list).to(a.device)

        return torch.sparse.FloatTensor(_i, _v, (a.shape[0]*a.shape[1]*a.shape[2], a.shape[0]*a.shape[1]*a.shape[3]))

    
    def forward(self, a, v):
        minibatch_size, times, num_nodes = a.shape[:3]
        v = rearrange(v, 'b t n c -> (b t n) c')
        a_sparse = self._collate_adjacency(a, self.sparsity)

        if self.gin_type == 'moe_gin':
            self.load_balance_loss = 0.0
            self.b_loss = 0.0
            self.s_loss = 0.0

        h_list = []
        for layer, gnn in enumerate(self.gnn_layers):
            if self.gin_type == 'gin':
                v = gnn(v, a_sparse)
            elif self.gin_type == 'moe_gin':
                v, _layer_loss = gnn(v, a_sparse, minibatch_size, times, num_nodes)
                (bal_loss, spa_loss) = _layer_loss
                loss = bal_loss + spa_loss
                self.b_loss += bal_loss
                self.s_loss += spa_loss
                self.load_balance_loss += loss
                
            if layer == self.num_layers - 1:
                #remove relu for the last layer
                v = F.dropout(v, self.drop_ratio, training = self.training)
            else:
                v = F.dropout(F.relu(v), self.drop_ratio, training = self.training)

            h_list.append(self.pool(rearrange(v, '(b t n) c -> b t n c', b=minibatch_size, t=times, n=num_nodes)))
        

        x_inputs = []
        for t in range(times):
            x_input = util.bold_voxel.flatten_upper_triangle_batch(a[:,t,:,:])
            x_input = self.bn(x_input)
            x_inputs.append(x_input)
        x_input = torch.stack(x_inputs, dim=1)
        x_input = rearrange(x_input, 'b t c -> (b t) c')

        h = torch.cat(h_list, dim=-1)
        h = self.bnh(rearrange(h, 'b t c -> (b t) c'))
        out = torch.cat((x_input, h), dim=-1)
        out = self.fc(out)
        return rearrange(out, '(b t) c -> b t c', b=minibatch_size, t=times)
    

class dFCExperts(nn.Module):
    def __init__(self, argv, num_features, num_classes):
        super().__init__()
        self.s_ex_loss_coeff = argv.state_ex_loss_coeff
        self.m_ex = ModularityEx(gin_type=argv.gin_type,
                                num_features=num_features,
                                gnn_hidden=argv.gin_hidden,
                                fc_hidden=argv.fc_hidden,
                                num_layers=argv.num_gin_layers,
                                sparsity=argv.sparsity,
                                drop_ratio=argv.dropout,
                                graph_pooling=argv.graph_pooling,
                                num_experts=argv.num_gin_experts,
                                s_coef=argv.gin_s_loss_coeff,
                                b_coef=argv.gin_b_loss_coeff)

        self.s_ex = StateEx(hidden_dim=argv.fc_hidden,
                            num_states=argv.num_states, 
                            orthogonal=argv.orthogonal,
                            freeze_center=argv.freeze_center, 
                            project_assignment=argv.project_assignment)

        self.predict = nn.Sequential(
            nn.Linear(argv.fc_hidden, argv.fc_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(argv.fc_hidden, argv.fc_hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear((argv.fc_hidden//2), num_classes),
        )


    def forward(self, a, v):
        x = self.m_ex(a, v)
        state_repr, state_assignments = self.s_ex(x)
        logits = self.predict(state_repr)
        return logits, state_assignments
    
    def loss(self, state_assignments):
        loss = self.s_ex_loss_coeff * self.s_ex.loss(state_assignments)
        return loss


# Percentile class based on
# https://github.com/aliutkus/torchpercentile
class Percentile(torch.autograd.Function):
    def __init__(self):
        super().__init__()


    def __call__(self, input, percentiles):
        return self.forward(input, percentiles)


    def forward(self, input, percentiles):
        input = torch.flatten(input) # find percentiles for flattened axis
        input_dtype = input.dtype
        input_shape = input.shape
        if isinstance(percentiles, int):
            percentiles = (percentiles,)
        if not isinstance(percentiles, torch.Tensor):
            percentiles = torch.tensor(percentiles, dtype=torch.double)
        if not isinstance(percentiles, torch.Tensor):
            percentiles = torch.tensor(percentiles)
        input = input.double()
        percentiles = percentiles.to(input.device).double()
        input = input.view(input.shape[0], -1)
        in_sorted, in_argsort = torch.sort(input, dim=0)
        positions = percentiles * (input.shape[0]-1) / 100
        floored = torch.floor(positions)
        ceiled = floored + 1
        ceiled[ceiled > input.shape[0] - 1] = input.shape[0] - 1
        weight_ceiled = positions-floored
        weight_floored = 1.0 - weight_ceiled
        d0 = in_sorted[floored.long(), :] * weight_floored[:, None]
        d1 = in_sorted[ceiled.long(), :] * weight_ceiled[:, None]
        self.save_for_backward(input_shape, in_argsort, floored.long(),
                               ceiled.long(), weight_floored, weight_ceiled)
        result = (d0+d1).view(-1, *input_shape[1:])
        return result.type(input_dtype)


    def backward(self, grad_output):
        """
        backward the gradient is basically a lookup table, but with weights
        depending on the distance between each point and the closest
        percentiles
        """
        (input_shape, in_argsort, floored, ceiled,
         weight_floored, weight_ceiled) = self.saved_tensors

        # the argsort in the flattened in vector

        cols_offsets = (
            torch.arange(
                    0, input_shape[1], device=in_argsort.device)
            )[None, :].long()
        in_argsort = (in_argsort*input_shape[1] + cols_offsets).view(-1).long()
        floored = (
            floored[:, None]*input_shape[1] + cols_offsets).view(-1).long()
        ceiled = (
            ceiled[:, None]*input_shape[1] + cols_offsets).view(-1).long()

        grad_input = torch.zeros((in_argsort.size()), device=self.device)
        grad_input[in_argsort[floored]] += (grad_output
                                            * weight_floored[:, None]).view(-1)
        grad_input[in_argsort[ceiled]] += (grad_output
                                           * weight_ceiled[:, None]).view(-1)

        grad_input = grad_input.view(*input_shape)
        return grad_input
