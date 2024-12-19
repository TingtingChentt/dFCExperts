import os
import util
import random
import torch
import numpy as np
from random import randrange
from dataset import *
from tqdm import tqdm
from einops import repeat
from model import dFCExperts

from torch.utils.tensorboard import SummaryWriter
import torch.profiler
from torchvision.utils import make_grid


def step(argv, model, criterion, dyn_v, dyn_a, label, clip_grad=0.0, device='cpu', optimizer=None, scheduler=None):
    if optimizer is None: model.eval()
    else: model.train()

    # run model
    logit, state_assignments = model(dyn_a.to(device), dyn_v.to(device))

    b, t, _ = logit.size()
    label_t = repeat(label, 'b -> b t', b=b, t=t)
    if argv.regression:
        pred_loss = criterion(logit.squeeze(), label_t.to(device))
    else:
        pred_loss = criterion(torch.permute(logit, (0,2,1)), label_t.to(device))

    state_loss = model.loss(state_assignments)
    loss = pred_loss +  model.gin.s_loss + model.gin.b_loss + state_loss
    loss_ = (pred_loss, model.gin.s_loss, model.gin.b_loss, state_loss)

    # optimize model
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        if clip_grad > 0.0: torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    if argv.regression:
        return logit.mean(1).reshape((-1,)), loss_
    else:
        return logit.mean(1), loss_


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(argv):
    # make directories
    os.makedirs(os.path.join(argv.targetdir, 'model'), exist_ok=True)
    os.makedirs(os.path.join(argv.targetdir, 'summary'), exist_ok=True)

    # set seed and device
    torch.manual_seed(argv.seed)
    np.random.seed(argv.seed)
    random.seed(argv.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(argv.seed)
    else:
        device = torch.device("cpu")

    # define dataset
    # hcp-rest, target_feature: Gender, PMAT24_A_CR, ReadEng_Unadj, PicVocab_Unadj
    if argv.dataset=='hcp-dyn': 
        dataset = DatasetHCPRest(argv.sourcedir, k_fold=argv.k_fold, target_feature=argv.target_feature, regression=argv.regression)
    # abcd, target_feature: sex, p_factor, pc1
    elif argv.dataset=='abcd-dyn': 
        dataset = DatasetABCD_dyn(argv.sourcedir, k_fold=argv.k_fold, target_feature=argv.target_feature, train=True, regression=argv.regression, dynamic_length=argv.dynamic_length)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=argv.minibatch_size, shuffle=False, num_workers=argv.num_workers, pin_memory=True)

    # resume checkpoint if file exists
    if os.path.isfile(os.path.join(argv.targetdir, 'checkpoint.pth')):
        print('resuming checkpoint experiment')
        checkpoint = torch.load(os.path.join(argv.targetdir, 'checkpoint.pth'), map_location=device)
    else:
        checkpoint = {
            'fold': 0,
            'epoch': 0,
            'model': None,
            'optimizer': None,
            'scheduler': None}

    # start experiment
    for k_index, k in enumerate(dataset.folds):
        if checkpoint['fold']:
            if k_index < dataset.folds.index(checkpoint['fold']):
                continue
        # make directories per fold
        os.makedirs(os.path.join(argv.targetdir, 'model', str(k)), exist_ok=True)
        # set dataloader
        dataset.set_fold(k, train=True)

        # define model
        model = dFCExperts(argv, num_features=dataset.num_nodes, num_classes=dataset.num_classes)
        model.to(device)

        if checkpoint['model'] is not None: model.load_state_dict(checkpoint['model'])
        criterion = torch.nn.CrossEntropyLoss() if dataset.num_classes > 1 else torch.nn.MSELoss()

        # define optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=argv.lr)
        if checkpoint['optimizer'] is not None: optimizer.load_state_dict(checkpoint['optimizer'])

        # define logging objects
        summary_writer = SummaryWriter(os.path.join(argv.targetdir, 'summary', str(k), 'train'), )
        summary_writer_val = SummaryWriter(os.path.join(argv.targetdir, 'summary', str(k), 'val'), )
        logger = util.logger.LoggerdFCExperts(dataset.folds, dataset.num_classes)

        best_acc, best_mse = 0.0, np.inf
        # start training
        for epoch in range(checkpoint['epoch'], argv.num_epochs):
            logger.initialize(k)
            dataset.set_fold(k, train=True)
            loss_pred_accu, loss_total_accu = 0.0, 0.0
            loss_s_gin_accu, loss_b_gin_accu = 0.0, 0.0
            loss_state_accu = 0.0
            for i, x in enumerate(tqdm(dataloader, ncols=60, desc=f'k:{k} e:{epoch}')):
                # process input data
                dyn_a, _ = util.bold.process_dynamic_fc(x['timeseries'], argv.window_size, argv.window_stride, argv.dynamic_length)
                dyn_a = torch.nan_to_num(dyn_a, nan=0.0)
                dyn_v = dyn_a
                label = x['label']

                logit, loss = step(
                    argv,
                    model=model,
                    criterion=criterion,
                    dyn_v=dyn_v,
                    dyn_a=dyn_a,
                    label=label,
                    clip_grad=argv.clip_grad,
                    device=device,
                    optimizer=optimizer
                )

                (loss_pred, loss_s_gin, loss_b_gin, loss_state) = loss
                loss_total_accu += loss.detach().cpu().numpy()
                loss_pred_accu += loss_pred.detach().cpu().numpy()
                loss_s_gin_accu += loss_s_gin.detach().cpu().numpy()
                loss_b_gin_accu += loss_b_gin.detach().cpu().numpy()
                loss_state_accu += loss_state.detach().cpu().numpy()

                pred = logit.argmax(1) if dataset.num_classes > 1 else logit
                prob = logit.softmax(1) if dataset.num_classes > 1 else  logit
                logger.add(k=k, pred=pred.detach().cpu().numpy(), true=label.detach().cpu().numpy(), prob=prob.detach().cpu().numpy())
                summary_writer.add_scalar('lr', argv.lr, i+epoch*len(dataloader))
                
            # summarize results
            samples = logger.get(k)
            metrics = logger.evaluate(k)
            summary_writer.add_scalar('loss_total', loss_total_accu/len(dataloader), epoch)
            summary_writer.add_scalar('loss_pred', loss_pred_accu/len(dataloader), epoch)
            summary_writer.add_scalar('loss_state', loss_state_accu/len(dataloader), epoch)
            summary_writer.add_scalar('loss_sparse_gin', loss_s_gin_accu/len(dataloader), epoch)
            summary_writer.add_scalar('loss_balance_gin', loss_b_gin_accu/len(dataloader), epoch)

            if dataset.num_classes > 1: summary_writer.add_pr_curve('precision-recall', samples['true'], samples['prob'][:,1], epoch)
            [summary_writer.add_scalar(key, value, epoch) for key, value in metrics.items() if not key=='fold']
            summary_writer.flush()

            # save checkpoint
            torch.save({
                'fold': k,
                'epoch': epoch+1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                os.path.join(argv.targetdir, 'checkpoint.pth'))
            
            if argv.validate:
                print('validating. not for testing purposes')
                logger.initialize(k)
                loss_pred_accu, loss_total_accu = 0.0, 0.0
                loss_s_gin_accu, loss_b_gin_accu = 0.0, 0.0
                loss_state_accu = 0.0
                dataset.set_fold(k, train=False, val=True)
                for i, x in enumerate(dataloader):
                    with torch.no_grad():
                        # input data
                        dyn_a, _ = util.bold.process_dynamic_fc(x['timeseries'], argv.window_size, argv.window_stride, argv.dynamic_length)
                        dyn_a = torch.nan_to_num(dyn_a, nan=0.0)
                        dyn_v = dyn_a
                        label = x['label']

                        logit, loss = step(
                            argv,
                            model=model,
                            criterion=criterion,
                            dyn_v=dyn_v,
                            dyn_a=dyn_a,
                            label=label,
                            clip_grad=argv.clip_grad,
                            device=device,
                            optimizer=None
                        )
                        (loss_pred, loss_s_gin, loss_b_gin, loss_state) = loss
                        loss_total_accu += loss.detach().cpu().numpy()
                        loss_pred_accu += loss_pred.detach().cpu().numpy()
                        loss_s_gin_accu += loss_s_gin.detach().cpu().numpy()
                        loss_b_gin_accu += loss_b_gin.detach().cpu().numpy()
                        loss_state_accu += loss_state.detach().cpu().numpy()

                        pred = logit.argmax(1) if dataset.num_classes > 1 else logit
                        prob = logit.softmax(1) if dataset.num_classes > 1 else  logit
                        logger.add(k=k, pred=pred.detach().cpu().numpy(), true=label.detach().cpu().numpy(), prob=prob.detach().cpu().numpy())
                       
                samples = logger.get(k)
                metrics = logger.evaluate(k)
                summary_writer_val.add_scalar('loss_total', loss_total_accu/len(dataloader), epoch)
                summary_writer_val.add_scalar('loss_pred', loss_pred_accu/len(dataloader), epoch)
                summary_writer_val.add_scalar('loss_state', loss_state_accu/len(dataloader), epoch)
                summary_writer_val.add_scalar('loss_sparse_gin', loss_s_gin_accu/len(dataloader), epoch)
                summary_writer_val.add_scalar('loss_balance_gin', loss_b_gin_accu/len(dataloader), epoch)


                if dataset.num_classes > 1: summary_writer_val.add_pr_curve('precision-recall', samples['true'], samples['prob'][:,1], epoch)
                [summary_writer_val.add_scalar(key, value, epoch) for key, value in metrics.items() if not key=='fold']
                summary_writer_val.flush()

                # save the model
                if argv.regression:
                    if metrics['mse'] < best_mse:
                        best_mse = metrics['mse']
                        torch.save(model.state_dict(), os.path.join(argv.targetdir, 'model', str(k), 'model_val_mse.pth'))
                    if metrics['corr'] > best_corr:
                        best_corr = metrics['corr']
                        torch.save(model.state_dict(), os.path.join(argv.targetdir, 'model', str(k), 'model_val_corr.pth'))
                else:
                    if metrics['accuracy'] > best_acc:
                        best_acc = metrics['accuracy']
                        torch.save(model.state_dict(), os.path.join(argv.targetdir, 'model', str(k), 'model_val_acc.pth'))

        # finalize fold
        torch.save(model.state_dict(), os.path.join(argv.targetdir, 'model', str(k), 'model.pth'))
        checkpoint.update({'epoch': 0, 'model': None, 'optimizer': None, 'scheduler': None})

    summary_writer.close()
    summary_writer_val.close()
    os.remove(os.path.join(argv.targetdir, 'checkpoint.pth'))


def test(argv):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # define dataset
    if argv.dataset=='hcp-dyn': dataset = DatasetHCPRest(argv.sourcedir, k_fold=argv.k_fold, target_feature=argv.target_feature, regression=argv.regression)
    elif argv.dataset=='abcd-dyn': dataset = DatasetABCD_dyn(argv.sourcedir, k_fold=argv.k_fold, target_feature=argv.target_feature, train=False, regression=argv.regression)
    else: raise
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=argv.num_workers, pin_memory=True)
     # define logging objects
    logger = util.logger.LoggerdFCExperts(num_classes=dataset.num_classes)

    for k in dataset.folds:
        model = dFCExperts(argv, num_features=dataset.num_nodes, num_classes=dataset.num_classes)
        model.to(device)
        model.load_state_dict(torch.load(os.path.join(argv.targetdir, 'model', str(k), argv.test_model_name+'.pth')))
        print('load model from:', os.path.join(argv.targetdir, 'model', str(k), argv.test_model_name+'.pth'))
        criterion = torch.nn.CrossEntropyLoss() if dataset.num_classes > 1 else torch.nn.MSELoss()

        logger.initialize(k)
        dataset.set_fold(k, train=False, test=True)
        summary_writer = SummaryWriter(os.path.join(argv.targetdir, 'summary', str(k), 'test'))
        loss_pred_accu, loss_total_accu = 0.0, 0.0
        loss_s_gin_accu, loss_b_gin_accu = 0.0, 0.0
        loss_state_accu = 0.0
        for i, x in enumerate(tqdm(dataloader, ncols=60)):
            with torch.no_grad():
                # use the whole timeseries
                dyn_a, _ = util.bold.process_dynamic_fc(x['timeseries'], argv.window_size, argv.window_stride)
                dyn_a = torch.nan_to_num(dyn_a, nan=0.0)
                dyn_v = dyn_a
                label = x['label']

                logit, loss = step(
                    argv,
                    model=model,
                    criterion=criterion,
                    dyn_v=dyn_v,
                    dyn_a=dyn_a,
                    label=label,
                    clip_grad=argv.clip_grad,
                    device=device,
                    optimizer=None,
                    scheduler=None,
                )

                (loss_pred, loss_s_gin, loss_b_gin, loss_state) = loss
                loss_pred_accu += loss_pred.detach().cpu().numpy()
                loss_total_accu += loss.detach().cpu().numpy()
                loss_s_gin_accu += loss_s_gin.detach().cpu().numpy()
                loss_b_gin_accu += loss_b_gin.detach().cpu().numpy()
                loss_state_accu += loss_state.detach().cpu().numpy()

                pred = logit.argmax(1) if dataset.num_classes > 1 else logit
                prob = logit.softmax(1) if dataset.num_classes > 1 else  logit
                logger.add(k=k, pred=pred.detach().cpu().numpy(), true=label.detach().cpu().numpy(), prob=prob.detach().cpu().numpy())
                
        # summarize results
        samples = logger.get(k)
        metrics = logger.evaluate(k)
        summary_writer.add_scalar('loss_total', loss_total_accu/len(dataloader))
        summary_writer.add_scalar('loss_pred', loss_pred_accu/len(dataloader))
        summary_writer.add_scalar('loss_state', loss_state_accu/len(dataloader))
        summary_writer.add_scalar('loss_sparse_gin', loss_s_gin_accu/len(dataloader))
        summary_writer.add_scalar('loss_balance_gin', loss_b_gin_accu/len(dataloader))

        if dataset.num_classes > 1: summary_writer.add_pr_curve('precision-recall', samples['true'], samples['prob'][:,1])
        [summary_writer.add_scalar(key, value) for key, value in metrics.items() if not key=='fold']
        summary_writer.flush()
        # finalize fold
        logger.to_csv(argv.targetdir, k)

    # finalize experiment
    logger.to_csv(argv.targetdir)
    final_metrics = logger.evaluate()
    print(final_metrics)
    summary_writer.close()
    torch.save(logger.get(), os.path.join(argv.targetdir, 'samples.pkl'))
