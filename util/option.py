import os
import csv
import argparse


def parse():
    parser = argparse.ArgumentParser(description='fMRIExperts')
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-n', '--exp_name', type=str, default='fmriexperts_experiment')
    parser.add_argument('-k', '--k_fold', default=5)
    parser.add_argument('-b', '--minibatch_size', type=int, default=8)
    parser.add_argument('-ds', '--sourcedir', type=str, default='./data')
    parser.add_argument('-dt', '--targetdir', type=str, default='./result')

    # for data
    parser.add_argument('--dataset', type=str, default='hcp-static', choices=['hcp-static', 'hcp-dyn', 'abcd-static', 'abcd-dyn'])
    parser.add_argument('--target_feature', type=str, default='Gender', choices=['Gender', 'cog', 'cog_norm', 'cog_std', 'PMAT24_A_CR', 'ReadEng_Unadj', 'PicVocab_Unadj', 'sex', 'p_factor_std', 'pc1'])
    parser.add_argument('--window_size', type=int, default=50)
    parser.add_argument('--window_stride', type=int, default=3)
    parser.add_argument('--dynamic_length', type=int, default=600)

    # for modularity experts
    parser.add_argument('--gin_type', type=str, default='gin', choices=['gin', 'moe_gin'])
    parser.add_argument('--num_gin_layers', type=int, default=3)
    parser.add_argument('--num_gin_experts', type=int, default=7)
    parser.add_argument('--sparsity', type=int, default=30)
    parser.add_argument('--gin_hidden', type=int, default=128)
    parser.add_argument('--fc_hidden', type=int, default=128)
    parser.add_argument('--gin_s_loss_coeff', type=float, default='1')
    parser.add_argument('--gin_b_loss_coeff', type=float, default='1')
    parser.add_argument('--graph_pooling', type=str, default='mean', choices=['sum', 'max', 'mean'])
    parser.add_argument('--dropout', type=float, default=0.5)

    # for state experts
    parser.add_argument('--num_states', type=int, default=6)
    parser.add_argument('--s_loss_coeff', type=float, default='1')
    parser.add_argument('--b_loss_coeff', type=float, default='1')
    parser.add_argument('--state_ex_loss_coeff', type=float, default='1')
    parser.add_argument('--orthogonal', action='store_true')
    parser.add_argument('--freeze_center', action='store_true')
    parser.add_argument('--project_assignment', action='store_true')

    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--max_lr', type=float, default=0.001)
    parser.add_argument('--clip_grad', type=float, default=0.0)
    parser.add_argument('--num_epochs', type=int, default=30)

    parser.add_argument('--regression', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_model_name', type=str, default='model_val_acc')
    parser.add_argument('--num_workers', type=int, default=4)

    argv = parser.parse_args()
    argv.targetdir = os.path.join(argv.targetdir, argv.exp_name)
    os.makedirs(argv.targetdir, exist_ok=True)
    with open(os.path.join(argv.targetdir, 'argv.csv'), 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(vars(argv).items())
    return argv
