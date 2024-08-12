import os
import torch
import numpy as np
import pandas as pd
from random import shuffle, randrange
from sklearn.model_selection import StratifiedKFold


class DatasetABCD_dyn(torch.utils.data.Dataset):
    def __init__(self, sourcedir, k_fold=None, target_feature='sex', train=True, regression=False, dynamic_length=None):
        super().__init__()

        self.train = train
        self.target_feature = target_feature
        self.sourcedir = sourcedir
        self.dynamic_length = dynamic_length
        self.filename = '6195_timeseries-?x352'

        if os.path.isfile(os.path.join(sourcedir, 'abcd_abcc', f'{self.filename}.pth')):
            self.timeseries_dict = torch.load(os.path.join(sourcedir, 'abcd_abcc', f'{self.filename}.pth'))
        self.num_nodes = list(self.timeseries_dict.values())[0].shape[1]

        if os.path.isfile(os.path.join(sourcedir, 'abcd_abcc', 'datasplit_5folds.pth')):
            self.split_subject = torch.load(os.path.join(sourcedir, 'abcd_abcc', 'datasplit_5folds.pth'))

        label_df = pd.read_csv(os.path.join(sourcedir, 'abcd_abcc', 'label.csv')).set_index('subject_id')
        self.num_classes = 1 if regression else len(label_df[target_feature].unique())
        self.label_dict = label_df[self.target_feature].to_dict()  # {'sub-NDARINV1FDC7YAJ': 1.0}

        if isinstance(k_fold, int):
            self.folds = list(range(k_fold))


    def __len__(self):
        return len(self.subject_list)


    def set_fold(self, fold, train=True, val=False, test=False):
        if train:
            self.subject_list = self.split_subject['train'][fold]
        if val:
            self.subject_list = self.split_subject['val'][fold]
        if test:
            self.subject_list = self.split_subject['test'][fold]


    def __getitem__(self, idx):
        subject = self.subject_list[idx]
        timeseries = self.timeseries_dict[subject]
        timeseries = (timeseries - np.mean(timeseries, axis=0, keepdims=True)) / (np.std(timeseries, axis=0, keepdims=True) + 1e-9)

        if self.dynamic_length is not None:
            if self.train:
                sampling_init = randrange(len(timeseries)-self.dynamic_length)
                timeseries = timeseries[sampling_init:sampling_init+self.dynamic_length]

        label = self.label_dict[subject]
        if self.target_feature == 'sex':
            if label == 2.0:  # female
                label = 0
            elif label== 1.0:  # male
                label = 1
            else:
                raise

        return {'id': subject,
                'timeseries': torch.tensor(timeseries, dtype=torch.float32),
                'label': torch.tensor(label)}
    


class DatasetABCD_static(torch.utils.data.Dataset):
    def __init__(self, sourcedir, k_fold=None, target_feature='sex', train=True, regression=False):
        super().__init__()

        self.target_feature = target_feature
        self.filename = 'ABCD_ABCC_baselineYear1Arm1_atlas-Gordon2014FreeSurferSubcortical_fc352x352'

        if os.path.isfile(os.path.join(sourcedir, 'abcd_abcc', f'{self.filename}.pth')):
            self.timeseries_dict = torch.load(os.path.join(sourcedir,'abcd_abcc', f'{self.filename}.pth'))
        self.num_nodes = list(self.timeseries_dict.values())[0].shape[0]

        self.full_subject_list = []
        self.val_subject_list = []
        if train:
            with open(os.path.join(sourcedir, 'abcd_abcc', 'train.txt'), 'r') as f:
                for line in f.readlines():
                    self.full_subject_list.append(line.strip('\n'))
            with open(os.path.join(sourcedir, 'abcd_abcc', 'val.txt'), 'r') as f:
                for line in f.readlines():
                    self.val_subject_list.append(line.strip('\n'))
        else:
            with open(os.path.join(sourcedir, 'abcd_abcc', 'test.txt'), 'r') as f:
                for line in f.readlines():
                    self.full_subject_list.append(line.strip('\n'))
        
        label_df = pd.read_csv(os.path.join(sourcedir, 'abcd_abcc', 'label.csv')).set_index('subject_id')
        self.num_classes = 1 if regression else len(label_df[target_feature].unique())
        self.label_dict = label_df[self.target_feature].to_dict()  # {'sub-NDARINV1FDC7YAJ': 1.0}

        if isinstance(k_fold, int):
            self.folds = list(range(k_fold))
            if k_fold > 1:
                self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0)
            else:
                self.k_fold = None
                self.subject_list = self.full_subject_list


    def __len__(self):
        return len(self.subject_list)


    def set_fold(self, fold, train=True):
        if train:
            shuffle(self.full_subject_list)
            self.subject_list = self.full_subject_list
            return
        
        self.subject_list = self.val_subject_list

      
    def __getitem__(self, idx):
        subject = self.subject_list[idx]
        fcs = self.timeseries_dict[subject]
        label = self.label_dict[subject]

        if self.target_feature == 'sex':
            if label == 2.0:  # female
                label = 0
            elif label== 1.0:  # male
                label = 1
            else:
                raise

        return {'id': subject,
                'fcs': torch.tensor(fcs, dtype=torch.float32),
                'label': torch.tensor(label)}


class DatasetHCPRest(torch.utils.data.Dataset):
    def __init__(self, sourcedir, k_fold=None, target_feature='Gender', regression=False):
        super().__init__()
        self.target_feature = target_feature
        # self.filename = 'hcp-rest1200_schaefer2018_200Parcels_7Networks_timeseries200x1200'
        data_dir = '/cbica/home/lihon/comp_space/ForTingting'
        self.filename = 'hcp_rfMRI_REST1_LR_tc_Schaefer2018_400Parcels'
        self.sourcedir = sourcedir
        if os.path.isfile(os.path.join(data_dir, f'{self.filename}.pt')):
            self.timeseries_dict = torch.load(os.path.join(data_dir, f'{self.filename}.pt'))
        
        if os.path.isfile(os.path.join(sourcedir, 'hcp1200', 'hcp_rest_datasplit_5folds.pth')):
            self.split_subject = torch.load(os.path.join(sourcedir, 'hcp1200', 'hcp_rest_datasplit_5folds.pth'))

        self.num_nodes = list(self.timeseries_dict.values())[0].shape[0]
        behavioral_df = pd.read_csv(os.path.join(sourcedir, 'hcp1200', 'label.csv')).set_index('Subject')
        self.num_classes = 1 if regression else len(behavioral_df[target_feature].unique())
        self.behavioral_dict = behavioral_df[target_feature].to_dict()  # {100206:'M'}

        if isinstance(k_fold, int):
            self.folds = list(range(k_fold))


    def __len__(self):
        return len(self.subject_list)


    def set_fold(self, fold, train=True, val=False, test=False):
        if train:
            self.subject_list = self.split_subject['train'][fold]
        if val:
            self.subject_list = self.split_subject['val'][fold]
        if test:
            self.subject_list = self.split_subject['test'][fold]


    def __getitem__(self, idx):
        subject = self.subject_list[idx]
        timeseries = self.timeseries_dict[subject].transpose()
        timeseries = (timeseries - np.mean(timeseries, axis=0, keepdims=True)) / (np.std(timeseries, axis=0, keepdims=True) + 1e-9)
        label = self.behavioral_dict[int(subject)]

        if self.target_feature == 'Gender':
            if label=='F':
                label = 0
            elif label=='M':
                label = 1
            else:
                raise

        return {'id': subject,
                'timeseries': torch.tensor(timeseries, dtype=torch.float32), 
                'label': torch.tensor(label)}
    
    

class DatasetHCPRest_static(torch.utils.data.Dataset):
    def __init__(self, sourcedir, k_fold=None, target_feature='Gender', train=True, regression=False):
        super().__init__()
        self.target_feature = target_feature
        data_dir = '/cbica/home/lihon/comp_space/ForTingting'
        self.filename = 'hcp_rfMRI_REST1_LR_fc_Schaefer2018_400Parcels'
        if os.path.isfile(os.path.join(data_dir, f'{self.filename}.pt')):
            self.timeseries_dict = torch.load(os.path.join(data_dir, f'{self.filename}.pt'))
        
        if os.path.isfile(os.path.join(sourcedir, 'hcp1200', 'hcp_rest_datasplit_5folds.pth')):
            self.split_subject = torch.load(os.path.join(sourcedir, 'hcp1200', 'hcp_rest_datasplit_5folds.pth'))

        self.num_nodes = list(self.timeseries_dict.values())[0][0].shape[0]
        behavioral_df = pd.read_csv(os.path.join(sourcedir, 'hcp1200', 'label.csv')).set_index('Subject')
        self.num_classes = 1 if regression else len(behavioral_df[target_feature].unique())
        self.behavioral_dict = behavioral_df[target_feature].to_dict()

        if isinstance(k_fold, int):
            self.folds = list(range(k_fold))


    def __len__(self):
        return len(self.subject_list)


    def set_fold(self, fold, train=True, val=False, test=False):
        if train:
            self.subject_list = self.split_subject['train'][fold]
        if val:
            self.subject_list = self.split_subject['val'][fold]
        if test:
            self.subject_list = self.split_subject['test'][fold]


    def __getitem__(self, idx):
        subject = self.subject_list[idx]
        fcs = self.timeseries_dict[subject]
        
        label = self.behavioral_dict[int(subject)]
        if self.target_feature == 'Gender':
            if label=='F':
                label = 0
            elif label=='M':
                label = 1
            else:
                raise

        return {'id': subject,
                'fcs': torch.tensor(fcs, dtype=torch.float32), 
                'label': torch.tensor(label)}