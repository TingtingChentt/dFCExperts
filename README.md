# dFCExperts
This repository is the official implementation of paper "dFCExperts: Learning Dynamic Functional Connectivity Patterns with Modularity and State Experts". 


## Dataset
The fMRI data used for the experiments of the paper should be downloaded from the [Human Connectome Project](https://db.humanconnectome.org/) and [ABCD_ABCC](https://osf.io/psv5m/). 

### Example structure of the data folder
```
data (specified by option --sourcedir)
├─── hcp1200
│    ├─── label.csv
│    ├─── hcp_rest_datasplit_5folds.pth
│    ├─── hcp_rfMRI_REST1_LR_fc_Schaefer2018_400Parcels.pt
│    └─── hcp_rfMRI_REST1_LR_tc_Schaefer2018_400Parcels.pt
├─── abcd_abcc
│    ├─── label.csv
│    ├─── hcp_rest_datasplit_5folds.pth
│    └─── hcp_rfMRI_REST1_LR_tc_Schaefer2018_400Parcels.pt
└─── samples
     ├─── sample_timeseries_data.pth
     ├─── sample_split_6folds.pth
     └─── label.csv

```
   
## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```


## Training

To train the model(s) with given sample data, run this command:

```train
python3 main.py --exp_name 'hcp_c' \
                --dataset 'hcp-sample' \
                --targetdir './result' \
                --target_feature 'Gender' \
                --gin_type 'moe_gin' \
                --num_gin_experts 5 \
                --num_states 7 \
                --state_ex_loss_coeff 10 \
                --orthogonal \
                --freeze_center \
                --project_assignment \
                --fc_hidden 256 \
                --num_epochs 30 \
                --minibatch_size 8 \
                --train \
                --validate \
                --test \
                --test_model_name 'model_val_acc'
```
