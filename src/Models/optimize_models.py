#%% libs
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import VotingRegressor
from optimizers import *
import optuna

#%% paths

DATA_PATH = Path(__file__).parents[2] / 'Data/Extended_Data'
TRAIN_DATA_FILE = DATA_PATH / 'extended_train.csv'
TEST_DATA_FILE = DATA_PATH / 'extended_test.csv'
SUBMISSION_PATH = Path(__file__).parents[2] / 'submissions'

#%% constants

optimize_optuna = False
n_splits = 9

#%% load data
train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)

# %% get splits grouped by participants
gkf = GroupKFold(n_splits)

for fold, (_, val_) in enumerate(gkf.split(X=train, y=train['bg+1:00'], groups=train['p_num'])):

    train.loc[val_,'fold'] = int(fold)

# %% Train XGBoost ensembles by cross validation 

training_cols = train.drop(['id','p_num','time','fold','bg+1:00'], axis=1).columns

objective_XGB = get_optimizer_for_XGBoost(train, n_splits, training_cols)

if optimize_optuna:
    study = optuna.create_study(direction="minimize")
    study.optimize(objective_XGB, n_trials=200)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

# %%
