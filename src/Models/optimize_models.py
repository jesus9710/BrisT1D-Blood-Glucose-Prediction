# libs
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import VotingRegressor
from optimizers import *
import optuna

# paths
DATA_PATH = Path(__file__).parents[2] / 'Data/Extended_Data'
TRAIN_DATA_FILE = DATA_PATH / 'extended_train.csv'
SUBMISSION_PATH = Path(__file__).parents[2] / 'submissions'

# constants
model_architecture = "CatBoost"
optimize_optuna = True
n_splits = 5
subsample_quantity = 50000
n_trials = 50

# load data
train = pd.read_csv(TRAIN_DATA_FILE)
train = train.sample(subsample_quantity, random_state=42).reset_index(drop=True)

# get splits grouped by participants
gkf = GroupKFold(n_splits)

for fold, (_, val_) in enumerate(gkf.split(X=train, y=train['bg+1:00'], groups=train['p_num'])):

    train.loc[val_,'fold'] = int(fold)

# model optimization
training_cols = train.drop(['id','p_num','time','fold','bg+1:00'], axis=1).columns

if model_architecture == "XGBoost":
    objective_fn = get_optimizer_for_XGBoost(train, n_splits, training_cols)

elif model_architecture == "CatBoost":
    objective_fn = get_optimizer_for_CatBoost(train, n_splits, training_cols)

elif model_architecture == "LGBM":
    pass

if optimize_optuna:
        
    study = optuna.create_study(direction="minimize")
    study.optimize(objective_fn, n_trials=n_trials)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
