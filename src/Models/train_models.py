# libs
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
from catboost import Pool
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error
from Models.model_wrappers import XGB_VotingRegressor, CatBst_Voting_Regressor
import os

# paths
DATA_PATH = Path(__file__).parents[2] / 'Data/Extended_Data'
TRAIN_DATA_FILE = DATA_PATH / 'extended_train.csv'
MODEL_PATH = Path(__file__).parents[2] / 'Models'

# constants
model_architecture = "CatBoost"
model_name = 'EDx1_Some_FE_5mins_3VR_CatBoost_optimized_CV_v2'
n_splits = 9
save_model = True

# load data
train = pd.read_csv(TRAIN_DATA_FILE)

# get splits grouped by participants
gkf = GroupKFold(n_splits)

for fold, (_, val_) in enumerate(gkf.split(X=train, y=train['bg+1:00'], groups=train['p_num'])):

    train.loc[val_,'fold'] = int(fold)

# Train XGBoost ensembles by cross validation 
if model_architecture == "XGBoost":

    XGB_models = []
    scores = []

    XGBparams = {'lambda': 0.00017826413308577333,
        'alpha': 1.6906439156282021e-06,
        'min_child_weight': 12.62231594443773,
        'max_depth': 12,
        'gamma': 5,
        'subsample': 0.8749869503982316,
        'colsample_bytree': 0.920081529416236,
        'max_leaves': 87,
        'device':'gpu'}

    '''XGBparams = {'lambda': 0.16013242438368455,
        'alpha': 2.59e-07,
        'min_child_weight': 11.515526188745607,
        'max_depth': 9,
        'gamma': 2,
        'subsample': 0.8266374812481546,
        'colsample_bytree': 0.5786151538882082,
        'max_leaves': 51,
        'device':'gpu'}'''

    for fold in range(n_splits):

        df_train = train[train['fold'] != fold].reset_index(drop=True)
        df_val = train[train['fold'] == fold].reset_index(drop=True)

        X_train = df_train.drop(['id','p_num','time','fold','bg+1:00'], axis=1)
        X_val = df_val.drop(['id','p_num','time','fold','bg+1:00'], axis=1)
        y_train = df_train['bg+1:00']
        y_val = df_val['bg+1:00']

        model = XGB_VotingRegressor(3,XGBparams)
        dtrain = xgb.DMatrix(X_train, label= y_train)
        dval = xgb.DMatrix(X_val)
        model.fit(dtrain)
        score = root_mean_squared_error(y_val, model.predict(dval))

        XGB_models.append(model)
        scores.append(score)

    print(scores)
    print(np.mean(scores))

# Train Catboost ensembles by cross validation
if model_architecture == "CatBoost":

    CBst_models = []
    scores = []

    CBstparams = {'task_type':'GPU',
                'loss_function': 'RMSE',
                'depth': 5,
                'l2_leaf_reg': 7.959120286912168,
                'bagging_temperature': 1.5200475414672314,
                'random_strength': 1.3172524386382656,
                'min_data_in_leaf': 4}

    for fold in range(n_splits):

        df_train = train[train['fold'] != fold].reset_index(drop=True)
        df_val = train[train['fold'] == fold].reset_index(drop=True)

        X_train = df_train.drop(['id','p_num','time','fold','bg+1:00'], axis=1)
        X_val = df_val.drop(['id','p_num','time','fold','bg+1:00'], axis=1)
        y_train = df_train['bg+1:00']
        y_val = df_val['bg+1:00']

        model = CatBst_Voting_Regressor(3,CBstparams)
        dtrain = Pool(X_train, label= y_train)
        dval = Pool(X_val)
        model.fit(dtrain)
        score = root_mean_squared_error(y_val, model.predict(dval))

        CBst_models.append(model)
        scores.append(score)

    print(scores)
    print(np.mean(scores))

# save model
if save_model:
    os.makedirs(MODEL_PATH / model_name, exist_ok=True)
    model.save_model(MODEL_PATH / model_name)
