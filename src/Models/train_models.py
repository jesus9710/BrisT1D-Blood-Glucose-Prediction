# %% libs
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import DMatrix
from catboost import Pool
from lightgbm import Dataset as lgb_data
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error
from model_wrappers import XGB_VotingRegressor, CatBst_VotingRegressor, LGBM_VotingRegressor
import os

# %% paths
DATA_PATH = Path(__file__).parents[2] / 'Data/Extended_Data'
TRAIN_DATA_FILE = DATA_PATH / 'extended_train.csv'
MODEL_PATH = Path(__file__).parents[2] / 'Models'

# %% constants
model_architecture = "CatBoost"
model_name = 'EDx1_FEv4_5m_3VR_CatB_opt_CV'
n_splits = 9
save_model = True

# %% load data
train = pd.read_csv(TRAIN_DATA_FILE)

# %% get splits grouped by participants
gkf = GroupKFold(n_splits)

for fold, (_, val_) in enumerate(gkf.split(X=train, y=train['bg+1:00'], groups=train['p_num'])):

    train.loc[val_,'fold'] = int(fold)

# %% Train XGBoost ensembles by cross validation 
if model_architecture == "XGBoost":

    XGB_models = []
    scores = []

    XGBparams = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'lambda': 1.1411668887698974e-05,
        'alpha': 1.6975854374364468e-07,
        'min_child_weight': 15.248055302836686,
        'max_depth': 2,
        'gamma': 2,
        'subsample': 0.9387419814817861,
        'colsample_bytree': 0.9477822297336089,
        'max_leaves': 113,
        'device': 'gpu'
        }
    
    XGBfit_params = {'num_boost_round': 337,
                     'early_stopping_rounds':50,
                     'verbose_eval':False}

    for fold in range(n_splits):

        df_train = train[train['fold'] != fold].reset_index(drop=True)
        df_val = train[train['fold'] == fold].reset_index(drop=True)

        X_train = df_train.drop(['id','p_num','time','fold','bg+1:00'], axis=1)
        X_val = df_val.drop(['id','p_num','time','fold','bg+1:00'], axis=1)
        y_train = df_train['bg+1:00']
        y_val = df_val['bg+1:00']

        dtrain = DMatrix(X_train, label = y_train)
        dval = DMatrix(X_val, label = y_val)
        XGBfit_params.update({'evals' : [(dval,'validation')]})

        model = XGB_VotingRegressor(3, XGBparams)
        model.fit(dtrain, XGBfit_params)
        score = root_mean_squared_error(y_val, model.predict(dval))

        XGB_models.append(model)
        scores.append(score)

    print(scores)
    print(np.mean(scores))

# %% Train Catboost ensembles by cross validation
if model_architecture == "CatBoost":

    CBst_models = []
    scores = []

    CBstparams = {
        'task_type':'GPU',
        'loss_function': 'RMSE',
        'depth': 5,
        'l2_leaf_reg': 4.179773551123609,
        'bagging_temperature': 1.356664543494321,
        'random_strength': 0.057645267420365354,
        'min_data_in_leaf': 10,
        'verbose' : False
        }

    CBstfit_params = {'early_stopping_rounds' : 30}

    for fold in range(n_splits):

        df_train = train[train['fold'] != fold].reset_index(drop=True)
        df_val = train[train['fold'] == fold].reset_index(drop=True)

        X_train = df_train.drop(['id','p_num','time','fold','bg+1:00'], axis=1)
        X_val = df_val.drop(['id','p_num','time','fold','bg+1:00'], axis=1)
        y_train = df_train['bg+1:00']
        y_val = df_val['bg+1:00']

        dtrain = Pool(X_train, label = y_train)
        dval = Pool(X_val, label = y_val)
        CBstfit_params.update({'eval_set' : dval})

        model = CatBst_VotingRegressor(3, CBstparams)
        model.fit(dtrain, CBstfit_params)
        score = root_mean_squared_error(y_val, model.predict(dval))

        CBst_models.append(model)
        scores.append(score)

    print(scores)
    print(np.mean(scores))

# %% Train LGBM ensembles by cross validation
if model_architecture == "LGBM":

    LGBM_models = []
    scores = []

    LGBMparams = {'verbosity': 0}

    for fold in range(n_splits):

        df_train = train[train['fold'] != fold].reset_index(drop=True)
        df_val = train[train['fold'] == fold].reset_index(drop=True)

        X_train = df_train.drop(['id','p_num','time','fold','bg+1:00'], axis=1)
        X_val = df_val.drop(['id','p_num','time','fold','bg+1:00'], axis=1)
        y_train = df_train['bg+1:00']
        y_val = df_val['bg+1:00']

        model = LGBM_VotingRegressor(3, LGBMparams)
        dtrain = lgb_data(X_train, label=y_train)
        dval = lgb_data(X_val, label=y_val)

        model.fit(dtrain)
        score = root_mean_squared_error(y_val, model.predict(X_val))

        LGBM_models.append(model)
        scores.append(score)

    print(scores)
    print(np.mean(scores))

# %% save model
if save_model:
    os.makedirs(MODEL_PATH / model_name, exist_ok=True)
    model.save_model(MODEL_PATH / model_name)
