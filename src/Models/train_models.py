#%% libs
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import VotingRegressor

#%% paths

DATA_PATH = Path(__file__).parents[2] / 'Data/Extended_Data'
TRAIN_DATA_FILE = DATA_PATH / 'extended_train.csv'
TEST_DATA_FILE = DATA_PATH / 'extended_test.csv'
SUBMISSION_PATH = Path(__file__).parents[2] / 'submissions'

#%% constants

submission_name = 'NoFE_10minsAGG_3VR_XGBoost_optimized_CV.csv'
n_splits = 9

#%%
train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)

# %% get splits grouped by participants
gkf = GroupKFold(n_splits)

for fold, (_, val_) in enumerate(gkf.split(X=train, y=train['bg+1:00'], groups=train['p_num'])):

    train.loc[val_,'fold'] = int(fold)

# %% Train XGBoost ensembles by cross validation 

XGB_models = []
scores = []

XGBparams = {'lambda': 0.00017826413308577333,
    'alpha': 1.6906439156282021e-06,
    'min_child_weight': 12.62231594443773,
    'max_depth': 12,
    'gamma': 5,
    'subsample': 0.8749869503982316,
    'colsample_bytree': 0.920081529416236,
    'max_leaves': 87}

for fold in range(n_splits):
    df_train = train[train['fold'] != fold].reset_index(drop=True)
    df_val = train[train['fold'] == fold].reset_index(drop=True)

    X_train = df_train.drop(['id','p_num','time','fold','bg+1:00'], axis=1)
    X_val = df_val.drop(['id','p_num','time','fold','bg+1:00'], axis=1)
    y_train = df_train['bg+1:00']
    y_val = df_val['bg+1:00']

    model = VotingRegressor([(f"XGB_{i}", XGBRegressor(random_state=i, **XGBparams)) for i in range(3)])
    model.fit(X_train, y_train)
    score = root_mean_squared_error(y_val, model.predict(X_val))

    XGB_models.append(model)
    scores.append(score)

print(scores)
print(np.mean(scores))

# %% get predictions

X_test = test.drop(['id','p_num','time','bg+1:00'], axis=1)

XGB_preds = np.mean([model.predict(X_test)[:] for model in XGB_models], 0)

# %% submission

submission_file = SUBMISSION_PATH / submission_name

ids = test['id']
submissions = pd.DataFrame({'id':ids,'bg+1:00':XGB_preds})
submissions.to_csv(submission_file, index=False)

# %%
