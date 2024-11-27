# libs
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error
from model_wrappers import XGB_VotingRegressor, CatBst_VotingRegressor, LGBM_VotingRegressor, XGB_Stacking_Regressor

# paths
DATA_PATH = Path(__file__).parents[2] / 'Data/Extended_Data'
TRAIN_DATA_FILE = DATA_PATH / 'extended_train.csv'
TEST_DATA_FILE = DATA_PATH / 'extended_test.csv'
MODEL_PATH = Path(__file__).parents[2] / 'Models'
SUBMISSION_PATH = Path(__file__).parents[2] / 'submissions/Ensemble'

# constants
models_dict = {'xgb': ['XGBoost', 'EDx12_FEv4_TW1h_RW5m_3VR_XGB_CV'],
               'CatB': ['CatBoost', 'EDx12_FEv4_TW1h_RW5m_3VR_CatB_CV'],
               'LGBM': ['LGBM', 'EDx12_FEv4_TW1h_RW5m_3VR_LGBM_CV']}

submission_name = 'SR_XGB_CatB_LGBM_EDx12_FEv4_TW1h_RW5m_3VR_CV.csv'
n_splits = 9
validation_fold = 7
validate = False
make_submission = True

# load data
train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)

# get splits grouped by participants
gkf = GroupKFold(n_splits)

for fold, (_, val_) in enumerate(gkf.split(X=train, y=train['bg+1:00'], groups=train['p_num'])):

    train.loc[val_,'fold'] = int(fold)

train_data = train[train['fold']!=validation_fold].reset_index(drop=True)
val_data = train[train['fold']==validation_fold].reset_index(drop=True)

# get train, validation and test datasets
X_full_train = train.drop(['id','p_num','time','fold','bg+1:00'], axis=1)
X_train = train_data.drop(['id','p_num','time','fold','bg+1:00'], axis=1)
X_val = val_data.drop(['id','p_num','time','fold','bg+1:00'], axis=1)
X_test = test.drop(['id','p_num','time','bg+1:00'], axis=1)
y_full_train = train['bg+1:00']
y_train = train_data['bg+1:00']
y_val = val_data['bg+1:00']

# get base models
estimators = []

for name, models in models_dict.items():

    architecture = models[0]
    file_name = models[1]

    for fold in range(n_splits):

        fold_file_name = file_name + f'_fold_{fold}'

        if architecture == 'XGBoost':
            estimator = XGB_VotingRegressor.load(MODEL_PATH/file_name/fold_file_name)

        elif architecture == 'CatBoost':
            estimator = CatBst_VotingRegressor.load(MODEL_PATH/file_name/fold_file_name)

        elif architecture == 'LGBM':
            estimator = LGBM_VotingRegressor.load(MODEL_PATH/file_name/fold_file_name)

        estimators.append(estimator)

# meta-learner
meta_learner_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'device': 'gpu'
        }

stacking_regressor = XGB_Stacking_Regressor(meta_learner_params, *estimators)

if validate:
    stacking_regressor.fit_meta(X_train, y_train)
else:
    stacking_regressor.fit_meta(X_full_train, y_full_train)

# validation
if validate:

    val_predictions = stacking_regressor.predict(X_val)
    score = root_mean_squared_error(y_val, val_predictions)
    print(score)

# predictions
if make_submission:

    submission_file = SUBMISSION_PATH / submission_name
    final_predictions = stacking_regressor.predict(X_test)
    ids = test['id']
    submissions = pd.DataFrame({'id':ids,'bg+1:00':final_predictions})
    submissions.to_csv(submission_file, index=False)
