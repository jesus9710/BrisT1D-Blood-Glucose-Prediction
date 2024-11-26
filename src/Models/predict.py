# libs
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
from catboost import Pool
from model_wrappers import XGB_VotingRegressor, CatBst_VotingRegressor, LGBM_VotingRegressor

# paths
DATA_PATH = Path(__file__).parents[2] / 'Data/Extended_Data'
TEST_DATA_FILE = DATA_PATH / 'extended_test.csv'
PREDICTION_PATH = Path(__file__).parents[2] / 'predictions'
MODEL_PATH = Path(__file__).parents[2] / 'Models'

# constants
model_architecture = "LGBM"
model_name = 'EDx1_FEv4_TW1h_RW5m_3VR_LGBM_CV'
submission_name = 'EDx1_FEv4_TW1h_RW5m_3VR_LGBM_CV.csv'
n_splits = 9
make_submission = True

# load data
test = pd.read_csv(TEST_DATA_FILE)
X_test = test.drop(['id','p_num','time','bg+1:00'], axis=1)


predictions = []

for fold in range(n_splits):

    model_name_fold = model_name + f'_fold_{fold}'

    if model_architecture == "XGBoost":
        dtest = xgb.DMatrix(X_test)
        model = XGB_VotingRegressor.load(MODEL_PATH/model_name/model_name_fold)

    elif model_architecture == "CatBoost":
        dtest = Pool(X_test)
        model = CatBst_VotingRegressor.load(MODEL_PATH/model_name/model_name_fold)

    elif model_architecture == "LGBM":
        dtest = X_test
        model = LGBM_VotingRegressor.load(MODEL_PATH/model_name/model_name_fold)

    prediction = model.predict(dtest)

    predictions.append(prediction)

# ensemble predictions
final_predictions = np.mean(predictions, axis=0)

# submission
if make_submission:
    submission_file = PREDICTION_PATH / submission_name

    ids = test['id']
    submissions = pd.DataFrame({'id':ids,'bg+1:00':final_predictions})
    submissions.to_csv(submission_file, index=False)
