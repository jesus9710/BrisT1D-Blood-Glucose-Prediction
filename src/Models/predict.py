# libs
import pandas as pd
from pathlib import Path
import xgboost as xgb
from Models.model_wrappers import XGB_VotingRegressor, CatBst_Voting_Regressor

# paths
DATA_PATH = Path(__file__).parents[2] / 'Data/Extended_Data'
TEST_DATA_FILE = DATA_PATH / 'extended_test.csv'
SUBMISSION_PATH = Path(__file__).parents[2] / 'submissions'
MODEL_PATH = Path(__file__).parents[2] / 'Models'

# constants
model_architecture = "XGBoost"
model_name = 'EDx1_Some_FE_5mins_3VR_XGBoost_optimized_CV_v2'
submission_name = 'EDx1_Some_FE_5mins_3VR_XGBoost_optimized_CV_v2.csv'
n_splits = 9
make_submission = True

# load data
test = pd.read_csv(TEST_DATA_FILE)
X_test = test.drop(['id','p_num','time','bg+1:00'], axis=1)
dtest = xgb.DMatrix(X_test)

# load model
if model_architecture == "XGBoost":

    model = XGB_VotingRegressor.load(MODEL_PATH/model_name)

elif model_architecture == "CatBoost":

    model = CatBst_Voting_Regressor.load(MODEL_PATH/model_name)

# predictions
predictions = model.predict(dtest)

# submission
if make_submission:
    submission_file = SUBMISSION_PATH / submission_name

    ids = test['id']
    submissions = pd.DataFrame({'id':ids,'bg+1:00':predictions})
    submissions.to_csv(submission_file, index=False)
