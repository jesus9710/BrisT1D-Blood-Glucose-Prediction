# libs
import pandas as pd
from pathlib import Path
import xgboost as xgb
from catboost import Pool
from model_wrappers import XGB_VotingRegressor, CatBst_Voting_Regressor

# paths
DATA_PATH = Path(__file__).parents[2] / 'Data/Extended_Data'
TEST_DATA_FILE = DATA_PATH / 'extended_test.csv'
PREDICTION_PATH = Path(__file__).parents[2] / 'predictions'
MODEL_PATH = Path(__file__).parents[2] / 'Models'

# constants
model_architecture = "XGBoost"
model_name = 'EDx1_FEv3_5mins_3VR_XGBoost_optimized_CV'
submission_name = 'EDx1_FEv3_5m_3VR_XGBoost_opt_CV.csv'
n_splits = 9
make_submission = True

# load data
test = pd.read_csv(TEST_DATA_FILE)
X_test = test.drop(['id','p_num','time','bg+1:00'], axis=1)

# load model
if model_architecture == "XGBoost":
    dtest = xgb.DMatrix(X_test)
    model = XGB_VotingRegressor.load(MODEL_PATH/model_name)

elif model_architecture == "CatBoost":
    dtest = Pool(X_test)
    model = CatBst_Voting_Regressor.load(MODEL_PATH/model_name)

# predictions
predictions = model.predict(dtest)

# submission
if make_submission:
    submission_file = PREDICTION_PATH / submission_name

    ids = test['id']
    submissions = pd.DataFrame({'id':ids,'bg+1:00':predictions})
    submissions.to_csv(submission_file, index=False)
