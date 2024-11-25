#%% libs
import numpy as np
import pandas as pd
from pathlib import Path

# %% Paths

ROOT_PATH = Path(__file__).parents[2]
DATA_PATH = ROOT_PATH / 'Data/Extended_Data'
PREDICTION_PATH = ROOT_PATH / 'Predictions'
ENSEMBLE_SUBMISSION_PATH = ROOT_PATH / 'Submissions/Ensemble'
TEST_DATA_FILE = DATA_PATH / 'extended_test.csv'

#%% constants

submission_name = 'Ensemble_XGB_CatB_LGBM_1xED_FEv4_5m_opt.csv'

# %% spredictions to ensemble

prediction_list = sorted([file for file in PREDICTION_PATH.glob('*.csv')])

df_test = pd.read_csv(prediction_list[0], header=0, names=['id',f'pred_{0}'])

for idx, prediction in enumerate(prediction_list[1:]):

    df = pd.read_csv(prediction, header=0, names=['id',f'pred_{idx}'])
    df_test.merge(df, on='id', how='inner')

# %% Ensemble

df_test['bg+1:00'] = df_test.drop('id', axis=1).mean(axis=1)

df_test = df_test[['id','bg+1:00']]

# %% submission

df_test.to_csv(ENSEMBLE_SUBMISSION_PATH / submission_name, index=False)

# %%
