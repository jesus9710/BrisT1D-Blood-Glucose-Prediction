# libs
import pandas as pd
from pathlib import Path
from datetime import datetime
import re

# paths
CLEAN_DATA_PATH = Path(__file__).parents[2] / 'Data/Clean_Data'
TRAIN_DATA_FILE = CLEAN_DATA_PATH / 'clean_train.csv'
TEST_DATA_FILE = CLEAN_DATA_PATH / 'clean_test.csv'
EXTENDED_DATA_PATH = Path(__file__).parents[2] / 'Data/Extended_Data'

# load data
train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)
dataset = pd.concat([train,test], axis=0).reset_index(drop=True)

train_index = range(0, train.shape[0])
test_index = range(train.shape[0], dataset.shape[0])

# feature engineering function
def feature_eng(df):

    bg_columns = [col for col in df.columns if re.search('bg_.*', col)]
    mean_by_per_and_hour = df.groupby(by=['p_num','time'])[bg_columns].mean()
    merged_data = df.merge(mean_by_per_and_hour, on=['p_num','time'], suffixes=['','_mean'])


    df['mins_since_12AM'] = ((pd.to_datetime(df['time'], format='%H:%M:%S') - datetime.strptime('00:00:00', '%H:%M:%S')).dt.total_seconds() / 60).astype('int')
    df['c_bg_mean_diff'] = df[bg_columns[-1]] - merged_data[bg_columns[-1] + '_mean']

    return df

# import csv
extended_dataset = feature_eng(dataset)
extended_train = extended_dataset.iloc[train_index,:]
extended_test = extended_dataset.iloc[test_index,:]

extended_train.to_csv(EXTENDED_DATA_PATH / 'extended_train.csv',index=False)
extended_test.to_csv(EXTENDED_DATA_PATH / 'extended_test.csv',index=False)