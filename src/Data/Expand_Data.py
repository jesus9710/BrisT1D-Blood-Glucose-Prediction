# libs
import pandas as pd
from pathlib import Path
from utils import expand_data

# paths
TRAIN_DATA_PATH = Path(__file__).parents[2] / 'Data/train.csv'
TEST_DATA_PATH = Path(__file__).parents[2] / 'Data/test.csv'
EXPANDED_DATA_PATH = Path(__file__).parents[2] / 'Data/Raw_Extended_Data'

# constants
time_window = 4 # past hours to take into account for predictions
time_len = 1 # length of culsters of 5-minute readings used to calculate an aggregation function
expand_n_times = 12

# load data
train_raw_data = pd.read_csv(TRAIN_DATA_PATH)
test_raw_data = pd.read_csv(TEST_DATA_PATH)

raw_dataset = pd.concat([train_raw_data, test_raw_data], axis=0).reset_index(drop=True)

# expand train data
expanded_train = expand_data(raw_dataset, time_window, expand_n_times)
expanded_dataset = pd.concat([expanded_train, test_raw_data], axis=0).reset_index(drop=True)

train_index = range(0, expanded_train.shape[0])
test_index = range(expanded_train.shape[0], expanded_dataset.shape[0])

# export to csv
extended_train_data = expanded_dataset.iloc[train_index,:]
extended_test_data = expanded_dataset.iloc[test_index,:]

extended_train_data.to_csv(EXPANDED_DATA_PATH / 'raw_expanded_train.csv', index = False)
extended_test_data.to_csv(EXPANDED_DATA_PATH / 'raw_expanded_test.csv', index = False)
