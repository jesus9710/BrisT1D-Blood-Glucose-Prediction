#%% Libs
from pathlib import Path
import pandas as pd

from utils import *

#%% Paths
TRAIN_DATA_PATH = Path(__file__).parents[2] / 'Data/train.csv'
TEST_DATA_PATH = Path(__file__).parents[2] / 'Data/test.csv'
CLEAN_DATA_PATH = Path(__file__).parents[2] / 'Data/Clean_Data'

#%% Constants

time_window = 4 # past hours to take into account for predictions
time_len = 1 # length of culsters of 5-minute readings used to calculate an aggregation function

#%% Load data
train_raw_data = pd.read_csv(TRAIN_DATA_PATH)
test_raw_data = pd.read_csv(TEST_DATA_PATH)

raw_dataset = pd.concat([train_raw_data, test_raw_data], axis=0).reset_index(drop=True)
train_index = range(0, train_raw_data.shape[0])
test_index = range(train_raw_data.shape[0], raw_dataset.shape[0])

#%% Dictionaries

aerobic_dict = {'Indoor climbing':0,'Run':3,
                         'Strength training':0,'Swim':2,
                         'Bike':3, 'Dancing':1,
                         'Stairclimber':1, 'Spinning':3,
                         'Walking':1,'HIIT':0,
                         'Outdoor Bike':3,'Walk':1,
                         'Aerobic Workout':3,'Tennis':2,
                         'Workout':0,'Hike':1,
                         'Zumba':2,'Sport':2,
                         'Yoga':0,'Swimming':2,
                         'Weights':0,'Running':3}

anaerobic_dict = {'Indoor climbing':2,'Run':0,
                         'Strength training':3,'Swim':0,
                         'Bike':0, 'Dancing':0,
                         'Stairclimber':2, 'Spinning':0,
                         'Walking':0,'HIIT':3,
                         'Outdoor Bike':0,'Walk':0,
                         'Aerobic Workout':0,'Tennis':2,
                         'Workout':3,'Hike':0,
                         'Zumba':0,'Sport':1,
                         'Yoga':0,'Swimming':0,
                         'Weights':3,'Running':0}

#%% encode activity columns
data_with_scores = encode_activity_columns(raw_dataset, aerobic_dict, anaerobic_dict)

#%% reduce time window used to predict and expand time between readings
reduced_data = reduce_time_window(data_with_scores, time_window)
expanded_time_data, new_column_cluster = expand_time_clusters(reduced_data, time_len)

#%% drop columns
hr_columns = [col for col in expanded_time_data.columns if re.search('hr_.*', col)]
filled_data = expanded_time_data.drop(hr_columns, axis=1)

#%% get dataset grouped by person and time
bg_columns = [col for col in expanded_time_data.columns if re.search('bg_.*', col)]
mean_by_per_and_hour = filled_data.groupby(by=['p_num','time'])[bg_columns].mean()

#%% fill missing values in the dataset grouped by person and time
p_null_list = get_participants_with_null_values(mean_by_per_and_hour)

for p_num in p_null_list:
    
    # fill missing values by interpolation
    mean_by_per_and_hour = interpolate_and_fill_rows(p_num, mean_by_per_and_hour, n_iters = 10)

#%% interpolation through columns

p_null_list = get_participants_with_null_values(mean_by_per_and_hour)

for p_num in p_null_list:

    mean_by_per_and_hour = interpolate_and_fill_columns(p_num, mean_by_per_and_hour, bg_columns)

#%% use dataset grouped by person and time to impute missing values in the original dataset
merged_data = filled_data.merge(mean_by_per_and_hour, on=['p_num', 'time'], suffixes=('', '_mean'))
for col in bg_columns:
    filled_data[col] = filled_data[col].fillna(merged_data[col + '_mean'])

# export to csv
clean_train_data = filled_data.iloc[train_index,:]
clean_test_data = filled_data.iloc[test_index,:]

clean_train_data.to_csv(CLEAN_DATA_PATH / 'clean_train.csv', index = False)
clean_test_data.to_csv(CLEAN_DATA_PATH / 'clean_test.csv', index = False)