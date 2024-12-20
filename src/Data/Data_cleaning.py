# libs
from pathlib import Path
import pandas as pd

from utils import *

# paths
TRAIN_DATA_PATH = Path(__file__).parents[2] / 'Data/train.csv'
TEST_DATA_PATH = Path(__file__).parents[2] / 'Data/test.csv'
CLEAN_DATA_PATH = Path(__file__).parents[2] / 'Data/Clean_Data'
EXPANDED_TRAIN_DATA_PATH = Path(__file__).parents[2] / 'Data/Raw_Extended_Data/raw_expanded_train.csv'
EXPANDED_TEST_DATA_PATH = Path(__file__).parents[2] / 'Data/Raw_Extended_Data/raw_expanded_test.csv'

# constants
time_window = 4 # past hours to take into account for predictions
time_len = 1 # length of culsters of 5-minute readings used to calculate an aggregation function
use_expanded_raw_data = True

# load data
if use_expanded_raw_data:
    train_raw_data = pd.read_csv(EXPANDED_TRAIN_DATA_PATH)
    test_raw_data = pd.read_csv(EXPANDED_TEST_DATA_PATH)

else:
    train_raw_data = pd.read_csv(TRAIN_DATA_PATH)
    test_raw_data = pd.read_csv(TEST_DATA_PATH)

raw_dataset = pd.concat([train_raw_data, test_raw_data], axis=0).reset_index(drop=True)
train_index = range(0, train_raw_data.shape[0])
test_index = range(train_raw_data.shape[0], raw_dataset.shape[0])

# dictionaries
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
                         'Weights':0,'Running':3,
                         'CoreTraining':0, 'Cycling':3,
                         'None':0}

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
                         'Weights':3,'Running':0,
                         'CoreTraining':1, 'Cycling':0,
                         'None':0}

# reduce time window used for predictions
reduced_data = reduce_time_window(raw_dataset, time_window)

# encode activity columns
columns = list(reduced_data.columns)
activity_columns = [col for col in columns if re.search(r'activity-.*', col)]

data_with_scores = encode_activity_columns(reduced_data, aerobic_dict, anaerobic_dict)

# aggregate time series (e.g. for time_len=2, each columns represents 10 minutes, with aggregated values from the original 5-minute columns)
expanded_time_data = aggregate_time_series(data_with_scores, time_len, skipna=False)

# drop columns
carb_columns = [col for col in expanded_time_data.columns if re.search('carbs_.*', col)]
to_drop = carb_columns

filled_data = expanded_time_data.drop(to_drop, axis=1)

# impute missing values for bg
columns = list(filled_data.columns)
bg_columns = [col for col in columns if re.search(r'bg_.*', col)]
hr_columns = [col for col in columns if re.search(r'hr_.*', col)]
cals_columns = [col for col in columns if re.search(r'cals_.*', col)]

filled_data = mean_ffill_bfill_imputation(filled_data, bg_columns)

'''to_fill = [bg_columns, hr_columns, cals_columns]

for group in to_fill:
    filled_data[group] = filled_data[group].interpolate(axis=1).ffill(axis=1).bfill(axis=1)
'''
# get bg columns grouped by person and time
bg_columns = [col for col in expanded_time_data.columns if re.search('bg_.*', col)]
mean_by_per_and_hour = filled_data.groupby(by=['p_num','time'])[bg_columns].mean()

# fill missing values in the dataset grouped by person and time
p_null_list = get_participants_with_null_values(mean_by_per_and_hour)

for p_num in p_null_list:
    
    # fill missing values by interpolation
    mean_by_per_and_hour = interpolate_and_fill_rows(p_num, mean_by_per_and_hour, n_iters = 10)

# interpolation through columns
p_null_list = get_participants_with_null_values(mean_by_per_and_hour)

for p_num in p_null_list:

    mean_by_per_and_hour = interpolate_and_fill_columns(p_num, mean_by_per_and_hour, bg_columns)

# use dataset grouped by person and time to impute missing values in the original dataset
merged_data = filled_data.merge(mean_by_per_and_hour, on=['p_num', 'time'], suffixes=('', '_mean'))
for col in bg_columns:
    filled_data[col] = filled_data[col].fillna(merged_data[col + '_mean'])

# impute missing values for steps, insulin and cals columns with 0

'''for group in to_fill:

    grouped_dataset = filled_data.groupby(by=['p_num','time'])[group].mean()
    grouped_dataset = grouped_dataset.interpolate(axis=1).ffill(axis=1).bfill(axis=1)
    grouped_dataset = grouped_dataset.interpolate(axis=0).ffill(axis=0).bfill(axis=0)
    merged_dataset = filled_data.merge(grouped_dataset, on=['p_num', 'time'], suffixes=('', '_mean'))
    
    for col in group:
        filled_data[col] = filled_data[col].fillna(merged_dataset[col + '_mean'])
'''
to_fill = [col for col in columns if re.search(r'steps_.*|insulin_.*|cals_.*|hr_.*',col)]

filled_data[to_fill] = filled_data[to_fill].fillna(value=-1)

# export to csv
clean_train_data = filled_data.iloc[train_index,:]
clean_test_data = filled_data.iloc[test_index,:]

clean_train_data.to_csv(CLEAN_DATA_PATH / 'clean_train.csv', index = False)
clean_test_data.to_csv(CLEAN_DATA_PATH / 'clean_test.csv', index = False)