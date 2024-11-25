#%% libs
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import re

#%% paths
CLEAN_DATA_PATH = Path(__file__).parents[2] / 'Data/Clean_Data'
TRAIN_DATA_FILE = CLEAN_DATA_PATH / 'clean_train.csv'
TEST_DATA_FILE = CLEAN_DATA_PATH / 'clean_test.csv'
EXTENDED_DATA_PATH = Path(__file__).parents[2] / 'Data/Extended_Data'

#%% load data
train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)
dataset = pd.concat([train,test], axis=0).reset_index(drop=True)

train_index = range(0, train.shape[0])
test_index = range(train.shape[0], dataset.shape[0])

#%% feature engineering function
def feature_eng(df):

    bg_columns = [col for col in df.columns if re.search('bg_.*', col)]
    ins_columns = [col for col in df.columns if re.search('insulin_.*', col)]
    hr_columns = [col for col in df.columns if re.search('hr_.*', col)]
    step_columns = [col for col in df.columns if re.search('steps_.*', col)]
    cal_columns = [col for col in df.columns if re.search('cals_.*', col)]

    mean_by_per_and_hour_bg = df.groupby(by=['p_num','time'])[bg_columns].mean()
    mean_by_per_and_hour_ins = df.groupby(by=['p_num','time'])[ins_columns].mean()
    mean_by_per_and_hour_step = df.groupby(by=['p_num','time'])[step_columns].mean()
    mean_by_per_and_hour_cal = df.groupby(by=['p_num','time'])[cal_columns].mean()

    merged_data_bg = df.merge(mean_by_per_and_hour_bg, on=['p_num','time'], suffixes=['','_mean']).copy()
    merged_data_ins = df.merge(mean_by_per_and_hour_ins, on=['p_num','time'], suffixes=['','_mean']).copy()
    merged_data_step = df.merge(mean_by_per_and_hour_step, on=['p_num','time'], suffixes=['','_mean']).copy()
    merged_data_cal = df.merge(mean_by_per_and_hour_cal, on=['p_num','time'], suffixes=['','_mean']).copy()

    hours = pd.to_datetime(df['time'], format='%H:%M:%S').dt.hour
    minutes = pd.to_datetime(df['time'], format='%H:%M:%S').dt.minute

    # time transformations variables
    df['time_hour_sin'] = np.sin(2 * np.pi * hours / 24)
    df['time_hour_cos'] = np.cos(2 * np.pi * hours / 24)
    df['time_min_sin'] = np.sin(2 * np.pi * minutes / 24)
    df['time_min_cos'] = np.cos(2 * np.pi * minutes / 24)

    # difference between the variable at the current point in time and the average value per participant at the current point in time
    df['c_bg_mean_diff'] = df[bg_columns[-1]] - merged_data_bg[bg_columns[-1]+ '_mean']
    # df['c_ins_mean_diff'] = df[ins_columns[-1]] - merged_data_ins[ins_columns[-1]+ '_mean']
    df['c_step_mean_diff'] = df[step_columns[-1]] - merged_data_step[step_columns[-1]+ '_mean']
    df['c_cal_mean_diff'] = df[cal_columns[-1]] - merged_data_cal[cal_columns[-1]+ '_mean']

    # ratios between the variable at the current point in time and the average value per participant at the current point in time
    df['c_ins_mean_ratio'] = df[ins_columns[-1]] / (merged_data_ins[ins_columns[-1]+ '_mean'] + 0.01)
    df['c_step_mean_ratio'] = df[step_columns[-1]] / (merged_data_step[step_columns[-1]+ '_mean'] + 0.01)
    df['c_cal_mean_ratio'] = df[cal_columns[-1]] / (merged_data_cal[cal_columns[-1]+ '_mean'] + 0.01)
    #### df['c_bg_mean_ratio'] = df[bg_columns[-1]] / (merged_data_bg[bg_columns[-1]+ '_mean'] + 0.01)

    # current time
    df['mins_since_12AM'] = ((pd.to_datetime(df['time'], format='%H:%M:%S') - datetime.strptime('00:00:00', '%H:%M:%S')).dt.total_seconds() / 60).astype('int')
    
    # sum of groups of variables along axis one
    df['total_steps'] = df[step_columns].sum(axis=1)
    #### df['total_ins'] = df[ins_columns].sum(axis=1)

    # change ratios (probar a aumentar la diferencia de tiempo. Valores cercanos tienen alta correlación)
    df['bg_change_ratio'] = df[bg_columns[-1]] / (df[bg_columns[-2]] + 0.01) # alta importancia en XGBoost (analizar la importancia de manera experimental)
    df['ins_change_ratio'] = df[ins_columns[-1]] / (df[ins_columns[-2]] + 0.01)
    df['cals_change_ratio'] = df[cal_columns[-1]] / (df[cal_columns[-2]] + 0.01) # was dropped

    # change ratios with 6 steps time delay
    df['bg_change_ratio_d'] = df[bg_columns[-1]] / (df[bg_columns[-6]] + 0.01) # alta importancia en XGBoost (analizar la importancia de manera experimental)
    df['ins_change_ratio_d'] = df[ins_columns[-1]] / (df[ins_columns[-6]] + 0.01)
    df['cals_change_ratio_d'] = df[cal_columns[-1]] / (df[cal_columns[-6]] + 0.01) # was dropped

    # approximate of the derivative
    df['ins_change_diff'] = (df[ins_columns[-1]] - df[ins_columns[-2]])
    df['hr_change_diff'] = (df[hr_columns[-1]] - df[hr_columns[-2]])
    df['cals_change_diff'] = (df[cal_columns[-1]] - df[cal_columns[-2]]) # was dropped
    df['bg_change_diff'] = (df[bg_columns[-1]] - df[bg_columns[-2]]) # was dropped

    # approximate of the derivative with 6 steps time delay
    df['ins_change_diff_d'] = (df[ins_columns[-1]] - df[ins_columns[-6]])
    df['hr_change_diff_d'] = (df[hr_columns[-1]] - df[hr_columns[-6]])
    #### df['cals_change_diff'] = (df[cal_columns[-1]] - df[cal_columns[-2]])
    #### df['bg_change_diff'] = (df[bg_columns[-1]] - df[bg_columns[-2]])

    # variable statistics in the last 6 steps
    df['bg_last_6steps_diff'] = df[bg_columns[-6:]].max(axis=1) - df[bg_columns[-6:]].min(axis=1)
    df['ins_last_6steps_diff'] = df[ins_columns[-6:]].max(axis=1) - df[ins_columns[-6:]].min(axis=1)
    # df['bg_last_6steps_max'] = df[bg_columns[-6:]].max(axis=1)
    # df['ins_last_6steps_max'] = df[ins_columns[-6:]].max(axis=1)
    # df['bg_last_6steps_min'] = df[bg_columns[-6:]].min(axis=1)
    # df['ins_last_6steps_min'] = df[ins_columns[-6:]].min(axis=1)
    # df['bg_last_6steps_std'] = df[bg_columns[-6:]].std(axis=1)
    df['steps_last_6steps_total'] = df[step_columns[-6:]].sum(axis=1)
    #### df['ins_last_6steps_std'] = df[ins_columns[-6:]].std(axis=1)
    #### df['bg_last_6steps_mean'] = df[bg_columns[-6:]].mean(axis=1)
    #### df['ins_last_6steps_mean'] = df[ins_columns[-6:]].mean(axis=1)

    # variable interactions
    df['total_cals_per_steps'] = df[cal_columns].sum(axis=1) / (df[step_columns].sum(axis=1) + 0.01)
    df['last_cals_per_steps'] = df[cal_columns[-1]] / (df[step_columns[-1]] + 0.01)
    #### df['total_ins_per_bg'] = df[ins_columns].sum(axis=1) / (df[bg_columns].sum(axis=1) + 0.01)
    #### df['last_6_cals_per_step_sum'] = df[cal_columns[-6:]].sum(axis=1) / (df[step_columns[-6:]].sum(axis=1) + 0.01)

    # current time variable interactions
    column_groups = [bg_columns[-1],ins_columns[-1],hr_columns[-1], step_columns[-1], cal_columns[-1]]

    for idx, col_i in enumerate(column_groups):
        for col_j in column_groups[(idx+1):]:
            
            # df[col_i.split('_')[0] + '+' + col_j.split('_')[0]] = df[col_i] + df[col_j]
            # df[col_i.split('_')[0] + '-' + col_j.split('_')[0]] = df[col_i] - df[col_j]
            df[col_i.split('_')[0] + '*' + col_j.split('_')[0]] = df[col_i] * df[col_j]
            df[col_i.split('_')[0] + '/' + col_j.split('_')[0]] = df[col_i] / (df[col_j] + 0.001)

    # remove odd-numbered bg columns to address high pairwise correlations.
    columns_to_drop = [col for idx, col in enumerate(bg_columns) if idx % 2 == 0]
    df = df.drop(columns_to_drop, axis=1)

    '''normalized_cals = (df[cal_columns] - df[cal_columns].mean(axis=0)) / df[cal_columns].std(axis=0)
    normalized_steps = (df[step_columns] - df[step_columns].mean(axis=0)) / df[step_columns].std(axis=0)
    normalized_aerobic_score = (df['Aerobic_score'] - df['Aerobic_score'].mean()) / df['Aerobic_score'].std()
    normalized_anaerobic_score = (df['Anaerobic_score'] - df['Anaerobic_score'].mean()) / df['Anaerobic_score'].std()

    effort = (normalized_steps.sum(axis=1) + normalized_aerobic_score + normalized_anaerobic_score)

    df['total_cal_per_effort'] = (normalized_cals.sum(axis=1) - effort) / (effort+0.001)
    '''

    return df

# apply feature engineering
extended_dataset = feature_eng(dataset)

#%% obtain train and test data
extended_train = extended_dataset.iloc[train_index,:].reset_index(drop=True)
extended_test = extended_dataset.iloc[test_index,:].reset_index(drop=True)

# export csv
extended_train.to_csv(EXTENDED_DATA_PATH / 'extended_train.csv',index=False)
extended_test.to_csv(EXTENDED_DATA_PATH / 'extended_test.csv',index=False)
# %%
