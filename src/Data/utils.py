import numpy as np
import pandas as pd
import re
from datetime import datetime
from datetime import timedelta

def anaerobic_time_function(x):
    return (1-1/(1+5*np.exp(-(x)*2)))

def aerobic_time_function(x):
    return (1-1/(1+8*np.exp(-(x)*1.5)))

def encode_activity_columns(df, aerobic_dict, anaerobic_dict, effort_dict = None):
    '''
    This function is used to encode the activity columns. In addition, two other columns representing anaerobic and aerobic scores will be generated.
    '''

    columns = list(df.columns)
    activity_columns = [col for col in columns if re.search(r'activity-.*', col)]

    data_with_scores = df.copy()

    # fill missing values with None, then they will be replaced by 0. It is logical to assume that these participant did not play sports.
    data_with_scores[activity_columns] = data_with_scores[activity_columns].fillna(value='None')

    aerobic_scores_dict = {}
    anaerobic_scores_dict = {}

    for col in activity_columns:

        # get hour from column name
        hour_str = col.split(sep='-')[1]

        hour_obj = datetime.strptime(hour_str, '%H:%M')
        hour = hour_obj.hour + (hour_obj.minute / 60)

        # get aerobic and anaerobic score
        aerobic_scores_dict['A-'+ hour_str] = data_with_scores[col].map(aerobic_dict).astype('float') * aerobic_time_function(hour)
        anaerobic_scores_dict['AN-'+ hour_str] = data_with_scores[col].map(anaerobic_dict).astype('float') * anaerobic_time_function(hour)

        # encode the original column
        if effort_dict:
            data_with_scores[col] = data_with_scores[col].map(effort_dict).astype('int')
    
    # if effort dict not include, original activity columns will be dropped
    if not(effort_dict):
        data_with_scores = data_with_scores.drop(activity_columns, axis=1)

    # obtain the sums of aerobic and anaerobic scores along time axis
    activity_scores_df = pd.concat([pd.DataFrame(aerobic_scores_dict),pd.DataFrame(anaerobic_scores_dict)], axis=1)

    aerobic_columns = aerobic_scores_dict.keys()
    anaerobic_columns = anaerobic_scores_dict.keys()

    sum_aerobic_scores_df = activity_scores_df[aerobic_columns].sum(axis=1).to_frame(name='Aerobic_score')
    sum_anaerobic_scores_df = activity_scores_df[anaerobic_columns].sum(axis=1).to_frame(name='Anaerobic_score')

    # add new columns
    data_with_scores = pd.concat([data_with_scores, sum_aerobic_scores_df, sum_anaerobic_scores_df], axis=1)

    return data_with_scores

def reduce_time_window(df, time_window):
    '''
    Function for reducing time window used for bg regression
    '''

    reduced_data = df.copy()
    columns = list(reduced_data.columns)
    column_filter = fr'-[{time_window}-9]:[0-9].*'

    # Delete columns older than (time_window) h
    delete_columns = [col for col in columns if re.search(column_filter, col)]
    reduced_data = reduced_data.drop(delete_columns, axis=1)

    return reduced_data

def generate_time_spans(column_names, hours):
    '''
    generate column names for each category in 5-minute intervals
    '''

    new_column_names = {}
    for name in column_names:
        h_list = []

        for h in range(hours):
            h_list += [f'{name}-{h}:0'+str(i) if i<10 else f'{name}-{h}:'+str(i) for i in range(0,60,5)]

        new_column_names.update({name:h_list})

    return new_column_names

def generate_time_spans(hours, name=''):
    '''
    generate column names for each category in 5-minute intervals
    '''
    h_list = []

    for h in range(hours):
        h_list += [f'{name}-{h}:0'+str(i) if i<10 else f'{name}-{h}:'+str(i) for i in range(0,60,5)]

    return h_list

def expand_data(df, time_window, n_times=1, one_hour = 12, n_hours=6):
    '''
    Get more data by reducing time window used for predictions
    '''

    features = ['bg','insulin','carbs','hr','steps','cals','activity']
    h_list = generate_time_spans(n_hours)
    time_columns = [col for col in df.columns if re.search(r'[0-9]:[0-9].*',col)]
    new_times = h_list[:(time_window*one_hour)]

    new_dfs = []

    for i in range(n_times):
        # define time ranges
        old_start = -(time_window * one_hour + i)
        old_end = -i if i else None
        old_times = h_list[old_start:old_end]
        new_label_time = h_list[old_start - one_hour]

        # generate old and new columns
        new_columns = [f'{feat}{lag}' for feat in features for lag in new_times]
        old_columns = [f'{feat}{lag}' for feat in features for lag in old_times]
        new_label = f'bg{new_label_time}'

        # get renamed dataframe
        renamed_data = pd.DataFrame(
            df[old_columns + [new_label]].values,
            columns= new_columns + ['bg+1:00']
        )

        # create new dataframe with updated times
        new_df = pd.concat([df.drop(time_columns, axis=1), renamed_data], axis=1)
        new_df['time'] = (pd.to_datetime(new_df['time'],format='%H:%M:%S') \
                          - timedelta(hours=n_hours - time_window)).dt.time.astype('str')
        new_df['id'] = new_df['id'] + '_new' + f'_{i}'

        # append data
        new_dfs.append(new_df)
    
    # concatenate data and drop duplicates and rows with missing target values
    extended_df = pd.concat([df] + new_dfs, axis=0, ignore_index=True)
    extended_df = extended_df.dropna(subset=['bg+1:00'], axis=0, ignore_index=True)
    extended_df = extended_df.drop_duplicates(subset=new_columns + ['bg+1:00'], ignore_index=True)

    return extended_df

def aggregate_time_series(df, time_len, skipna = True):
    '''
    This function is used to aggregate the 5-minute intervals of each time variable into a larger interval
    '''

    columns = list(df.columns)
    column_filter = [r'bg-.*',r'insulin-.*',r'carbs-.*',r'hr-.*',r'steps-.*',r'cals-.*',r'activity-.*']

    column_clusters = {}
    column_cluster_aux = []

    expanded_time_data = df.drop([col for col in columns if re.search('-[0-9]:[0-9].*',col)], axis=1).copy()

    # Get dictionary with column clusters
    for filter in column_filter:
        
        column_cluster_aux = [col for col in columns if re.search(filter, col)]
        column_clusters[filter.split('-')[0]] = column_cluster_aux

    new_column_clusters = column_clusters.copy()

    # Get columns grouped by ("time_len" * 5) minutes intervals 
    for key, val in column_clusters.items():

        time_clusters = {}
        time_list_aux = val

        for i in range(len(val) // time_len):

            time_clusters[key+'_'+str(i+1)] = time_list_aux[0:time_len]
            time_list_aux = time_list_aux[time_len:]

        new_column_clusters.update({key : time_clusters})

    operation_dict = {'bg' : 'mean','insulin':'sum','carbs':'sum','hr':'mean','steps':'sum','cals':'sum','activity':'sum'}

    # Define aggrgation functions
    def apply_operation(df, operation):

        match operation:

            case 'sum':
                return df.sum(axis=1, skipna=skipna)

            case 'mean':
                return df.mean(axis=1, skipna=skipna)

            case _:
                return df.sum(axis=1, skipna=skipna)

    reduced_data_dict = {}

    # Apply one operation to each type of column in order to aggregate time window vectors 
    for col, new_col_dict in new_column_clusters.items():
        for new_col, time_clust in new_col_dict.items():

            reduced_data_dict[new_col] = apply_operation(df[time_clust], operation_dict[col])

    expanded_time_data = pd.concat([expanded_time_data, pd.DataFrame(reduced_data_dict)], axis=1)

    return expanded_time_data

def get_null_index_from_list(df, list_of_index):
    '''
    This function is used to get the indixes of a dataframe that have null values from a given list of indexes. It also returns the position of the index in the given list.
    '''

    prov_mask = df.loc[list_of_index,:].isnull().any(axis=1)
    indexes = list(df.loc[list_of_index,:][prov_mask].index)
    position = []

    for time in indexes:
        position.append(list_of_index.index(time))

    return indexes, position

def get_null_time_indexes(p_num, mean_by_per_and_hour):
    '''
    This function is used to obtain indexes with null values.
    '''

    # get provisional dataset for a person and get the null index
    person_dataframe = mean_by_per_and_hour.loc[p_num,:].reset_index().sort_values(by='time').copy()
    person_mask = person_dataframe.isnull().any(axis=1)

    null_index = person_dataframe[person_mask].index
    null_time_index = list(person_dataframe.iloc[null_index,:]['time'])

    person_dataframe = person_dataframe.set_index('time')

    return person_dataframe, null_index, null_time_index

def fill_missing_values_by_interpolation(mean_by_per_and_hour, p_num, null_time_index, time_index_above, time_index_bellow):
    '''
    This function is used to interpolate missing values from two other rows
    '''
    # fill missing values with a weighted average of rows above and bellow
    for idx, index in enumerate(null_time_index):
    
        # get rows above and bellow
        readings_above = mean_by_per_and_hour.loc[(p_num, (time_index_above[idx])), :].copy()
        readings_bellow = mean_by_per_and_hour.loc[(p_num, (time_index_bellow[idx])), :].copy()

        # get times for each rows
        start_time = datetime.strptime(time_index_above[idx], '%H:%M:%S')
        current_time = datetime.strptime(index, '%H:%M:%S')
        end_time = datetime.strptime(time_index_bellow[idx], '%H:%M:%S')

        # if one time correspond to another day, adjust it
        if start_time > current_time:
            start_time -= timedelta(days=1)

        if end_time < current_time:
            end_time += timedelta(days=1)

        # get time differences (hours)
        time_diff = (end_time - start_time).total_seconds() / 3600
        start_diff = (current_time - start_time).total_seconds() / 3600
        end_diff = (end_time - current_time).total_seconds() / 3600

        # get weights for rows above and below
        time_above_weight = 1 - start_diff / time_diff
        time_bellow_weight = 1 - end_diff / time_diff

        # use the weights to interpolate missing values
        weighted_mean_row = time_above_weight * readings_above + time_bellow_weight * readings_bellow
        mean_by_per_and_hour.loc[(p_num, index), :] = mean_by_per_and_hour.loc[(p_num, index), :].fillna(weighted_mean_row)

    return mean_by_per_and_hour

def interpolate_and_fill_rows(p_num, mean_by_per_and_hour, n_iters):
    '''
    This function is used to fill null values by interpolating along axis 0
    '''

    # get null time indexes
    person_dataframe, null_index, null_time_index = get_null_time_indexes(p_num, mean_by_per_and_hour)

    # Get adapters to obtain rows bellow and above
    num_index_adapter_1 = np.array(list(null_index))
    num_index_adapter_2 = np.array(list(null_index))

    for j in range(n_iters):

        # Manage when hourly indexes correspond to first hour in the morning or last hour in the evening
        for i in range(len(null_index)):
            if num_index_adapter_1[i] == 0:
                num_index_adapter_1[i] = person_dataframe.shape[0]

            if num_index_adapter_2[i] >= (person_dataframe.shape[0]-1):
                num_index_adapter_2[i] = -1

        # get time index for rows above and bellow
        time_index_above = list(person_dataframe.iloc[num_index_adapter_1-1,:].index)
        time_index_bellow = list(person_dataframe.iloc[num_index_adapter_2+1,:].index)

        # interpolate and fill missing values
        mean_by_per_and_hour = fill_missing_values_by_interpolation(mean_by_per_and_hour, p_num, null_time_index, time_index_above, time_index_bellow)

        num_index_adapter_1 -= 1
        num_index_adapter_2 += 1
        
    return mean_by_per_and_hour

def interpolate_and_fill_columns(p_num, mean_by_per_and_hour, set_columns):
    '''
    This function is used to fill null values by interpolating along axis 1
    '''

    # get bg mean by hour for participant i
    person_dataframe = mean_by_per_and_hour.loc[p_num,:].reset_index().sort_values(by='time').copy()
    index = list(person_dataframe['time'])

    for idx, col in enumerate(set_columns[:-1]):

        # The first column cannot be interpolated. It is inferred from the following column
        if idx == 0:
            right_column = person_dataframe[set_columns[idx+1]].copy()
            right_column.index = index
            mean_by_per_and_hour.loc[p_num, col] = mean_by_per_and_hour.loc[p_num, col].fillna(right_column, axis=0).to_numpy()

            continue
        
        # get the following and previous columns
        left_column = person_dataframe[set_columns[idx-1]].copy()
        right_column = person_dataframe[set_columns[idx+1]].copy()

        # if the following or previous columns have missing values, they can not be used to interpolate
        if left_column.isnull().sum() > 0 or right_column.isnull().sum() > 0:
            continue
        
        else:
            mean_values = left_column + right_column / 2
            mean_values.index = index
            mean_by_per_and_hour.loc[p_num, col] = mean_by_per_and_hour.loc[p_num, col].fillna(mean_values, axis=0).to_numpy()

    return mean_by_per_and_hour

def get_participants_with_null_values(mean_by_per_and_hour):
    '''
    Get a list with participants who have null values in their data grouped by hour
    '''

    p_nums = mean_by_per_and_hour.index.get_level_values('p_num').unique()

    p_null_list = []
    for p in p_nums:
        sum_of_nulls = mean_by_per_and_hour.loc[p,:].isnull().sum().sum()

        if sum_of_nulls:
            p_null_list.append(p)

    return p_null_list

def impute_and_encode_hr(df, encode_dictionary, steps_to_impute, drop_other_cols):
    '''
    This function is used to encode and impute the current heart rate.
    The encoding is done by creating differents groups or ranges
    The imputation is done with the values from previous steps. Then, the rest of missing values are imputed with the mean.
    '''

    # get columns needed for imputation
    hr_columns = [col for col in df.columns if re.search('hr_.*',col)]
    current_index = len(hr_columns)
    columns_for_imputation = [f'hr_{i}' for i in range(current_index, current_index-steps_to_impute-1,-1)]
    column_to_impute, columns_for_imputation = (columns_for_imputation[0], columns_for_imputation[1:])

    # imputation with values from previous steps
    for col in list(columns_for_imputation):
        df[column_to_impute] = df[column_to_impute].fillna(df[col])
    
    # imputation with mean value
    mean_value = df[column_to_impute].mean()
    df[column_to_impute] = df[column_to_impute].fillna(mean_value)

    # encoding
    bins = list(encode_dictionary.values())
    labels = range(1,len(bins))
    df[f'current_hr'] = pd.cut(df[f'hr_{current_index}'], bins=bins, labels=labels, right=True)
    
    if drop_other_cols:
        df = df.drop(hr_columns, axis=1) 

    return df

def mean_ffill_bfill_imputation(df, columns):

    df_ffill = df.copy()
    df_ffill[columns] = df_ffill[columns].fillna(method='ffill', axis=1)

    df_bfill = df.copy()
    df_bfill[columns] = df_bfill[columns].fillna(method='bfill', axis=1)

    ffill_cols = df_ffill[columns].fillna(df_bfill[columns])
    bfill_cols = df_bfill[columns].fillna(df_ffill[columns])

    df_ffill[columns] = ffill_cols
    df_bfill[columns] = bfill_cols

    df_combined = df.copy()
    df_combined[columns] = (df_ffill[columns] + df_bfill[columns]) / 2

    df[columns] = df[columns].fillna(df_combined[columns])

    return df

def impute_cals(cals_mean_by_per_and_hour, impute_dict=None):
    '''
    This function is used to impute missing values of cals_mean_by_per_and_hour dataset.
    The values are filled with a blend of ffill and bfill due to high correlation betwen two consecutive columns.
    If there are still missing values, they are imputed with BMR (calculated in a 5-minute interval)
    '''

    # iterate for each participant
    for p_num, group in cals_mean_by_per_and_hour.groupby(level=0):

        # the imputation dictionary is used to obtain a sleep factor, in case it is observed that a participant is missing values at night.
        sleep_factor = impute_dict[p_num] if p_num in impute_dict.keys() else 1

        # calculate BMR in a 5-minute interval
        BMR = group.median(axis=0, skipna=True).median()
        
        # fill missing values with ffill and bfill
        group = mean_ffill_bfill_imputation(group, cals_mean_by_per_and_hour.columns)

        # if there are still missing values, impute them with BMR
        cals_mean_by_per_and_hour.loc[p_num] = group.fillna(BMR * sleep_factor)
    
    return cals_mean_by_per_and_hour
