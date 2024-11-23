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

def generate_time_lags(column_names, hours):
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

def expand_data(df, time_window):
    '''
    Get more data by reducing time window used for predictions
    '''
    
    columns = list(df.columns)
    not_time_columns = [col for col in columns if not(re.search(r'[0-9]:[0-9].*',col))]
    column_filter = [r'bg-.*',r'insulin-.*',r'carbs-.*',r'hr-.*',r'steps-.*',r'cals-.*',r'activity-.*']
    
    # get time window for original dataset
    start_time = datetime.strptime(str(6 - time_window), '%H')
    end_time = start_time + timedelta(hours=time_window-1, minutes=55)

    old_columns = []
    new_columns = []

    for filter in column_filter:
        
        # get new column names by generating a new time window
        name = filter.split('-')[0]
        column_group = [col for col in columns if re.search(filter,col)]
        new_columns += list(generate_time_lags([name], time_window).values())[0]

        old_columns_aux = []

        # get column names of original dataset that represents the time window
        for col in column_group:
            column_date = datetime.strptime(col.split('-')[1],'%H:%M')

            if (column_date >= start_time) and (column_date<=end_time):
                old_columns_aux.append(col)

        old_columns_aux.sort()
        old_columns += old_columns_aux
    
    # get the new bg+1:00
    new_bgp1h = (start_time - timedelta(hours=1)).time().isoformat(timespec='minutes')

    # create the new dataframe, modify current time variable and assign new ids
    new_df = df[not_time_columns].copy()
    new_df['time'] = (pd.to_datetime(new_df['time'],format='%H:%M:%S') - timedelta(hours=6-time_window)).dt.time.astype('str')
    new_df['bg+1:00'] = df['bg-'+str(new_bgp1h[1:])].copy()
    new_df['id'] = new_df['id'] + '_new'

    #get the rest variables from original df
    renamed_df = df[old_columns].copy()
    renamed_df.columns = new_columns

    new_df = pd.concat([new_df, renamed_df], axis=1)

    return new_df.dropna(subset=['bg+1:00'])

def expand_time_clusters(df, time_len, skipna = True):

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

def impute_cals(cals_mean_by_per_and_hour, impute_dict=None):
    '''
    This function is used to impute missing values of burnt calories
    '''

    # obtain participants with missing calorie values, in the dataset grouped by p_num and time
    nulls_count = cals_mean_by_per_and_hour.groupby('p_num').apply(lambda group: group.isnull().sum()).sum(axis=1)
    p_nums = nulls_count[nulls_count>0].index

    for p_num in p_nums:
        
        # get participant dataset grouped by time
        person_dataframe = cals_mean_by_per_and_hour.loc[p_num]
        null_index = person_dataframe.isnull().any(axis=1)[person_dataframe.isnull().any(axis=1)].index

        # if the sleep factor is applied to a participant, it is assumed that the missing values correspond to nighttime hours
        if impute_dict:
            sleep_factor = impute_dict[p_num] if p_num in impute_dict.keys() else 1
        else:
            sleep_factor = 1

        # get the basal metabolic rate (BMR) to impute missing values in the dataset grouped by p_num and time
        daily_cals = cals_mean_by_per_and_hour.loc[p_num].sum(axis=0)
        not_null_cals_count = cals_mean_by_per_and_hour.loc[p_num].apply(lambda group: group.notnull().sum())

        BMR = (daily_cals / not_null_cals_count).median()

        # impute missing values with BMR
        for index in null_index:
            cals_mean_by_per_and_hour.loc[(p_num,index),:] = cals_mean_by_per_and_hour.loc[(p_num,index),:].fillna(BMR*sleep_factor)

    return cals_mean_by_per_and_hour

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