import numpy as np
import pandas as pd
import re
from datetime import datetime

def anaerobic_time_function(x):
    return (1-1/(1+5*np.exp(-(x)*2)))

def aerobic_time_function(x):
    return (1-1/(1+8*np.exp(-(x)*1.5)))

def encode_activity_columns(df, aerobic_dict, anaerobic_dict):

    columns = list(df.columns)
    activity_columns = [col for col in columns if re.search(r'activity-.*', col)]

    data_with_scores = df.copy()

    # fill missing values with 0. In most cases, it is logical to assume that the participant did not play sports.
    data_with_scores[activity_columns] = data_with_scores[activity_columns].fillna(value=0)

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

    # obtain the sums of aerobic and anaerobic scores along time axis
    activity_scores_df = pd.concat([pd.DataFrame(aerobic_scores_dict),pd.DataFrame(anaerobic_scores_dict)], axis=1)

    aerobic_columns = aerobic_scores_dict.keys()
    anaerobic_columns = anaerobic_scores_dict.keys()

    sum_aerobic_scores_df = activity_scores_df[aerobic_columns].sum(axis=1).to_frame(name='Aerobic_score')
    sum_anaerobic_scores_df = activity_scores_df[anaerobic_columns].sum(axis=1).to_frame(name='Anaerobic_score')

    # drop original columns and add the new ones
    data_with_scores = data_with_scores.drop(activity_columns, axis=1)
    data_with_scores = pd.concat([data_with_scores, sum_aerobic_scores_df, sum_anaerobic_scores_df], axis=1)

    return data_with_scores

def reduce_time_window(df, time_window):

    reduced_data = df.copy()
    columns = list(reduced_data.columns)
    column_filter = fr'-[{time_window}-9]:[0-9].*'

    # Delete columns older than (time_window) h
    delete_columns = [col for col in columns if re.search(column_filter, col)]
    reduced_data = reduced_data.drop(delete_columns, axis=1)

    return reduced_data

def expand_time_clusters(df, time_len):

    expanded_time_data = df[['id','p_num','time','bg+1:00','Aerobic_score','Anaerobic_score']].copy()

    columns = list(df.columns)
    column_filter = [r'bg-.*',r'insulin-.*',r'carbs-.*',r'hr-.*',r'steps-.*',r'cals-.*']

    column_clusters = {}
    column_cluster_aux = []

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

    operation_dict = {'bg' : 'mean','insulin':'sum','carbs':'sum','hr':'mean','steps':'sum','cals':'sum'}

    # Define aggrgation functions
    def apply_operation(df, operation):

        match operation:

            case 'sum':
                return df.sum(axis=1)

            case 'mean':
                return df.mean(axis=1)

            case _:
                return df.sum(axis=1)

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

def get_time_indexes(p_num, mean_by_per_and_hour):
    '''
    This function is used to obtain indexes with null values and the rows needed to interpolate those values.
    '''
    # get provisional dataset for a person and get the null index
    person_dataframe = mean_by_per_and_hour.loc[p_num,:].reset_index().sort_values(by='time').copy()
    person_mask = person_dataframe.isnull().any(axis=1)

    null_index = person_dataframe[person_mask].index
    null_time_index = list(person_dataframe.iloc[null_index,:]['time'])

    person_dataframe = person_dataframe.set_index('time')

    # Get adapters to obtain rows bellow and above
    num_index_adapter_1 = np.array(list(null_index))
    num_index_adapter_2 = np.array(list(null_index))

    # In this loop it is checked whether or not rows above or bellow also have null values. If so, then get the next ones
    for i in range(10):

        # Manage when hourly indexes correspond to first hour in the morning or last hour in the evening
        for i in range(len(null_index)):
            if num_index_adapter_1[i] == 0:
                num_index_adapter_1[i] = person_dataframe.shape[0]

            if num_index_adapter_2[i] >= (person_dataframe.shape[0]-1):
                num_index_adapter_2[i] = -1

        # get time index for rows above and bellow
        time_index_above = list(person_dataframe.iloc[num_index_adapter_1-1,:].index)
        time_index_bellow = list(person_dataframe.iloc[num_index_adapter_2+1,:].index)

        # get time index in rows above and bellow that also have null values 
        time_above_null_index, _ = get_null_index_from_list(person_dataframe, time_index_above)
        time_bellow_null_index, _ = get_null_index_from_list(person_dataframe, time_index_bellow)

        # if rows above or bellow also have null values, then get the next one
        if time_above_null_index:
            num_index_adapter_1 += -1

        if time_bellow_null_index:
            num_index_adapter_2 += 1
        
        if not(time_bellow_null_index or time_above_null_index):
            break

    return person_dataframe, null_index, null_time_index, time_index_above, time_index_bellow

def fill_missing_values_by_interpolation(mean_by_per_and_hour, p_num, null_time_index, time_index_above, time_index_bellow):
    '''
    This function is used to interpolate missing values from two other rows
    '''
    # fill missing values with a weighted average of rows above and bellow
    for idx, index in enumerate(null_time_index):
    
        # get rows above and bellow
        readings_above = mean_by_per_and_hour.loc[(p_num, (time_index_above[idx])), :].copy()
        readings_bellow = mean_by_per_and_hour.loc[(p_num, (time_index_bellow[idx])), :].copy()

        if readings_above.isnull().sum() > 0 or readings_bellow.isnull().sum() > 0:
            continue

        # get times for each rows
        start_time = datetime.strptime(time_index_above[idx], '%H:%M:%S')
        current_time = datetime.strptime(index, '%H:%M:%S')
        end_time = datetime.strptime(time_index_bellow[idx], '%H:%M:%S')

        start_hour = start_time.hour + start_time.minute / 60
        current_hour = current_time.hour + current_time.minute / 60
        end_hour = end_time.hour + end_time.minute / 60

        # get weights for rows above and below
        time_diff = np.abs(end_hour-start_hour)
        time_above_weight = 1 - np.abs(current_hour-start_hour) / time_diff
        time_bellow_weight = 1 - np.abs(current_hour-end_hour) / time_diff

        # use the weights to interpolate missing values
        weighted_mean_row = time_above_weight * readings_above + time_bellow_weight * readings_bellow
        mean_by_per_and_hour.loc[(p_num, index), :] = mean_by_per_and_hour.loc[(p_num, index), :].fillna(weighted_mean_row)

    return mean_by_per_and_hour