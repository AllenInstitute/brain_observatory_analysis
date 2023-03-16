import os
import json
# import warnings
import numpy as np
import pandas as pd
from pathlib import Path

# import logging
# logger = logging.getLogger(__name__)


def add_layer_column(df):
    """
    Adds a column called 'layer' that is based on the 'imaging_depth' for each experiment.
    if imaging_depth is <250um, layer is 'upper, if >250um, layer is 'lower'
    :param df:
    :return:
    """
    df.loc[:, 'layer'] = None

    indices = df[(df.depth <= 250)].index.values
    df.loc[indices, 'layer'] = 'upper'

    indices = df[(df.depth > 250)].index.values
    df.loc[indices, 'layer'] = 'lower'

    return df

def add_area_layer_column(df):
    """
    creates columns called 'area_layer' and that contains the conjunction of 'targeted_area' and 'layer'
    input df must have 'layer' and 'targeted_structure' columns, the former created with the utilities.add_layer_column() function
    """
    df['area_layer'] = None
    for row in df.index.values:
        row_data = df.loc[row]
        layer = row_data.layer
        area = row_data.targeted_structure
        df.loc[row, 'area_layer'] = area + '_' + layer
    return df

def dateformat(exp_date):
    """
    reformat date of acquisition for accurate sorting by date
    """
    from datetime import datetime
    if exp_date is not str:
        exp_date = str(exp_date)[:-3] # remove 3 zeros from seconds, otherwise the str is too long
    date = int(datetime.strptime(exp_date, '%Y-%m-%d  %H:%M:%S.%f').strftime('%Y%m%d'))
    return date

def add_date_string(df):
    """
    Adds a new column called "date" that is a string version of the date_of_acquisition column,
    with the format year-month-date, such as 20210921
    """
    df['date'] = df['date_of_acquisition'].apply(dateformat)
    return df

def add_first_novel_column(df):
    """
    Adds a column called 'first_novel' that indicates (with a Boolean) whether a session is the first true novel image session or not
    Equivalent to experience_level == 'Novel 1'
    """
    df.loc[:, 'first_novel'] = False
    indices = df[(df.session_number == 4) & (df.prior_exposures_to_image_set == 0)].index.values
    df.loc[indices, 'first_novel'] = True
    return df

def get_n_relative_to_first_novel(group):
    """
    Function to apply to experiments_table data grouped on 'ophys_container_id'
    For each container, determines the numeric order of sessions relative to the first novel image session
    returns a pandas Series with column 'n_relative_to_first_novel' indicating this value for all session in the container
    If the container does not have a truly novel session, all values are set to NaN
    """
    group = group.sort_values(by='date')  # must sort for relative ordering to be accurate
    if 'Novel 1' in group.experience_level.values:
        novel_ind = np.where(group.experience_level == 'Novel 1')[0][0]
        n_relative_to_first_novel = np.arange(-novel_ind, len(group) - novel_ind, 1)
    else:
        n_relative_to_first_novel = np.empty(len(group))
        n_relative_to_first_novel[:] = np.nan
    return pd.Series({'n_relative_to_first_novel': n_relative_to_first_novel})


def add_n_relative_to_first_novel_column(df):
    """
    Add a column called 'n_relative_to_first_novel' that indicates the session number relative to the first novel session for each experiment in a container.
    If a container does not have a first novel session, the value of n_relative_to_novel for all experiments in the container is NaN.
    Input df must have column 'experience_level' and 'date'
    Input df is typically ophys_experiment_table
    """
    # add simplified string date column for accurate sorting
    df = add_date_string(df)  # should already be in the table, but adding again here just in case
    df = df.sort_values(by=['ophys_container_id', 'date'])  # must sort for ordering to be accurate
    numbers = df.groupby('ophys_container_id').apply(get_n_relative_to_first_novel)
    df['n_relative_to_first_novel'] = np.nan
    for container_id in df.ophys_container_id.unique():
        indices = df[df.ophys_container_id == container_id].index.values
        df.loc[indices, 'n_relative_to_first_novel'] = list(numbers.loc[container_id].n_relative_to_first_novel)
    return df

def add_last_familiar_column(df):
    """
    adds column to df called 'last_familiar' which indicates (with a Boolean) whether
    a session is the last familiar image session prior to the first novel session for each container
    If a container has no truly first novel session, all sessions are labeled as NaN
    input df must have 'experience_level' and 'n_relative_to_first_novel'
    """
    df['last_familiar'] = False
    indices = df[(df.n_relative_to_first_novel == -1) & (df.experience_level == 'Familiar')].index.values
    df.loc[indices, 'last_familiar'] = True
    return df

def add_second_novel_column(df):
    """
    Adds a column called 'second_novel' that indicates (with a Boolean) whether a session
    was the second passing novel image session after the first truly novel session, including passive sessions.
    If a container has no truly first novel session, all sessions are labeled as NaN
    input df must have 'experience_level' and 'n_relative_to_first_novel'
    """
    df['second_novel'] = False
    indices = df[(df.n_relative_to_first_novel == 1) & (df.experience_level == 'Novel >1')].index.values
    df.loc[indices, 'second_novel'] = True
    return df


def limit_to_last_familiar_second_novel(df):
    """
    Drops rows that are not the last familiar session or the second novel session
    """
    # drop novel sessions that arent the second one
    indices = df[(df.experience_level == 'Novel >1') & (df.second_novel == False)].index.values
    df = df.drop(labels=indices, axis=0)

    # drop Familiar sessions that arent the last one
    indices = df[(df.experience_level == 'Familiar') & (df.last_familiar == False)].index.values
    df = df.drop(labels=indices, axis=0)

    return df