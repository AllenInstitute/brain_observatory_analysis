# import os
# import json
# import warnings
import numpy as np
import pandas as pd
# from pathlib import Path

from allensdk.brain_observatory.behavior.behavior_project_cache import \
    VisualBehaviorOphysProjectCache

# TO DO: change experiment table loading to not use cache
cache = VisualBehaviorOphysProjectCache.from_lims()

# import logging
# logger = logging.getLogger(__name__)

# to do: make below function to work with bisect_layer function
# def add_area_layer_column(df):
#     """
#     creates columns called 'area_layer' and that contains the conjunction of 'targeted_area' and 'layer'
#     input df must have 'layer' and 'targeted_structure' columns, the former created with the utilities.add_layer_column() function
#     """
#     df['area_layer'] = None
#     for row in df.index.values:
#         row_data = df.loc[row]
#         layer = row_data.layer
#         area = row_data.targeted_structure
#         df.loc[row, 'area_layer'] = area + '_' + layer
#     return df


def dateformat(exp_date):
    """
    reformat date of acquisition for accurate sorting by date
    """
    from datetime import datetime
    if exp_date is not str:
        exp_date = str(exp_date)[:-3]  # remove 3 zeros from seconds, otherwise the str is too long
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


def get_n_relative_to_first_novel(df):
    """
    Function to apply to experiments_table data grouped on 'ophys_container_id'
    For each container, determines the numeric order of sessions relative to the first novel image session
    returns a pandas Series with column 'n_relative_to_first_novel' indicating this value for all session in the container
    If the container does not have a truly novel session, all values are set to NaN
    """
    df = df.sort_values(by='date')  # must sort for relative ordering to be accurate
    if 'Novel 1' in df.experience_level.values:
        novel_ind = np.where(df.experience_level == 'Novel 1')[0][0]
        n_relative_to_first_novel = np.arange(-novel_ind, len(df) - novel_ind, 1)
    else:
        n_relative_to_first_novel = np.empty(len(df))
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
    if 'n_relative_to_first_novel' not in df.columns:
        df = add_n_relative_to_first_novel_column(df)
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
    if 'n_relative_to_first_novel' not in df.columns:
        df = add_n_relative_to_first_novel_column(df)
    df['second_novel'] = False

    indices = df[(df.n_relative_to_first_novel == 1) & (df.experience_level == 'Novel >1')].index.values
    df.loc[indices, 'second_novel'] = True
    return df


def limit_to_last_familiar_second_novel(df):
    """
    Drops rows that are not the last familiar session or the second novel session
    """
    print('starting with', len(df), 'experiments...')

    if 'second_novel' not in df.columns:
        df = add_second_novel_column(df)
    # drop novel sessions that arent the second one
    indices = df[(df.experience_level == 'Novel >1') & (df.second_novel == False)].index.values  # noqa
    df = df.drop(labels=indices, axis=0)

    print('dropped', len(indices), 'experiments that were not the second novel session')

    if 'last_familiar' not in df.columns:
        df = add_last_familiar_column(df)
    # drop Familiar sessions that arent the last one
    indices = df[(df.experience_level == 'Familiar') & (df.last_familiar == False)].index.values  # noqa
    df = df.drop(labels=indices, axis=0)

    print('dropped', len(indices), 'experiments that were not the last familiar session')
    print('ending with', len(df), 'experiments')

    return df

# Cell specimen table


def add_experience_level_column(cells_table, experiment_table=None):
    """
    adds column to cells_table called 'experience_level' which indicates the experience level of the session for each cell
    input cells_table must have column 'ophys_experiment_id', such as in ophys_cells_table
    """

    if experiment_table is None:
        experiment_table = cache.get_ophys_experiment_table()

    cells_table = cells_table.merge(experiment_table.reset_index()[['ophys_experiment_id', 'experience_level']], on='ophys_experiment_id', how='left')
    print('adding column "exprience_level" ')
    return cells_table


def get_cell_specimen_ids_with_all_experience_levels(cells_table, experiment_table=None):
    """
    identifies cell_specimen_ids with all 3 experience levels in ['Familiar', 'Novel 1', 'Novel >1'] in the input dataframe
    input dataframe must have column 'cell_specimen_id', such as in ophys_cells_table
    """

    if 'oexperience_level' not in cells_table.columns:
        cells_table = add_experience_level_column(cells_table, experiment_table)

    experience_level_counts = cells_table.groupby(['cell_specimen_id', 'experience_level']).count().reset_index().groupby(['cell_specimen_id']).count()[['experience_level']]
    cell_specimen_ids_with_all_experience_levels = experience_level_counts[experience_level_counts.experience_level == 3].index.unique()
    return cell_specimen_ids_with_all_experience_levels


def limit_to_cell_specimen_ids_matched_in_all_experience_levels(cells_table, experiment_table=None):
    """
    returns dataframe limited to cell_specimen_ids that are present in all 3 experience levels in ['Familiar', 'Novel 1', 'Novel >1']
    input dataframe is typically ophys_cells_table but can be any df with columns 'cell_specimen_id' and 'experience_level'
    """
    cell_specimen_ids_with_all_experience_levels = get_cell_specimen_ids_with_all_experience_levels(cells_table, experiment_table)
    matched_cells_table = cells_table[cells_table.cell_specimen_id.isin(cell_specimen_ids_with_all_experience_levels)].copy()
    return matched_cells_table
