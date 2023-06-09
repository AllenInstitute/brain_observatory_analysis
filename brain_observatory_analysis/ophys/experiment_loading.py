from allensdk.internal.api import PostgresQueryMixin
import os
import warnings
from functools import partial
from typing import Union
from pathlib import Path
import logging
from collections import ChainMap
import pandas as pd
import multiprocessing as mp

from . import expt_table_fix
import brain_observatory_analysis.utilities.file_utils as fu

from allensdk.brain_observatory.behavior.behavior_project_cache import \
    VisualBehaviorOphysProjectCache
import brain_observatory_analysis.utilities.experiment_table_utils as etu

from brain_observatory_analysis.ophys.behavior_ophys_experiment_dev import \
    BehaviorOphysExperimentDev
from allensdk.brain_observatory.behavior.behavior_ophys_experiment \
    import BehaviorOphysExperiment

########################################################################
# loading experiment table
########################################################################


def start_lamf_analysis(verbose=False, include_dev_projects=False):

    if include_dev_projects:
        projects = ["LearningmFISHTask1A",
                    "LearningmFISHDevelopment"]
    else:
        projects = ["LearningmFISHTask1A"]

    recent_expts = get_recent_expts(date_after="2021-08-01",
                                    projects=projects,
                                    pkl_workaround=False)

    recent_expts = etu.experiment_table_extended(recent_expts)

    # # filter by lamftask1a mice
    # names = ["Gold", "Silver", "Bronze", "Copper", "Nickel",
    #          "Titanium", "Silicon", "Aluminum", "Mercury", "Iron", "Cobalt"]
    # recent_expts = recent_expts[recent_expts["mouse_name"].isin(names)]

    if not include_dev_projects:
        recent_expts = etu.experiment_table_extended_project(recent_expts,
                                                             project="LearningmFISHTask1A")

    return recent_expts


def start_gh_analysis():
    projects = ["VisualBehaviorMultiscope4areasx2d"]
    expt_table = get_expts(projects=projects,
                           pkl_workaround=False)
    expt_table = etu.experiment_table_extended(expt_table)
    return expt_table


def start_vb_analysis():
    projects = ['VisualBehaviorMultiscope', 'VisualBehavior', 'VisualBehaviorTask1B']
    expt_table = get_expts(projects=projects,
                           pkl_workaround=False)
    expt_table = etu.experiment_table_extended(expt_table)
    return expt_table


def get_expts_by_mouseid(mouse_ids):

    cache = VisualBehaviorOphysProjectCache.from_lims()
    expts_table = cache.get_ophys_experiment_table(passed_only=False)
    expts_table = expts_table[expts_table['mouse_id'].isin(mouse_ids)]
    expts_table = etu.experiment_table_extended(expts_table)

    return expts_table


def apply_func_df(df, func):
    return df.apply(lambda x: func(x.name), axis=1)


def apply_func_df_row(df, func, **kwargs):
    return df.apply(func, axis=1, **kwargs)


def get_expts(projects: list = [],
              pkl_workaround: bool = False,
              verbose: bool = False,
              passed_only: bool = True) -> pd.DataFrame:
    experiments_table = get_expt_table(pkl_workaround=pkl_workaround, passed_only=passed_only)
    if projects:
        experiments_table = experiments_table.query(
            "project_code in @projects").copy()

    # pretty other cols
    experiments_table['cre'] = (experiments_table['cre_line']
                                .apply(lambda x: x.split('-')[0]))
    # print useful info
    if verbose:
        print("Table Info \n----------")

        print(f"Experiments: {experiments_table.shape[0]}")
        print(experiments_table.experiment_workflow_state.unique())
        print(experiments_table.project_code.unique())
    return experiments_table


def get_expt_table(pkl_workaround: bool = False,
                   passed_only: bool = False):
    if pkl_workaround:
        print("Implementing pkl workaround hack will PIKA fixes LIMS/MTRAIN")
        experiments_table = expt_table_fix.get_ophys_experiment_table()
    else:
        print("--------------------------------------------------------\n"
              "You have requested a direct experiment table LIMS call."
              "\nYou should be on older version/branch of AllenSDK (like MJD's 'lamf_hacks')."
              "\nIf you see a progress bar that is taking a long time, you are on the wrong version."
              "\n-------------------------------------------------------")
        cache = VisualBehaviorOphysProjectCache.from_lims()
        experiments_table = cache.get_ophys_experiment_table(passed_only=passed_only)

    experiments_table = experiments_table.sort_values(by=["date_of_acquisition"])
    return experiments_table


def get_recent_expts(date_after: str = '2021-01-01',
                     projects: list = [],
                     pkl_workaround: bool = False,
                     passed_only: bool = False,
                     verbose: bool = False) -> pd.DataFrame:
    """
    Returns a list of recent ophys experiments

    Parameters
    ----------
    date_after : str
        date after which to return experiments
    projects : list
        list of project codes to filter by
    pkl_workaround : bool
        use pkl workaround to get experiments table
    passed_only : bool
        only return experiments that have passed QC
    verbose : bool
        print useful info

    Returns
    -------
    pd.DataFrame
        list of experiment (Format: Experiment_Table)
    """
    experiments_table = get_expt_table(pkl_workaround=pkl_workaround, passed_only=passed_only)
    recent_expts = experiments_table.copy()
    recent_expts = recent_expts.loc[recent_expts.date_of_acquisition > date_after]

    if projects:
        recent_expts = recent_expts[recent_expts.project_code.isin(projects)]

    recent_expts.sort_values("date_of_acquisition", inplace=True)
    recent_expts = etu.experiment_table_extended(recent_expts)

    # print useful info
    if verbose:
        print("Table Info \n----------")
        print(f"N Experiments: {recent_expts.shape[0]}")
        print(f"QC states: {recent_expts.experiment_workflow_state.unique()}")
        print(f"Projects: {recent_expts.project_code.unique()}")
        print(f"Cre lines: {recent_expts.cre_line.unique()}")
        print(f"Mouse IDs: {recent_expts.mouse_id.unique()}")

    return recent_expts


def display_expt_table(df, extra_cols=[]):
    """Note must have new cols "cre", "date_string" "mouse_name" in df"""

    assert "cre" in df.columns
    assert "date_string" in df.columns
    assert "mouse_name" in df.columns

    return df[["mouse_id", "mouse_name", "project_code", "date_string", "cre",
               "imaging_depth", "targeted_structure", "session_type",
               "equipment_name"] + extra_cols]


def check_mc_params(expt_table):
    """Look at mc paramters (sigma for time and space) and see if they have been updated.

    ~mid 2022, mc parameters were updated to fix residual motion artifacts in rigid translation.
    This fixed the offending params, which were changed for SSF project.

    """
    expt_table["new_mc_params"] = apply_func_df_row(expt_table, check_expt_mc_params)
    return expt_table


########################################################################
# loading experiments
########################################################################


def load_ophys_expts(expts_to_analyze: Union[list, pd.DataFrame],
                     multi: bool = True,
                     return_failed=False,
                     dev=False,
                     skip_eye_tracking=False,
                     dev_dff_path: Path = None,
                     dev_events_path: Path = None,
                     ) -> dict:
    """Load expts from LIMS and return datasets, single or multi core

    Parameters
    ----------
    expts_to_analyze : Union[list, pd.DataFrame]
        List of experiment ids or dataframe with experiment ids
    multi : bool
        Whether to use multiprocessing
    return_failed : bool
        Whether to return failed experiments
    dev : bool, optional
        Whether to use pipeline_dev.BehaviorOphysExperimentDev object,
        by default False
    dev_dff_path : Path
        Path to dff file if dev
    dev_events_path : Path
        Path to events file if dev
    skip_eye_tracking : bool, optional
        Whether to skip eye tracking, by default False

    Returns
    -------
    dict
        Dictionary of experiment ids and datasets

    """
    assert isinstance(expts_to_analyze, list) or \
        isinstance(expts_to_analyze, pd.DataFrame)

    if isinstance(expts_to_analyze, pd.DataFrame):
        expts_to_analyze = expts_to_analyze.index.to_list()

    # TODO: implement check that all ids are valid

    if multi:
        datasets_dict = \
            get_ophys_expt_multi(expts_to_analyze,
                                 return_failed=return_failed,
                                 dev=dev,
                                 dev_dff_path=dev_dff_path,
                                 dev_events_path=dev_events_path,
                                 skip_eye_tracking=skip_eye_tracking)
    else:
        datasets_dict = {}
        for expt_id in expts_to_analyze:
            new_item = get_ophys_expt(expt_id,
                                      as_dict=True,
                                      dev=dev,
                                      dev_dff_path=dev_dff_path,
                                      dev_events_path=dev_events_path,
                                      skip_eye_tracking=skip_eye_tracking)
            datasets_dict.update(new_item)
    if return_failed:
        failed = None  # TODO: check if needed
        return datasets_dict, failed
    else:
        return datasets_dict


def get_ophys_expt(ophys_expt_id: int, as_dict: bool = False, log=False,
                   dev=False, dev_dff_path=None, dev_events_path=None,
                   skip_eye_tracking=False,
                   **kwargs) -> Union[BehaviorOphysExperiment, dict]:
    """get ophys experiment from lims

    Parameters
    ----------
    ophys_expt_id : int
        ophys experiment id
    as_dict : bool, optional
        return as dict, good for multiprocessing functions
    log: bool, optional
        turn on logging
    dev: bool, optional
        use pipeline_dev.BehaviorOphysExperimentDev object
    dev_dff_path: Path, optional
        path to dev dff file, default None
    dev_events_path: Path, optional
        path to dev events file, default None
    kwargs : dict
        kwargs to pass to BehaviorOphysExperiment or dev object

    Returns
    -------
    dict
        single ophys experiment in dict 
        (key: ophys_expt_id, value: BehaviorOphysExperiment)

    """
    if log:
        # logger = logging.getLogger("get_ophys_expt")
        logging.exception(f"ophys_expt_id: {ophys_expt_id}")
    experiment = None
    try:
        print(f"Loading expt: {ophys_expt_id}\n")
        if not dev:
            experiment = BehaviorOphysExperiment.from_lims(ophys_expt_id,
                                                           skip_eye_tracking=skip_eye_tracking,
                                                           **kwargs)
        else:
            experiment = BehaviorOphysExperimentDev(ophys_expt_id,
                                                    dev_dff_path=dev_dff_path,
                                                    dev_events_path=dev_events_path,
                                                    skip_eye_tracking=skip_eye_tracking,
                                                    **kwargs)
    except Exception as e:
        #logging.exception(f"Failed to load ophys_expt_id: {ophys_expt_id}")
        # logger.exception(f"Failed to load expt: {ophys_expt_id}")
        print(f"Failed to load expt: {ophys_expt_id}")
        print("exc1")
        raise e 

    if as_dict:
        return {ophys_expt_id: None}
    else:
        return None


def get_ophys_expt_multi(expt_ids: list,
                         return_failed: bool = False,
                         dev=False,
                         dev_dff_path=None,
                         dev_events_path=None,
                         skip_eye_tracking=False) -> dict:
    """Use multiprocessing to load list of ophys experiments

    Parameters
    ----------
    expt_ids : list
        list of ophys experiment ids
    return_failed : bool, optional
        whether to return failed experiments
    dev: bool, optional
        use pipeline_dev.BehaviorOphysExperimentDev object
    dev_dff_path: Path, optional
        path to dev dff file, default None
    dev_events_path: Path, optional
        path to dev events file, default None
    skip_eye_tracking: bool, optional
        whether to skip eye tracking

    Returns
    -------
    dict
        dictionary of ophys experiments

    """
    # switched to thread pool 9/22 when PIKA put pkl reading mp
    # with mp.pool.ThreadPool(mp.cpu_count()-2) as P:

    with mp.Pool(mp.cpu_count() - 2) as P:
        func = partial(get_ophys_expt, as_dict=True, dev=dev,
                       dev_dff_path=dev_dff_path,
                       dev_events_path=dev_events_path,
                       skip_eye_tracking=skip_eye_tracking)
        result = P.map(func, expt_ids)

    # remove failures from result, store in another list
    # failed_expts = []
    # for expt in result:
    # if not all(expt.values()):
    # result.remove(expt)
    # failed_expts.append(expt.keys())
    # failed_expts = list(itertools.chain(*failed_expts))

    ophys_expt_dicts = dict(ChainMap(*result))

    if return_failed:
        return ophys_expt_dicts,  # failed_expts # TODO: see if needed
    else:
        return ophys_expt_dicts

# TODO move to expt_table_utils


def add_cum_sessions_column(df):
    """Add cumulative sessions columns to dataframe

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with mouse_id and date_of_acquisition columns

    Returns
    -------
    df : pandas.DataFrame
        Dataframe with cumulative sessions columns
    """
    df = df.sort_values(["mouse_id", "date_of_acquisition"])
    dfs_tmp = []
    for n, g in df.groupby(["mouse_id"]):
        df_tmp = g.reset_index().reset_index().rename(
            columns={"index": "session_number_cumulative"})
        dfs_tmp.append(df_tmp)

    return pd.concat(dfs_tmp)

# `
# Check data processing params
###############################################################################


def check_expt_mc_params(expt_id, param_string="\"smooth_sigma_time\": 0.0",
                         file_type='mc',
                         verbose=True):

    if file_type == "mc":
        file_parts_list = ["SUITE2P_MOTION_CORRECTION_QUEUE", ".log"]

    ed = get_experiment_directory_from_ophys_experiment_id(expt_id)

    files = fu.find_files_with_string(ed, file_parts_list)
    if len(files) == 0:
        new_params = False
        if verbose:
            print(f"No {file_type} for expt {expt_id}")
    # TODO does not work properly
    for f in files:

        # open file and check from matching string
        with open(f, 'r', encoding="utf8", errors='ignore') as file:
            data = file.read().replace('\n', '')
            print(data)
            if param_string in data:
                new_params = True
                print('new params')
            else:
                new_params = False
                if verbose:
                    print(f"Expt {expt_id} has old {file_type} params")
        """
        with open(f, 'rb') as file:
            content = file.read()

            # check if string present or not
            if param_string.encode() in content:
                new_params = True
                break 
            else:
                new_params = False
        """
    return new_params


# `
# TODO MOVE THIS WHOLE BLOCK
###############################################################################

# Accessing Lims Database
try:
    lims_dbname = os.environ["LIMS_DBNAME"]
    lims_user = os.environ["LIMS_USER"]
    lims_host = os.environ["LIMS_HOST"]
    lims_password = os.environ["LIMS_PASSWORD"]
    lims_port = os.environ["LIMS_PORT"]

    lims_engine = PostgresQueryMixin(
        dbname=lims_dbname,
        user=lims_user,
        host=lims_host,
        password=lims_password,
        port=lims_port
    )

    # building querys
    mixin = lims_engine

except Exception as e:
    warn_string = 'failed to set up LIMS/mtrain credentials\n{}\n\n \
        internal AIBS users should set up environment variables \
        appropriately\nfunctions requiring database access will fail'.format(e)
    warnings.warn(warn_string)


def get_experiment_directory_from_ophys_experiment_id(expt_id: int):
    """Get experiment directory from LIMS

    Parameters:
    -----------
    experiment_id : int
        ophys experiment id

    Returns:
    --------
    experiment_directory : str
        path to experiment directory
    """
    query = f'''
            SELECT
            oe.id AS ophys_experiment_id,
            oe.storage_directory AS experiment_storage_directory
            FROM
            ophys_experiments oe
            WHERE oe.id = {expt_id}
            '''
    return mixin.select(query).experiment_storage_directory.values[0]
