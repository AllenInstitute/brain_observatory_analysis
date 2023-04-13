
import click
import pathlib as Path
import pandas as pd
import mindscope_qc.projects.lamf_associative_pilot as constants
import multiprocessing as mp
import mindscope_qc.data_formatting.change_detection.change_detection_dataset \
    as ChangeDetectionDataset
from tdqm import tqdm

# create arg input with click
@click.command()
@click.option('--mouse_ids', default=None, help='mouse ids to run')

LEARNING_PATH = Path('/allen/programs/mindscope/workgroups/learning/')
PROCESSED_TRIALS_DF = 'extended_trials_df.pkl'

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def cohort_extended_trials_df(mouse_ids):
    expt_table = load_expt_table_by_mice(mouse_ids)
    expts_to_analyze = expt_table.index.to_list()

    df_cohort = bu.mp_extended_trials(expts_to_analyze)
    df_cohort = bu.add_consecutive_session_codes(df_cohort)


    return


def load_cohorts(cohort_dict, save_dir: str = None):
    dfs = []
    for cohort in cohort_dict:
        df = cohort_extended_trials_df(cohort_dict[cohort])
        dfs.append(df)

    dfs = pd.concat(dfs)

    if save_dir:
        dfs.to_pickle(save_dir / PROCESSED_TRIALS_DF)

    return 

def multi_session_extended_trials(sessions):
    """
    Imports: pandas"""

    dfs = []
    all_dfs = pd.DataFrame() # TODO: what happens id all_dfs is blank?
    for session in tqdm(sessions):
        try:
            dataset = ChangeDetectionDataset(session, catch_hack=True)
            core_data = dataset.core_data
            extended_trials = dataset.extended_trials_df
            dfs.append(extended_trials)

            # HACK: fix mouse id for 2022 mice
            extended_trials["mouse_id"] = \
                core_data["metadata"]["params"]["mouse_id"]
            
            all_dfs = (pd.concat(dfs)
                         .reset_index()
                         .rename(columns={'index':'trial'})
                         .drop(columns={'level_0'}))
        except:
            print(f"Error processing session: {session}")
            
    return all_dfs

def get_extended_trials(session):

    try:
        logger.info(f"Loading_session: {session}")
        extended_trials, core_data = extended_trials_dataframe_from_session(session, catch_trial_hack=True, return_core_data=True)
        extended_trials["mouse_id"] = core_data["metadata"]["params"]["mouse_id"] # HACK: fix mouse id for 2022 mice
    except Exception:
        logger.exception(f"Can't load session: {session}")
        extended_trials = pd.DataFrame() # send dummy back
        pass

    return extended_trials


def mp_extended_trials_df(sessions):
    """
    Imports: pandas"""

    assert type(sessions) == list, "sessions must be a list of session ids"
    assert len(sessions) > 1, "sessions must be a list of session ids"

    with mp.Pool(mp.cpu_count()-2) as P:

        result = P.map(get_extended_trials, sessions)

    all_dfs = (pd.concat(result)
                .reset_index()
                .rename(columns={'index':'trial'}))

    return all_dfs

def load_expt_table_by_mice(mouse_ids):
    # TODO: DUPLICATE IN behavior utils right not
    """Get experiment table from LIMS, filtered by mouse_ids
    TODO: does this duplicate other functions

    Parameters
    ----------
    mouse_ids : list
        list of mouse ids

    Returns
    -------
    pd.DataFrame
        experiment table
    """
    from allensdk.brain_observatory.behavior.behavior_project_cache \
        import VisualBehaviorOphysProjectCache

    cache = VisualBehaviorOphysProjectCache.from_lims()
    behavior_sessions = cache.get_behavior_session_table(passed_only=False)
    ass_mice_table = behavior_sessions.query('mouse_id in @mouse_ids')
    print(f"# of behavior sessions in LIMS: {ass_mice_table.shape[0]}")
    #return ass_mice_table[["mouse_id","project_code","date_of_acquisition","session_type"]]
    return ass_mice_table


# make main function to run all of the above functions
def main():

    mouse_ids_dict = constants.mouse_ids()


    behavior_folder = 'behavior_pilots'
    cohort_folder = '20230103_AssociativePilotCohort5'
    save_dir = LEARNING_PATH / behavior_folder / cohort_folder
    # get extended trials dataframes
    cohort_extended_trials_df(mouse_ids_dict["cohort5"], save_dir)

    # get summary dataframes
    #cohort_summary_df(mouse_ids)

    # make summary figure
    #make_summary_figure(cohort_summary_df(mouse_ids))

    return


if __name__ == '__main__':
    main()
