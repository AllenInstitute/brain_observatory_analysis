import numpy as np
import pandas as pd
from pathlib import Path
from mindscope_utilities.visual_behavior_ophys import data_formatting
# TODO: change mindscope_utilities to brain_observatory_utilities after the migration


GH_RESPONSE_DF_DIR = Path(r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\Jinho\data\GH_data\response_df'.replace(
    '\\', '/'))
GH_TRACE_DF_DIR = Path(r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\Jinho\data\GH_data\trace_df'.replace(
    '\\', '/'))

VB_RESPONSE_DF_DIR = Path(r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\Jinho\data\VB_data\response_df'.replace(
    '\\', '/'))
VB_TRACE_DF_DIR = Path(r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\Jinho\data\VB_data\trace_df'.replace(
    '\\', '/'))

LAMF_RESPONSE_DF_DIR = Path(r'\\allen\programs\mindscope\workgroups\learning\analysis_data_cache\response_df'.replace(
    '\\', '/'))
LAMF_TRACE_DF_DIR = Path(r'\\allen\programs\mindscope\workgroups\learning\analysis_data_cache\trace_df'.replace(
    '\\', '/'))


def get_all_annotated_session_response_df(expt_group, session_name, data_type, cache_dir=None, save_cache=True,
                                          time_window=[-3, 3.1], output_sampling_rate=20):
    """Get all response_df for a session from all experiments in a expt_group

    Parameters
    ----------
    expt_group: Experiment group object
        Experiment group object for Learning and mFISH project
    session_name: str
        Session name
    data_type: str
        Data type, e.g., 'dff' and 'events'
    cache_dir: str, optional
        Cache directory, default None
    time_window: list, optional
        Window for stimulus response, default [-3, 3.1]
    output_sampling_rate: float, optional
        Output sampling rate in Hz, default 20

    Returns
    -------
    session_response_df: pandas DataFrame
        DataFrame with stimulus presentations for all experiments in a session
    """

    oeids = np.sort(expt_group.expt_table[expt_group.expt_table.session_name.str.lower(
    ) == session_name.lower()].index.values)
    session_response_df = pd.DataFrame()

    for oeid in oeids:
        exp = expt_group.experiments[oeid]

        exp_response_df = get_stim_annotated_response_df(
            exp, data_type=data_type, stim_type='all', cache_dir=cache_dir, time_window=time_window, output_sampling_rate=output_sampling_rate)
        if exp_response_df is not None:
            session_response_df = session_response_df.append(exp_response_df)
    return session_response_df


def get_stim_annotated_response_df(exp, data_type, stim_type, cache_dir=None, load_from_cache=True, save_to_cache=True,
                                   image_order=3, time_window=[-3, 3.1], output_sampling_rate=20):
    """ Get stimulus presentations with stim annotations
    Merge dataframe from data_formatting.get_annotated_stimulus_presentations and data_formatting.annotate_stimuli
    to data_formatting.get_stimulus_response_df
    Look for caching directory first, if not found, generate the dataframe.
    The resulting cache if created new can be saved if the caching directory is specified.

    Parameters
    ----------
    exp: BehaviorOphysExperiment
        Behavior ophys experiment object
    data_type: str
        Data type, e.g., 'dff' and 'events'
    stim_type: str
        Visual stimulus type, e.g. 'images', 'images>n-changes', 'changes', 'omissions', etc.
    image_order: int, optional
        Image order for images>n parameters, default 3
    time_window: list, optional
        Window for stimulus response, default [-3, 3.1]
    output_sampling_rate: float, optional
        Output sampling rate, default 20
        When not None, interpolate the response_df to the output_sampling_rate
        When None, shift to the nearest timepoint instead of interpolation

    Returns
    -------
    response_df: pandas.DataFrame
        A dataframe containing stimulus presentations with stim annotations
    """
    if exp.cell_specimen_table.index.unique()[0] is not None:
        if cache_dir is not None:
            cache_dir = Path(cache_dir)

            if output_sampling_rate is None:
                cache_fn = f'{exp.ophys_experiment_id}_{data_type}_{stim_type}_nointerp.pkl'
            else:
                cache_fn = f'{exp.ophys_experiment_id}_{data_type}_{stim_type}_{output_sampling_rate}Hz.pkl'
            cache_path = cache_dir / cache_fn
            if load_from_cache and cache_path.exists():
                print(f'Loading from cache: {cache_path}')
                response_df = pd.read_pickle(cache_path)
            else:
                response_df = data_formatting.get_stimulus_response_df(exp, data_type=data_type, stim_type=stim_type, image_order=image_order,
                                                                       time_window=time_window, output_sampling_rate=output_sampling_rate)
                stim_df = get_all_annotated_stimulus_presentations(exp)
                response_df = response_df.merge(
                    stim_df, how='left', on='stimulus_presentations_id', validate='m:1')
                response_df['oeid'] = exp.ophys_experiment_id
                if save_to_cache:
                    print(f'Saving to cache: {cache_path}')
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    response_df.to_pickle(cache_path)
    else:
        response_df = None

    return response_df


def get_all_annotated_stimulus_presentations(exp):
    """ Get stimulus presentations with all behavioral annotations
    Merge dataframe from data_formatting.get_annotated_stimulus_presentations and data_formatting.annotate_stimuli

    Parameters
    ----------
    exp: Experiment
        Experiment object

    Returns
    -------
    stim_df: pandas.DataFrame
        A dataframe containing stimulus presentations with all behavioral annotations
    """
    stim_df = data_formatting.get_annotated_stimulus_presentations(exp)
    stim_df_2 = data_formatting.annotate_stimuli(exp)
    overlapping_column_names = np.intersect1d(
        stim_df.keys().tolist(), stim_df_2.keys().tolist())
    stim_df_2 = stim_df_2.drop(columns=overlapping_column_names)
    stim_df = stim_df.join(stim_df_2)
    return stim_df
