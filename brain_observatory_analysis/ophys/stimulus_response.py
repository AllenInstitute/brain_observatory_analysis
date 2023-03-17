import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union
import json
import datetime

from mindscope_qc.data_access.behavior_ophys_experiment_dev import BehaviorOphysExperimentDev
from allensdk.brain_observatory.behavior.behavior_ophys_experiment import BehaviorOphysExperiment

# TODO: change to brain_observatory utilities
from mindscope_utilities.visual_behavior_ophys.data_formatting import get_stimulus_response_df

from .experiment_group import ExperimentGroup

from brain_observatory_analysis.behavior.change_detection.data_wrangling.extended_stimulus_presentations import get_extended_stimulus_presentations

import hashlib

import pickle


def get_mean_stimulus_response_expt_group(expt_group: ExperimentGroup,
                                          event_type: str = "changes",
                                          data_type: str = "dff",
                                          load_from_file: bool = True) -> pd.DataFrame:
    """
    Get mean stimulus response for each cell in the experiment group

    Parameters
    ----------
    expt_grp: ExperimentGroup
        An ExperimentGroup object
    event_type: str
        The event type ["changes", "omissions", "images", "all"]
        see get_stimulus_response_df() for more details

    Returns
    -------
    pd.DataFrame
        A dataframe with the mean response traces & metrics
        trace metrics = mean_trace, sem_trace, trace_timstamps
        response metrics = mean_response, sem_response, mean_baseline,
                            sem_baseline, response_latency, fano_factor,
                            peak_response, time_to_peak, reliablity

                            p_value, respone_window_duration,
                            sd_over_baseline,
                            fraction_significant_p_value_gray_screen
                            correlation_values
        """

    mdfs = []
    for expt_id, expt in expt_group.experiments.items():
        try:    

            # TODO: MOVE TO DEV
            expt.extended_stimulus_presentations = \
                get_extended_stimulus_presentations(expt.stimulus_presentations.copy(),
                                                    expt.licks,
                                                    expt.rewards,
                                                    expt.running_speed,
                                                    expt.eye_tracking)

            sdf = _get_stimulus_response_df(expt,
                                            event_type=event_type,
                                            data_type=data_type,
                                            load_from_file=load_from_file,
                                            save_to_file=True)
            
            sdf = sdf.merge(expt.extended_stimulus_presentations, on='stimulus_presentations_id')
            mdf = get_standard_mean_df(sdf)
            mdf["ophys_experiment_id"] = expt_id
            mdf["event_type"] = event_type
            mdf["data_type"] = data_type
            mdfs.append(mdf)
        except Exception as e:
            print(f"Failed to get stim response for: {expt_id}, {e}")
            continue
    
    mdfs = pd.concat(mdfs).reset_index(drop=True)

    # cells_filtered has expt_table info that is useful
    expt_table = expt_group.expt_table
    oct = expt_group.grp_ophys_cells_table.reset_index()

    # calculate more metrics, will likely move to own functions
    mdfs["mean_baseline_diff"] = mdfs["mean_response"] - \
        mdfs["mean_baseline"]
    mdfs["mean_baseline_diff_trace"] = mdfs["mean_trace"] - \
        mdfs["mean_baseline"]

    merged_mdfs = (mdfs.merge(expt_table, on=["ophys_experiment_id"])
                       .merge(oct, on=["cell_roi_id"]))

    return merged_mdfs


def _get_stimulus_response_df(experiment: Union[BehaviorOphysExperiment, BehaviorOphysExperimentDev],
                              event_type: str = 'changes',
                              data_type: str = "dff",
                              output_sampling_rate: float = 10.7,
                              save_to_file: bool = False,
                              load_from_file: bool = False,
                              # # "/allen/programs/mindscope/workgroups/learning/qc_plots/dev/mattd/3_lamf_mice/stim_response_cache"
                              cache_dir: Union[str, Path] = "/allen/programs/mindscope/workgroups/learning/analysis_data_cache/stim_response_df"):
    """Helper function for get_stimulus_response_df

    Parameters
    ----------
    experiment: BehaviorOphysExperiment or BehaviorOphysExperimentDev
        An experiment object
    event_type: str
        The event type ["changes", "omissions", "images", "all"]
        see get_stimulus_response_df() for more details
    output_sampling_rate: float
        The sampling rate of the output trace
    save_to_file: bool
        If True, save the stimulus response dataframe to a file
    load_from_file: bool
        If True, load the stimulus response dataframe from a file

    Returns
    -------
    pd.DataFrame
        A dataframe with the stimulus response traces & metrics

    # TODO: unhardcode cache_dir
    # TODO: output_sampling_rate smart calculation

    """
    if not isinstance(experiment, (BehaviorOphysExperiment, BehaviorOphysExperimentDev)):
        raise TypeError("experiment must be a BehaviorOphysExperiment or BehaviorOphysExperimentDev")

    if event_type not in ["changes", "omissions", "images", "all"]:
        raise ValueError("event_type must be one of ['changes', 'omissions', 'images', 'all']")

    if save_to_file and load_from_file:
        raise ValueError("save_to_file and load_from_file cannot both be True")

    expt_id = experiment.metadata["ophys_experiment_id"]

    cache_dir = Path(cache_dir)

    # dev object can report correct frame rate, but different frame
    # rate are possible across sessions, this would produce different
    # trace sizes in stim response df, do for now just use constant 10.7.
    # consider intelligent ways of handling this
    # frame_rate = experiment.metadata["ophys_frame_rate"]

    try:

        # may implement later, using unique filenames
        # now = datetime.datetime.now()
        # dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        # base_fn = f"{expt_id}_{data_type}_{event_type}.pkl"
        # unique_fn = f"{dt_string}_{base_fn}"
        base_fn = f"{expt_id}_{data_type}_{event_type}"
        fn = f"{base_fn}.pkl"

        if (cache_dir / fn).exists() and load_from_file:
            sdf = pd.read_pickle(cache_dir / fn)
            print(f"Loading stim response df for {expt_id} from file")
        else:
            # generally don't need to change
            time_window = [-3, 3]
            response_window_duration = 0.5
            interpolate = True

            sdf = get_stimulus_response_df(experiment,
                                           event_type=event_type,
                                           data_type=data_type,
                                           time_window=time_window,
                                           response_window_duration=response_window_duration,
                                           interpolate=interpolate,
                                           output_sampling_rate=output_sampling_rate)
            if save_to_file:
                # gather all inputs into params dict
                func_params = {"event_type": event_type,
                               "data_type": data_type,
                               "output_sampling_rate": output_sampling_rate,
                               "time_window": time_window,
                               "response_window_duration": response_window_duration,
                               "interpolate": interpolate
                               }
                
                # save dff has to see if anything changes later
                data_verify = {# "cell_specimen_id": experiment.cell_specimen_table.index.values,
                               # "cell_roi_id": experiment.cell_specimen_table["cell_roi_id"].values,
                               "dff_hash": hashlib.sha256(pickle.dumps(experiment.dff_traces.values)).hexdigest()}

                params = {"experiment_id": expt_id,
                          "data_verify": data_verify,
                          "function": "_get_stimulus_response_df",
                          "function_params": func_params}
                
                if (cache_dir / fn).exists():
                    print(f"Overwriting stim response df for {expt_id} in file")
                sdf.to_pickle(cache_dir / fn)
                print(f"Saving stim response df for {expt_id} to file")

                params["stimulus_response_df_path"] = str(cache_dir / fn)

                # save params to file
                with open(cache_dir / f"{base_fn}_params.json", "w") as f:
                    json.dump(params, f)
        return sdf
    except Exception as e:
        print(f"Failed to get stim response for: {expt_id}, {e}")
        return None


####################################################################################################
# utilites
####################################################################################################

# TODO: clean + document
def get_standard_mean_df(sr_df):
    time_window = [-3, 3.1]
    output_sampling_rate = 10.7
    get_pref_stim = False  # relevant to image_name conditions
    exclude_omitted_from_pref_stim = True  # relevant to image_name conditions  

    if "response_window_duration" in sr_df.keys():
        response_window_duration = sr_df.response_window_duration.values[0]

    output_sampling_rate = sr_df.ophys_frame_rate.unique()[0]
    conditions = ["cell_roi_id"]

    if get_pref_stim:
        # options for groupby "change_image_name", "image_name", "prior_image_name"
        # conditions.append('change_image_name')
        conditions.append('image_name')
    mdf = get_mean_df(sr_df,
                      conditions=conditions,
                      frame_rate=output_sampling_rate,
                      window_around_timepoint_seconds=time_window,
                      response_window_duration_seconds=response_window_duration,
                      get_pref_stim=get_pref_stim,
                      exclude_omitted_from_pref_stim=exclude_omitted_from_pref_stim)

    # annotate with stimulus info

    return mdf


# TODO: clean + document
def get_mean_df(stim_response_df: pd.DataFrame,
                conditions=['cell_roi_id', 'image_name'],
                frame_rate=11.0,
                window_around_timepoint_seconds: list = [-3, 3],
                response_window_duration_seconds: float = 0.5,
                get_pref_stim=True,
                exclude_omitted_from_pref_stim=True):
    """
    # MJD NOTES

    1) groupby "conditions": "cell" makes sense, TODO: "change_image_name" tho?
    2) apply get_mean_sem_trace()
    3) "response_window_duration_seconds" in df already
    4) "frame_rate" in df already
    5) get_pre_stim:
    """
    window = window_around_timepoint_seconds
    response_window_duration = response_window_duration_seconds

    rdf = stim_response_df.copy()
    mdf = rdf.groupby(conditions).apply(get_mean_sem_trace)
    mdf = mdf[['mean_response', 'sem_response', 'mean_trace', 'sem_trace',
               'trace_timestamps', 'mean_responses', 'mean_baseline', 'sem_baseline']]
    mdf = mdf.reset_index()
    # save response window duration as a column for reference
    mdf['response_window_duration'] = response_window_duration

    if get_pref_stim:
        if ('image_name' in conditions) or ('change_image_name' in conditions) or ('prior_image_name' in conditions):
            mdf = annotate_mean_df_with_pref_stim(
                mdf, exclude_omitted_from_pref_stim)
            print(mdf)

    try:
        mdf = annotate_mean_df_with_fano_factor(mdf)
        mdf = annotate_mean_df_with_time_to_peak(mdf, window, frame_rate)
        mdf = annotate_mean_df_with_p_value(
            mdf, window, response_window_duration, frame_rate)
        mdf = annotate_mean_df_with_sd_over_baseline(
            mdf, window, response_window_duration, frame_rate)
    except Exception as e:  # NOQA E722
        print(e)
        pass

    if 'p_value_gray_screen' in rdf.keys():
        fraction_significant_p_value_gray_screen = rdf.groupby(conditions).apply(
            get_fraction_significant_p_value_gray_screen)
        fraction_significant_p_value_gray_screen = fraction_significant_p_value_gray_screen.reset_index()
        mdf['fraction_significant_p_value_gray_screen'] = fraction_significant_p_value_gray_screen.fraction_significant_p_value_gray_screen

    try:
        reliability = rdf.groupby(conditions).apply(
            compute_reliability, window, response_window_duration, frame_rate)
        reliability = reliability.reset_index()
        mdf['reliability'] = reliability.reliability
        mdf['correlation_values'] = reliability.correlation_values
        # print('done computing reliability')
    except Exception as e:
        print('failed to compute reliability')
        print(e)
        pass

    if 'index' in mdf.keys():
        mdf = mdf.drop(columns=['index'])
    return mdf


####################################################################################################
# trace metrics
####################################################################################################


def get_successive_frame_list(timepoints_array, timestanps):
    # This is a modification of get_nearest_frame for speedup
    #  This implementation looks for the first 2p frame consecutive to the stim
    successive_frames = np.searchsorted(timestanps, timepoints_array)

    return successive_frames


def get_trace_around_timepoint(timepoint, trace, timestamps, window, frame_rate):
    #   frame_for_timepoint = get_nearest_frame(timepoint, timestamps)
    frame_for_timepoint = get_successive_frame_list(timepoint, timestamps)
    lower_frame = frame_for_timepoint + (window[0] * frame_rate)
    upper_frame = frame_for_timepoint + (window[1] * frame_rate)
    trace = trace[int(lower_frame):int(upper_frame)]
    timepoints = timestamps[int(lower_frame):int(upper_frame)]
    return trace, timepoints


def get_responses_around_event_times(trace, timestamps, event_times, frame_rate, window=[-2, 3]):
    responses = []
    for event_time in event_times:
        response, times = get_trace_around_timepoint(event_time, trace, timestamps,
                                                     frame_rate=frame_rate, window=window)
        responses.append(response)
    responses = np.asarray(responses)
    return responses


def get_mean_in_window(trace, window, frame_rate, use_events=False):
    mean = np.nanmean(trace[int(window[0] * frame_rate): int(window[1] * frame_rate)])
    return mean


def get_sd_in_window(trace, window, frame_rate):
    std = np.std(
        trace[int(window[0] * frame_rate): int(window[1] * frame_rate)])
    return std


def get_n_nonzero_in_window(trace, window, frame_rate):
    datapoints = trace[int(np.round(window[0] * frame_rate)): int(np.round(window[1] * frame_rate))]
    n_nonzero = len(np.where(datapoints > 0)[0])
    return n_nonzero


def get_sd_over_baseline(trace, response_window, baseline_window, frame_rate):
    baseline_std = get_sd_in_window(trace, baseline_window, frame_rate)
    response_mean = get_mean_in_window(trace, response_window, frame_rate)
    return response_mean / (baseline_std)


def get_p_val(trace, response_window, frame_rate):
    from scipy import stats
    response_window_duration = response_window[1] - response_window[0]
    baseline_end = int(response_window[0] * frame_rate)
    baseline_start = int((response_window[0] - response_window_duration) * frame_rate)
    stim_start = int(response_window[0] * frame_rate)
    stim_end = int((response_window[0] + response_window_duration) * frame_rate)
    (_, p) = stats.f_oneway(trace[baseline_start:baseline_end], trace[stim_start:stim_end])
    return p

####################################################################################################
# metrics for grouped stim_response_df
####################################################################################################


# TODO: clean + document
def get_mean_sem_trace(group):
    mean_response = np.mean(group['mean_response'])
    mean_baseline = np.mean(group['baseline_response'])
    mean_responses = group['mean_response'].values
    sem_response = np.std(group['mean_response'].values) / \
        np.sqrt(len(group['mean_response'].values))
    sem_baseline = np.std(group['baseline_response'].values) / \
        np.sqrt(len(group['baseline_response'].values))
    mean_trace = np.mean(group['trace'], axis=0)
    sem_trace = np.std(group['trace'].values) / \
        np.sqrt(len(group['trace'].values))
    trace_timestamps = np.mean(group['trace_timestamps'], axis=0)
    return pd.Series({'mean_response': mean_response, 'sem_response': sem_response,
                      'mean_baseline': mean_baseline, 'sem_baseline': sem_baseline,
                      'mean_trace': mean_trace, 'sem_trace': sem_trace,
                      'trace_timestamps': trace_timestamps,
                      'mean_responses': mean_responses})


# TODO: clean + document
def get_mean_sem(group):
    mean_response = np.mean(group['mean_response'])
    sem_response = np.std(group['mean_response'].values) / \
        np.sqrt(len(group['mean_response'].values))
    return pd.Series({'mean_response': mean_response, 'sem_response': sem_response})


# TODO: clean + document
def get_fraction_significant_trials(group):
    fraction_significant_trials = len(
        group[group.p_value < 0.05]) / float(len(group))
    return pd.Series({'fraction_significant_trials': fraction_significant_trials})


# TODO: clean + document
def get_fraction_significant_p_value_gray_screen(group):
    fraction_significant_p_value_gray_screen = len(
        group[group.p_value_gray_screen < 0.05]) / float(len(group))
    return pd.Series({'fraction_significant_p_value_gray_screen': fraction_significant_p_value_gray_screen})


# TODO: clean + document
def get_fraction_significant_p_value_omission(group):
    fraction_significant_p_value_omission = len(
        group[group.p_value_omission < 0.05]) / float(len(group))
    return pd.Series({'fraction_significant_p_value_omission': fraction_significant_p_value_omission})


# TODO: clean + document
def get_fraction_significant_p_value_stimulus(group):
    fraction_significant_p_value_stimulus = len(
        group[group.p_value_stimulus < 0.05]) / float(len(group))
    return pd.Series({'fraction_significant_p_value_stimulus': fraction_significant_p_value_stimulus})


# TODO: clean + document
def get_fraction_active_trials(group):
    fraction_active_trials = len(
        group[group.mean_response > 0.05]) / float(len(group))
    return pd.Series({'fraction_active_trials': fraction_active_trials})


# TODO: clean + document
def get_fraction_responsive_trials(group):
    fraction_responsive_trials = len(
        group[(group.p_value_baseline < 0.05)]) / float(len(group))
    return pd.Series({'fraction_responsive_trials': fraction_responsive_trials})


# TODO: clean + document
def get_fraction_nonzero_trials(group):
    fraction_nonzero_trials = len(
        group[group.n_events > 0]) / float(len(group))
    return pd.Series({'fraction_nonzero_trials': fraction_nonzero_trials})


# TODO: clean + document
def compute_reliability_vectorized(traces):
    '''
    Compute average pearson correlation between pairs of rows of the input matrix.
    Args:
        traces(np.ndarray): trace array with shape m*n, with m traces and n trace timepoints
    Returns:
        reliability (float): Average correlation between pairs of rows
    '''
    # Compute m*m pearson product moment correlation matrix between rows of input.
    # This matrix is 1 on the diagonal (correlation with self) and mirrored across
    # the diagonal (corr(A, B) = corr(B, A))
    corrmat = np.corrcoef(traces)
    # We want the inds of the lower triangle, without the diagonal, to average
    m = traces.shape[0]
    lower_tri_inds = np.where(np.tril(np.ones([m, m]), k=-1))
    # Take the lower triangle values from the corrmat and averge them
    correlation_values = list(corrmat[lower_tri_inds[0], lower_tri_inds[1]])
    reliability = np.nanmean(correlation_values)
    return reliability, correlation_values


# TODO: clean + document
def compute_reliability(group, window=[-3, 3], response_window_duration=0.5, frame_rate=30.):
    # computes trial to trial correlation across input traces in group,
    # only for portion of the trace after the change time or flash onset time

    onset = int(np.abs(window[0]) * frame_rate)
    response_window = [onset, onset + (int(response_window_duration * frame_rate))]
    traces = group['trace'].values
    traces = np.vstack(traces)
    if traces.shape[0] > 5:
        # limit to response window
        traces = traces[:, response_window[0]:response_window[1]]
        reliability, correlation_values = compute_reliability_vectorized(
            traces)
    else:
        reliability = np.nan
        correlation_values = []
    return pd.Series({'reliability': reliability, 'correlation_values': correlation_values})


####################################################################################################
# Annotate various data frames
####################################################################################################


# TODO: clean + document
def get_time_to_peak(trace, window=[-4, 8], frame_rate=30.):
    response_window_duration = 0.75
    response_window = [np.abs(window[0]), np.abs(
        window[0]) + response_window_duration]
    response_window_trace = trace[int(
        response_window[0] * frame_rate):(int(response_window[1] * frame_rate))]
    peak_response = np.amax(response_window_trace)
    peak_frames_from_response_window_start = np.where(
        response_window_trace == np.amax(response_window_trace))[0][0]
    time_to_peak = peak_frames_from_response_window_start / float(frame_rate)
    return peak_response, time_to_peak


# TODO: clean + document
def annotate_mean_df_with_time_to_peak(mean_df, window=[-4, 8], frame_rate=30.):
    ttp_list = []
    peak_list = []
    for idx in mean_df.index:
        mean_trace = mean_df.iloc[idx].mean_trace
        peak_response, time_to_peak = get_time_to_peak(
            mean_trace, window=window, frame_rate=frame_rate)
        ttp_list.append(time_to_peak)
        peak_list.append(peak_response)
    mean_df['peak_response'] = peak_list
    mean_df['time_to_peak'] = ttp_list
    return mean_df


# TODO: clean + document
def annotate_mean_df_with_fano_factor(mean_df):
    ff_list = []
    for idx in mean_df.index:
        mean_responses = mean_df.iloc[idx].mean_responses
        sd = np.nanstd(mean_responses)
        mean_response = np.nanmean(mean_responses)
        # take abs value to account for negative mean_response
        fano_factor = np.abs((sd * 2) / mean_response)
        ff_list.append(fano_factor)
    mean_df['fano_factor'] = ff_list
    return mean_df


# TODO: clean + document
def annotate_mean_df_with_p_value(mean_df, window=[-4, 8], response_window_duration=0.5, frame_rate=30.):
    response_window = [np.abs(window[0]), np.abs(
        window[0]) + response_window_duration]
    p_val_list = []
    for idx in mean_df.index:
        mean_trace = mean_df.iloc[idx].mean_trace
        p_value = get_p_val(mean_trace, response_window, frame_rate)
        p_val_list.append(p_value)
    mean_df['p_value'] = p_val_list
    return mean_df


# TODO: clean + document
def annotate_mean_df_with_sd_over_baseline(mean_df, window=[-4, 8], response_window_duration=0.5, frame_rate=30.):
    response_window = [np.abs(window[0]), np.abs(
        window[0]) + response_window_duration]
    baseline_window = [
        np.abs(window[0]) - response_window_duration, (np.abs(window[0]))]
    sd_list = []
    for idx in mean_df.index:
        mean_trace = mean_df.iloc[idx].mean_trace
        sd = get_sd_over_baseline(
            mean_trace, response_window, baseline_window, frame_rate)
        sd_list.append(sd)
    mean_df['sd_over_baseline'] = sd_list
    return mean_df


# TODO: clean + document
def annotate_mean_df_with_pref_stim(mean_df, exclude_omitted_from_pref_stim=True):
    if 'prior_image_name' in mean_df.keys():
        image_name = 'prior_image_name'
    elif 'image_name' in mean_df.keys():
        image_name = 'image_name'
    else:
        image_name = 'change_image_name'
    mdf = mean_df.reset_index()
    mdf['pref_stim'] = False
    if 'cell_specimen_id' in mdf.keys():
        cell_key = 'cell_specimen_id'
    else:
        cell_key = 'cell_roi_id'
    for cell in mdf[cell_key].unique():
        mc = mdf[(mdf[cell_key] == cell)]
        if exclude_omitted_from_pref_stim:
            if 'omitted' in mdf[image_name].unique():
                mc = mc[mc[image_name] != 'omitted']
        pref_image = mc[(mc.mean_response == np.max(
            mc.mean_response.values))][image_name].values[0]
        row = mdf[(mdf[cell_key] == cell) & (
            mdf[image_name] == pref_image)].index
        mdf.loc[row, 'pref_stim'] = True
    return mdf


# TODO: clean + document
def annotate_trial_response_df_with_pref_stim(trial_response_df):
    rdf = trial_response_df.copy()
    rdf['pref_stim'] = False
    if 'cell_specimen_id' in rdf.keys():
        cell_key = 'cell_specimen_id'
    else:
        cell_key = 'cell'
    mean_response = rdf.groupby(
        [cell_key, 'change_image_name']).apply(get_mean_sem_trace)
    m = mean_response.unstack()
    for cell in m.index:
        image_index = np.where(m.loc[cell]['mean_response'].values == np.max(
            m.loc[cell]['mean_response'].values))[0][0]
        pref_image = m.loc[cell]['mean_response'].index[image_index]
        trials = rdf[(rdf[cell_key] == cell) & (
            rdf.change_image_name == pref_image)].index
        for trial in trials:
            rdf.loc[trial, 'pref_stim'] = True
    return rdf


# TODO: clean + document
def annotate_flash_response_df_with_pref_stim(fdf):
    fdf = fdf.reset_index()
    if 'cell_specimen_id' in fdf.keys():
        cell_key = 'cell_specimen_id'
    else:
        cell_key = 'cell'
    fdf['pref_stim'] = False
    mean_response = fdf.groupby([cell_key, 'image_name']).apply(get_mean_sem)
    m = mean_response.unstack()
    for cell in m.index:
        image_index = \
            np.where(m.loc[cell]['mean_response'].values == np.nanmax(
                m.loc[cell]['mean_response'].values))[0][0]
        pref_image = m.loc[cell]['mean_response'].index[image_index]
        trials = fdf[(fdf[cell_key] == cell) & (
            fdf.image_name == pref_image)].index
        for trial in trials:
            fdf.loc[trial, 'pref_stim'] = True
    return fdf


# TODO: clean + document
def annotate_flashes_with_reward_rate(dataset):
    last_time = 0
    reward_rate_by_frame = []
    trials = dataset.trials[dataset.trials.trial_type != 'aborted']
    flashes = dataset.stimulus_table.copy()
    for change_time in trials.change_time.values:
        reward_rate = trials[trials.change_time == change_time].reward_rate.values[0]
        for start_time in flashes.start_time:
            if (start_time < change_time) and (start_time > last_time):
                reward_rate_by_frame.append(reward_rate)
                last_time = start_time
    # fill the last flashes with last value
    for i in range(len(flashes) - len(reward_rate_by_frame)):
        reward_rate_by_frame.append(reward_rate_by_frame[-1])
    flashes['reward_rate'] = reward_rate_by_frame
    return flashes
