import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import scipy
from mindscope_utilities.visual_behavior_ophys import data_formatting


# TODO: Make them work with events as well.
# Need to consider timepoint alignment between events.
# Functions to get trace_df and calculate correlations
def get_trace_df_all(lamf_group, session_name, trace_type='dff'):
    """Get a dataframe of all traces from a session (all ophys trace, either dff or events)

    Parameters
    ----------
    lamf_group : ExperimentGroup
        ExperimentGroup object
    session_name : str
        session name
    trace_type : str, optional
        'dff' or 'events', by default 'dff'
    
    Returns
    -------
    trace_df : pd.DataFrame
        dataframe of all traces from a session
    
    """
    oeids = np.sort(lamf_group.expt_table[lamf_group.expt_table.session_name==session_name].index.values)
    # load all the traces from this session
    trace_df = pd.DataFrame(columns=['cell_specimen_id', 'trace', 'timepoints', 'oeid', 'target_region', 'depth_order', 'bisect_layer']).set_index('cell_specimen_id')
    for oeid in oeids:
        temp_df = pd.DataFrame(columns=['cell_specimen_id', 'trace', 'timepoints', 'oeid', 'target_region', 'depth_order', 'bisect_layer'])
        if trace_type == 'dff':
            trace_df = lamf_group.experiments[oeid].dff_traces
            trace = trace_df.dff.values
        elif trace_type == 'events':
            trace_df = lamf_group.experiments[oeid].events
            trace = trace_df.events.values
        else:
            raise ValueError('trace_type must be either dff or events')
        csid = trace_df.index.values
        temp_df['cell_specimen_id'] = csid
        temp_df['trace'] = trace
        temp_df['timepoints'] = lamf_group.experiments[oeid].ophys_timestamps
        temp_df['oeid'] = oeid
        temp_df['target_region'] = lamf_group.expt_table.loc[oeid].targeted_structure
        temp_df['depth_order'] = lamf_group.expt_table.loc[oeid].depth_order
        temp_df['bisect_layer'] = lamf_group.expt_table.loc[oeid].bisect_layer
        temp_df.set_index('cell_specimen_id', inplace=True)
        temp_df.sort_index(inplace=True)
        trace_df = pd.concat([trace_df, temp_df])
    return trace_df


def get_trace_df_no_task(lamf_group, session_name, trace_type='dff'):
    """Get a dataframe of traces from a session (all ophys trace, either dff or events) with no task
    Assume 5 min gray screen before and after task
    and 5 min fingerprint (movie-watching; 30 sec 10 iterations) imaging at the end
    Some sessions don't have them.

    Parameters
    ----------
    lamf_group : ExperimentGroup
        ExperimentGroup object
    session_name : str
        session name
    trace_type : str, optional
        'dff' or 'events', by default 'dff'
    
    Returns
    -------
    trace_df_gray_pre_task : pd.DataFrame
        dataframe of traces from gray screen before task
    trace_df_gray_post_task : pd.DataFrame
        dataframe of traces from gray screen after task
    trace_df_fingerprint : pd.DataFrame
        dataframe of traces from fingerprint (movie-watching; 30 sec 10 iterations) imaging at the end
    """
    gray_period = 5 * 60  # 5 minutes
    oeids = np.sort(lamf_group.expt_table[lamf_group.expt_table.session_name==session_name].index.values)
    stim_df = data_formatting.annotate_stimuli(lamf_group.experiments[oeids[0]])  # First experiment represents the session stimulus presentations
    
    # Match the # of indices from each experiment
    start_inds = []
    end_inds = []
    post_gray_end_inds = []
    for oeid in oeids:
        timestamps = lamf_group.experiments[oeid].ophys_timestamps
        
        first_stim_start_time = stim_df.start_time.values[0]
        if first_stim_start_time > gray_period - 30:  # 30 is an arbitrary buffer
            run_pre_gray = True
            first_stim_start_frame = np.where(timestamps > first_stim_start_time)[0][0]
            start_inds.append(first_stim_start_frame)
        else:
            run_pre_gray = False

        last_time = timestamps[-1]
        last_stim_end_time = stim_df.stop_time.values[-1]
        if last_time - last_stim_end_time > gray_period - 30:  # 30 is an arbitrary buffer
            run_post_gray = True
            last_stim_end_frame = np.where(timestamps > last_stim_end_time)[0][0]
            end_inds.append(last_stim_end_frame)
            post_gray_end_time = last_stim_end_time + gray_period

            if last_time - post_gray_end_time > -60:  # -60 is an arbitrary buffer
                run_fingerprint = True
                post_gray_end_frame = np.where(timestamps > post_gray_end_time)[0][0]
                post_gray_end_inds.append(post_gray_end_frame)
            else:
                run_fingerprint = False
        else:
            run_post_gray = False
        
    if run_pre_gray:
        min_start_ind = np.min(start_inds)
    if run_post_gray:
        max_end_ind = np.max(end_inds)
        min_post_gray_end_ind = np.min(post_gray_end_inds)

    # Initialize
    trace_df_gray_pre_task = pd.DataFrame(columns=['cell_specimen_id', 'trace', 'timepoints', 'oeid', 'target_region', 'depth_order', 'bisect_layer']).set_index('cell_specimen_id')
    trace_df_gray_post_task = pd.DataFrame(columns=['cell_specimen_id', 'trace', 'timepoints', 'oeid', 'target_region', 'depth_order', 'bisect_layer']).set_index('cell_specimen_id')
    trace_df_fingerprint = pd.DataFrame(columns=['cell_specimen_id', 'trace', 'timepoints', 'oeid', 'target_region', 'depth_order', 'bisect_layer']).set_index('cell_specimen_id')
    for oeid in oeids:
        # Initialize for each experiment
        temp_df = pd.DataFrame(columns=['cell_specimen_id', 'trace', 'timepoints', 'oeid', 'target_region', 'depth_order', 'bisect_layer'])

        if trace_type == 'dff':
            trace_df = lamf_group.experiments[oeid].dff_traces
        elif trace_type == 'events':
            trace_df = lamf_group.experiments[oeid].events
        else:
            raise ValueError('trace_type must be dff or events')
        timestamps = lamf_group.experiments[oeid].ophys_timestamps
        # cell_specimen_id, oeid, target_region, depth_order, bisect_layer are the same
        # across epochs in no-task window
        # Only switch trace and timepoints for each epoch
        csid = trace_df.index.values
        temp_df['cell_specimen_id'] = csid        
        temp_df['oeid'] = oeid
        temp_df['target_region'] = lamf_group.expt_table.loc[oeid].targeted_structure
        temp_df['depth_order'] = lamf_group.expt_table.loc[oeid].depth_order
        temp_df['bisect_layer'] = lamf_group.expt_table.loc[oeid].bisect_layer
        temp_df.set_index('cell_specimen_id', inplace=True)
        if run_pre_gray:
            if trace_type=='dff':
                temp_df['trace'] = trace_df.dff.apply(lambda x: x[:min_start_ind]).values
            else:  # Other types already dealt with from above
                temp_df['trace'] = trace_df.events.apply(lambda x: x[:min_start_ind]).values
            temp_df['timepoints'] = [timestamps[:min_start_ind]] * len(csid)
            sorted_temp_df = temp_df.sort_index(inplace=False)
            trace_df_gray_pre_task = pd.concat([trace_df_gray_pre_task, sorted_temp_df])
        if run_post_gray:
            if trace_type=='dff':
                temp_df['trace'] = trace_df.dff.apply(lambda x: x[max_end_ind:min_post_gray_end_ind]).values
            else:
                temp_df['trace'] = trace_df.events.apply(lambda x: x[max_end_ind:min_post_gray_end_ind]).values
            temp_df['timepoints'] = [timestamps[max_end_ind:min_post_gray_end_ind]] * len(csid)
            sorted_temp_df = temp_df.sort_index(inplace=False)
            trace_df_gray_post_task = pd.concat([trace_df_gray_post_task, sorted_temp_df])
            if run_fingerprint:
                if trace_type=='dff':
                    temp_df['trace'] = trace_df.dff.apply(lambda x: x[min_post_gray_end_ind:]).values
                else:
                    temp_df['trace'] = trace_df.events.apply(lambda x: x[min_post_gray_end_ind:]).values
                temp_df['timepoints'] = [timestamps[min_post_gray_end_ind:]] * len(csid)
                sorted_temp_df = temp_df.sort_index(inplace=False)
                trace_df_fingerprint = pd.concat([trace_df_fingerprint, sorted_temp_df])
    return trace_df_gray_pre_task, trace_df_gray_post_task, trace_df_fingerprint


# A function to get non-auto-rewarded start and end times
def get_non_auto_rewarded_start_end_times(stim_df):
    """ Get start and end times of non-auto-rewarded stimuli

    Parameters
    ----------
    stim_df: pandas.DataFrame
        A dataframe containing stimulus information
    
    Returns
    -------
    start_times: list
        A list of start times of non-auto-rewarded stimuli
    stop_times: list
        A list of stop times of non-auto-rewarded stimuli
    """
    auto_rewarded_inds = np.where(stim_df.auto_rewarded == True)[0]  # noqa: E712
    diff_ar_inds = np.diff(auto_rewarded_inds)
    skip_ar_inds = np.where(diff_ar_inds > 1)[0]
    start_times = []  # inclusive
    stop_times = []  # inclusive
    for skip_ar_ind in skip_ar_inds:
        start_times.append(stim_df.iloc[auto_rewarded_inds[skip_ar_ind] + 1].start_time)  # removing those before the first auto-rewarded stim
        stop_times.append(stim_df.iloc[auto_rewarded_inds[skip_ar_ind + 1]].stop_time)
    if auto_rewarded_inds[-1] != len(stim_df) - 1:
        start_times.append(stim_df.iloc[auto_rewarded_inds[-1] + 1].start_time)
        stop_times.append(stim_df.iloc[-1].stop_time)
    return start_times, stop_times


def get_start_end_inds(start_times, stop_times, lamf_group, oeids):
    """ Get start and end indices matching to the start and stop times
    from lamf_group ophys_experiment_ids

    Parameters
    ----------
    start_times: list
        A list of start times
    stop_times: list
        A list of stop times
    lamf_group : ExperimentGroup
        ExperimentGroup object
    oeids: list
        A list of ophys_experiment_ids
    
    Returns
    -------
    max_start_inds: list
        A list of start indices of non-auto-rewarded stimuli
    min_end_inds: list
        A list of end indices of non-auto-rewarded stimuli
    """
    # The behavior is different with the new version of allensdk
    # TODO: when using updated version of allensdk (>2.13.6), change the code accordingly:
    max_start_inds = []
    min_end_inds = []  # stop for time, end for frame or index
    for start_time, end_time in zip(start_times, stop_times):
        start_inds = []
        end_inds = []
        for oeid in oeids:    
            timestamps = lamf_group.experiments[oeid].ophys_timestamps
            start_inds.append(np.where(timestamps >= start_time)[0][0])
            end_inds.append(np.where(timestamps <= end_time)[0][-1])
        max_start_inds.append(np.max(start_inds))
        min_end_inds.append(np.min(end_inds))
    return max_start_inds, min_end_inds


def get_trace_df_task(lamf_group, session_name, remove_auto_rewarded=True):
    """ Get trace_df for a given session

    Parameters
    ----------
    lamf_group : ExperimentGroup
        ExperimentGroup object
    session_name: str
        A string of session name
    remove_auto_rewarded: bool, optional
        Whether to remove auto-rewarded stimuli, default True

    Returns
    -------
    trace_df: pandas.DataFrame
        A dataframe containing trace information during the whole task
    """
    
    oeids = np.sort(lamf_group.expt_table[lamf_group.expt_table.session_name==session_name].index.values)
    stim_df = data_formatting.annotate_stimuli(lamf_group.experiments[oeids[0]])  # First experiment represents the session stimulus presentations
    # Only works on old allensdk version (or lamf_hacks branch of MJD's fork)
    # TODO: when using updated version of allensdk (>2.13.6), change the code accordingly:
    if remove_auto_rewarded:
        start_times, stop_times = get_non_auto_rewarded_start_end_times(stim_df)  # across all experiments
    else:
        start_times = stim_df.start_time.values[0]
        stop_times = stim_df.stop_time.values[-1]
    # To match the frame indices across experiments
    max_start_inds, min_end_inds = get_start_end_inds(start_times, stop_times, lamf_group, oeids)
    trace_df = pd.DataFrame(columns=['cell_specimen_id', 'trace', 'timepoints', 'oeid', 'target_region', 'depth_order', 'bisect_layer']).set_index('cell_specimen_id')
    for oeid in oeids:
        temp_df = pd.DataFrame(columns=['cell_specimen_id', 'trace', 'timepoints', 'oeid', 'target_region', 'depth_order', 'bisect_layer'])

        dff_df = lamf_group.experiments[oeid].dff_traces
        test_dff = dff_df.iloc[0].dff
        timestamps = lamf_group.experiments[oeid].ophys_timestamps
        assert len(test_dff) == len(timestamps)

        test_dff_crop = np.concatenate([test_dff[msi: mei] for msi, mei in zip(max_start_inds, min_end_inds)])
        timepoints = np.concatenate([timestamps[msi: mei] for msi, mei in zip(max_start_inds, min_end_inds)])
        assert len(timepoints) == len(test_dff_crop)

        csid = dff_df.index.values
        temp_df['cell_specimen_id'] = csid
        temp_df['trace'] = dff_df.dff.apply(lambda x: np.concatenate([x[msi: mei] for msi, mei in zip(max_start_inds, min_end_inds)])).values
        temp_df['timepoints'] = [timepoints] * len(csid)
        temp_df['oeid'] = oeid
        temp_df['target_region'] = lamf_group.expt_table.loc[oeid].targeted_structure
        temp_df['depth_order'] = lamf_group.expt_table.loc[oeid].depth_order
        temp_df['bisect_layer'] = lamf_group.expt_table.loc[oeid].bisect_layer
        temp_df.set_index('cell_specimen_id', inplace=True)
        # sort temp_df by cell_specimen_id
        temp_df.sort_index(inplace=True)
        trace_df = pd.concat([trace_df, temp_df])
    return trace_df


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
    overlapping_column_names = np.intersect1d(stim_df.keys().tolist(), stim_df_2.keys().tolist())
    stim_df_2 = stim_df_2.drop(columns=overlapping_column_names)
    stim_df = stim_df.join(stim_df_2)
    return stim_df


def get_event_annotated_response_df(exp, event_type, data_type='dff', image_order=3, inter_image_interval=0.75, output_sampling_rate=20):
    """ Get stimulus presentations with event annotations
    Merge dataframe from data_formatting.get_annotated_stimulus_presentations and data_formatting.annotate_stimuli
    to data_formatting.get_stimulus_response_df

    Parameters
    ----------
    exp: BehaviorOphysExperiment
        Behavior ophys experiment object
    event_type: str
        Visual stimulus type, e.g. 'images', 'images>n-changes', 'changes', 'omissions', etc.
    data_type: str, optional
        Data type, e.g., 'dff' and 'events', default 'dff'
    image_order: int, optional
        Image order for images>n parameters, default 3
    inter_image_interval: float, optional
        Inter image interval, default 0.75
    output_sampling_rate: float, optional
        Output sampling rate, default 20

    Returns
    -------
    response_df: pandas.DataFrame
        A dataframe containing stimulus presentations with event annotations
    """
    response_df = data_formatting.get_stimulus_response_df(exp, data_type=data_type, event_type=event_type, image_order=image_order,
                                                           time_window=[0, inter_image_interval], output_sampling_rate=output_sampling_rate)
    stim_df = get_all_annotated_stimulus_presentations(exp)
    response_df = response_df.merge(stim_df, how='left', on='stimulus_presentations_id', validate='m:1')
    response_df['oeid'] = exp.ophys_experiment_id

    return response_df


def get_trace_df_event(lamf_group, session_name, event_type, data_type='dff', image_order=3,
                       inter_image_interval=0.75, output_sampling_rate=20, remove_auto_rewarded=True):
    """ Get trace dataframe for a given session and event type
    Parameters
    ----------
    lamf_group: ExperimentGroup
        ExperimentGroup object
    session_name: str
        Session name
    event_type: str
        Visual stimulus type, e.g. 'images', 'images>n-changes', 'changes', 'omissions', etc.
    data_type: str, optional
        Data type, e.g., 'dff' and 'events', default 'dff'
    image_order: int, optional
        Image order for images>n parameters, default 3
    inter_image_interval: float, optional
        Inter image interval, default 0.75
    output_sampling_rate: float, optional
        Output sampling rate, default 20
    remove_auto_rewarded: bool, optional
        Remove auto rewarded trials, default True
    
    Returns
    -------
    trace_df: pandas.DataFrame
        A dataframe containing traces for a given session and event type
    """

    oeids = np.sort(lamf_group.expt_table[lamf_group.expt_table.session_name==session_name].index.values)
    # load all the traces from this session
    trace_df = pd.DataFrame(columns=['cell_specimen_id', 'trace', 'oeid', 'target_region', 'depth_order', 'bisect_layer']).set_index('cell_specimen_id')
    for oeid in oeids:
        if event_type=='images>n-changes':
            response_df = get_event_annotated_response_df(lamf_group.experiments[oeid], data_type=data_type, event_type=event_type, image_order=image_order,
                                                          inter_image_interval=inter_image_interval, output_sampling_rate=output_sampling_rate)
        else:
            response_df = get_event_annotated_response_df(lamf_group.experiments[oeid], data_type=data_type, event_type=event_type,
                                                          inter_image_interval=inter_image_interval, output_sampling_rate=output_sampling_rate)
        
        if len(response_df) > 0:
            if remove_auto_rewarded:
                response_df = response_df[response_df.auto_rewarded==False]  # noqa: E712

            first_timestamps = response_df.trace_timestamps.values[0]
            assert response_df.trace_timestamps.apply(lambda x: np.all(x==first_timestamps)).all()
            start_index = np.where(first_timestamps>=0)[0][0]  
            end_index = np.where(first_timestamps<=inter_image_interval)[0][-1]  # The last index is going to be not included

            csids = response_df.cell_specimen_id.unique()
            trace_all = []
            for csid in csids:
                csid_df = response_df[response_df.cell_specimen_id==csid]
                csid_trace = np.concatenate(csid_df.trace.apply(lambda x: x[start_index:end_index]).values)
                trace_all.append(csid_trace)
            assert np.diff([len(trace) for trace in trace_all]).any() == False  # noqa: E712
            timepoints = np.concatenate([st + tts[1:] for st, tts in zip(csid_df.start_time.values, csid_df.trace_timestamps.values)])
            assert len(timepoints) == len(trace_all[0])
            
            temp_df = pd.DataFrame(columns=['cell_specimen_id', 'trace', 'oeid', 'target_region', 'depth_order', 'bisect_layer'])
            temp_df['cell_specimen_id'] = csids
            temp_df['trace'] = trace_all
            temp_df['timepoints'] = [timepoints] * len(csids)
            temp_df['oeid'] = oeid
            temp_df['target_region'] = lamf_group.expt_table.loc[oeid].targeted_structure
            temp_df['depth_order'] = lamf_group.expt_table.loc[oeid].depth_order
            temp_df['bisect_layer'] = lamf_group.expt_table.loc[oeid].bisect_layer
            temp_df.set_index('cell_specimen_id', inplace=True)
            temp_df.sort_index(inplace=True)
            trace_df = pd.concat([trace_df, temp_df])
    return trace_df


def get_correlation_matrices(trace_df, nan_frame_prop_threshold=0.2, nan_cell_prop_threshold=0.2):
    """ Get correlation matrices for a given trace dataframe
    If number of frames with nan values is small, it's okay to just remove those frames
    If number of frames with nan values is large, something is wrong with the data
        If number of cells with large number of nan frames is small, it's okay to just remove those cells

    Parameters
    ----------
    trace_df: pandas.DataFrame
        A dataframe containing traces for a given session and event type
    nan_frame_prop_threshold: float, optional
        Threshold for the proportion of nan frames in a cell, default 0.2
    nan_cell_prop_threshold: float, optional
        Threshold for the proportion of nan cells in a session, default 0.2
    
    Returns
    -------
    corr: np.ndarray (2d)
        Correlation matrix (No sorting, i.e., sorted by oeid and cell specimen id)
    corr_ordered: np.ndarray (2d)
        Correlation matrix (Sorted by mean correlation coefficient)    
    corr_ordered_by_region: np.ndarray (2d)
        Correlation matrix (Sorted by region and depth)
    xy_labels: list
        List of labels for corr_ordered_by_region
    xy_label_pos: list
        List of label positions for corr_ordered_by_region
    mean_corr_sorted_ind: np.ndarray (1d)
        Sorted indices of mean correlation coefficients (applied to corr to produce corr_ordered)
    remove_ind: np.ndarray (1d)
        Indices of cells to be removed due to large number of nan frames
    """
    trace_array = np.vstack(trace_df.trace.values)
    
    # Check if there are too many nan frames or cells with too many nan frames
    num_cell = len(trace_df)
    num_nan_frames_threshold = int(trace_array.shape[1] * nan_frame_prop_threshold)
    nan_frames = np.where(np.isnan(trace_array).sum(axis=0)>0)[0]
    num_nan_frames = len(nan_frames)

    remove_ind = None
    if num_nan_frames > num_nan_frames_threshold:
        num_nan_frames_each = np.isnan(trace_array).sum(axis=1)
        ind_many_nan_frames = np.where(num_nan_frames_each > num_nan_frames_threshold)[0]
        num_nan_cells = len(ind_many_nan_frames)
        if num_nan_cells / num_cell > nan_cell_prop_threshold:
            raise ValueError(f"Too many cells with nan frames > threshold {nan_frame_prop_threshold}: {num_nan_cells} out of {num_cell}")
        else:
            print(f"Removing {num_nan_cells} cells with nan frames proportion > threshold {nan_frame_prop_threshold}")
            remove_ind = ind_many_nan_frames
            trace_array = np.delete(trace_array, remove_ind, axis=0)
            trace_df = trace_df.reset_index()
            trace_df = trace_df.drop(trace_df.index[remove_ind])
            trace_df = trace_df.set_index('cell_specimen_id')

            nan_frames = np.where(np.isnan(trace_array).sum(axis=0)>0)[0]
            num_nan_frames = len(nan_frames)
            trace_array = np.delete(trace_array, nan_frames, axis=1)
            print(f"Removing {num_nan_frames} frames with nan values")
    else:
        print(f"Removing {num_nan_frames} frames with nan values")
        trace_array = np.delete(trace_array, nan_frames, axis=1)
    
    corr = np.corrcoef(trace_array)

    # sort by global mean correlation
    mean_corr = np.nanmean(corr, axis=0)
    mean_corr_sorted_ind = np.argsort(mean_corr)[::-1]
    corr_ordered = corr[mean_corr_sorted_ind, :][:, mean_corr_sorted_ind]

    numcell_cumsum = np.cumsum([x[0] for x in trace_df.groupby(['oeid']).count().values])
    xy_ticks = np.insert(numcell_cumsum, 0, 0)
    xy_labels = [f"{x}-{y}" for x, y in zip(trace_df.groupby(['oeid']).first().target_region.values, trace_df.groupby(['oeid']).first().bisect_layer.values)]
    xy_label_pos = xy_ticks[:-1] + np.diff(xy_ticks) / 2

    # order the corr matrix by the mean correlation of each cell within each oeid
    trace_array_ordered_by_region = np.zeros_like(trace_array)
    for i in range(len(numcell_cumsum)):
        # find the order within each oeid
        within_mean_corr = np.mean(corr[xy_ticks[i]: xy_ticks[i + 1], xy_ticks[i]: xy_ticks[i + 1]], axis=1)
        within_order = np.argsort(-within_mean_corr)  # descending order
        trace_array_ordered_by_region[xy_ticks[i]:xy_ticks[i + 1], :] = trace_array[xy_ticks[i]: xy_ticks[i + 1], :][within_order, :]
    corr_ordered_by_region = np.corrcoef(trace_array_ordered_by_region)

    return corr, corr_ordered, corr_ordered_by_region, xy_labels, xy_label_pos, mean_corr_sorted_ind, remove_ind


# Comparing between different epochs and events
# Functions to get all epoch trace dfs and correlation matrices
def _append_results(results, trace_df, epoch):
    """Append results from trace_df

    Parameters
    ----------
    results: dict
        Dictionary to store results
    trace_df: pd.DataFrame
        Dataframe containing trace and other information
    epoch: str
        Epoch name
    
    Returns
    -------
    results: dict
        Dictionary to store results
    """
    results['epochs'].append(epoch)
    results['trace_dfs'].append(trace_df)
    corr, corr_ordered, corr_ordered_by_region, xy_labels, xy_label_pos, sorted_ind, remove_ind = get_correlation_matrices(trace_df)
    results['corr_matrices'].append(corr)
    results['corr_ordered_matrices'].append(corr_ordered)
    results['corr_ordered_by_region_matrices'].append(corr_ordered_by_region)
    results['xy_label_pos'].append(xy_label_pos)
    results['xy_labels'].append(xy_labels)
    results['sorted_inds'].append(sorted_ind)
    return results


def get_all_epoch_trace_df_and_correlation_matrices(lamf_group, session_name, image_order=3,
                                                    inter_image_interval=0.75, output_sampling_rate=20):
    """Get all epoch trace dfs and correlation matrices

    Parameters
    ----------
    lamf_group: Experiment group
        Experiment group object
    session_name: str
        Session name
    image_order: int, optional
        Image order for 'images>n-change' event type, default 3
    inter_image_interval: float, optioanl
        Inter image interval, default 0.75
    output_sampling_rate: float
        Output sampling rate, default 20 (Hz)
    
    Returns
    -------
    epochs: list
        List of epochs
    trace_dfs: list
        List of trace dataframes
    corr_matrices: list
        List of correlation matrices
    corr_ordered_matrices: list
        List of correlation matrices ordered by global mean correlation
    corr_ordered_by_region_matrices: list
        List of correlation matrices ordered by mean correlation within each region and depth
    xy_label_pos_list: list
        List of x and y label positions for corr_ordered_by_region_matrices
    xy_labels_list: list
        List of x and y labels for corr_ordered_by_region_matrices
    sorted_inds_list: list
        List of sorted indices for corr_ordered_by_region_matrices
    """

    results = {'epochs': [],
               'trace_dfs': [],
               'corr_matrices': [],
               'corr_ordered_matrices': [],
               'corr_ordered_by_region_matrices': [],
               'xy_label_pos': [],
               'xy_labels': [],
               'sorted_inds': []}

    trace_task_df = get_trace_df_task(lamf_group, session_name)
    results = _append_results(results, trace_task_df, 'task')
    
    trace_graypre_df, trace_graypost_df, trace_fingerprint_df = get_trace_df_no_task(lamf_group, session_name)
    if len(trace_graypre_df) > 0:
        assert (trace_task_df.index.values - trace_graypre_df.index.values).any() == False  # noqa: E712
        results = _append_results(results, trace_graypre_df, 'graypre')

    if len(trace_graypost_df) > 0:
        assert (trace_task_df.index.values - trace_graypost_df.index.values).any() == False  # noqa: E712
        results = _append_results(results, trace_graypost_df, 'graypost')
    
    if len(trace_fingerprint_df) > 0:
        assert (trace_task_df.index.values - trace_fingerprint_df.index.values).any() == False  # noqa: E712
        results = _append_results(results, trace_fingerprint_df, 'fingerprint')
    
    events = ['images>n-changes', 'changes', 'omissions']
    for event in events:
        trace_event_df = get_trace_df_event(lamf_group, session_name=session_name, event_type=event,
                                            image_order=image_order, inter_image_interval=inter_image_interval, 
                                            output_sampling_rate=output_sampling_rate)
        if len(trace_event_df) > 0:
            assert (trace_task_df.index.values - trace_event_df.index.values).any() == False  # noqa: E712
            if event == 'images>n-changes':
                event = 'images'
            results = _append_results(results, trace_event_df, event)
    
    epochs, trace_dfs, corr_matrices, corr_ordered_matrices, corr_ordered_by_region_matrices, \
        xy_label_pos_list, xy_labels_list, sorted_inds_list = results.values()

    return epochs, trace_dfs, corr_matrices, corr_ordered_matrices, corr_ordered_by_region_matrices, \
        xy_label_pos_list, xy_labels_list, sorted_inds_list


def compare_correlation_matrices(compare_epochs, epochs, corr_matrices, corr_ordered_matrices, corr_ordered_by_region_matrices,
                                 xy_label_pos_list, xy_labels_list, sorted_inds_list, session_name, vmin=-0.4, vmax=0.8, cb_shrink_factor=0.7):
    """Compare correlation matrices
    Plot correlation matrices for different epochs and events
    along with their sorted correlation matrices by one correlation matrix (Reference matrix)

    Parameters
    ----------
    compare_epochs: list
        List of epochs to compare
    epochs: list
        List of epochs (results from get_all_epoch_trace_df_and_correlation_matrices)
    corr_matrices: list
        List of corr (results from get_all_epoch_trace_df_and_correlation_matrices)
    corr_ordered_matrices: list
        List of corr_ordered (results from get_all_epoch_trace_df_and_correlation_matrices)
    corr_ordered_by_region_matrices: list
        List of corr_ordered_by_region (results from get_all_epoch_trace_df_and_correlation_matrices)
    xy_label_pos_list: list
        List of xy_label_pos (results from get_all_epoch_trace_df_and_correlation_matrices)
    xy_labels_list: list
        List of xy_labels (results from get_all_epoch_trace_df_and_correlation_matrices)
    sorted_inds_list: list
        List of sorted_inds (results from get_all_epoch_trace_df_and_correlation_matrices)
    session_name: str
        Session name
    vmin: float, optional
        Minimum value for correlation matrix plot colorbar, default -0.4
    vmax: float, optional
        Maximum value for correlation matrix plot colorbar, default 0.8
    cb_shrink_factor: float, optional
        Colorbar shrink factor for sns.heatmap, default 0.7
    """

    num_rows = len(compare_epochs)
    inds = []
    for i in range(num_rows):
        inds.append(epochs.index(compare_epochs[i]))
    
    ref_sort_ind = sorted_inds_list[inds[0]]
    comp_sorted_by_ref = []
    for i in range(1, num_rows):
        comp_sorted_by_ref.append(corr_matrices[inds[i]][ref_sort_ind, :][:, ref_sort_ind])
    fig, ax = plt.subplots(num_rows, 3, figsize=(15, num_rows * 5))
    # Plot reference matrices
    sns.heatmap(corr_matrices[inds[0]], cmap='RdBu_r', square=True, cbar_kws={'shrink': cb_shrink_factor},
                vmin=vmin, vmax=vmax, ax=ax[0, 0])
    ax[0, 0].set_title(compare_epochs[0])
    sns.heatmap(corr_ordered_matrices[inds[0]], cmap='RdBu_r', square=True, cbar_kws={'shrink': cb_shrink_factor},
                vmin=vmin, vmax=vmax, ax=ax[0, 1])
    ax[0, 1].set_title(compare_epochs[0] + ' sorted')
    sns.heatmap(corr_ordered_by_region_matrices[inds[0]], cmap='RdBu_r', square=True, cbar_kws={'shrink': cb_shrink_factor}, 
                vmin=vmin, vmax=vmax, ax=ax[0, 2])
    ax[0, 2].set_title(compare_epochs[0] + ' sorted within region')

    # Plot comparison matrices
    for i in range(1, num_rows):
        sns.heatmap(corr_matrices[inds[i]], cmap='RdBu_r', square=True, cbar_kws={'shrink': cb_shrink_factor},
                    vmin=vmin, vmax=vmax, ax=ax[i, 0])
        ax[i, 0].set_title(compare_epochs[i])
        sns.heatmap(comp_sorted_by_ref[i - 1], cmap='RdBu_r', square=True, cbar_kws={'shrink': cb_shrink_factor},
                    vmin=vmin, vmax=vmax, ax=ax[i, 1])
        ax[i, 1].set_title(compare_epochs[i] + ' sorted by ' + compare_epochs[0])
        sns.heatmap(corr_ordered_by_region_matrices[inds[i]], cmap='RdBu_r', square=True, cbar_kws={'shrink': cb_shrink_factor},
                    vmin=vmin, vmax=vmax, ax=ax[i, 2])
        ax[i, 2].set_title(compare_epochs[i] + ' sorted within region')

    # Label axes
    num_cell = corr_matrices[inds[0]].shape[0]
    interval = 100 if num_cell>300 else 50
    xy_label_pos_ref = xy_label_pos_list[inds[0]]
    xy_labels_ref = xy_labels_list[inds[0]]
    for i in range(num_rows):
        for j in range(2):
            ax[i, j].set_xticks(range(0, num_cell, interval))
            ax[i, j].set_yticks(range(0, num_cell, interval))
            ax[i, j].set_xticklabels(range(0, num_cell, interval))
            ax[i, j].set_yticklabels(range(0, num_cell, interval))
        ax[i, 2].set_xticks(xy_label_pos_ref)
        ax[i, 2].set_xticklabels(xy_labels_ref, rotation=90)
        ax[i, 2].set_yticks(xy_label_pos_ref)
        ax[i, 2].set_yticklabels(xy_labels_ref, rotation=0)
    fig.suptitle(f'{session_name}\n{compare_epochs[0]} vs {compare_epochs[1]}')
    fig.tight_layout()

    return fig


def get_all_annotated_session_response_df(lamf_group, session_name, inter_image_interval=0.75, output_sampling_rate=20):
    """Get all response_df for a session from all experiments in a lamf_group

    Parameters
    ----------
    lamf_group: Experiment group object
        Experiment group object for Learning and mFISH project
    session_name: str
        Session name
    inter_image_interval: float, optional
        Inter image interval in seconds, default 0.75
    output_sampling_rate: float, optional
        Output sampling rate in Hz, default 20
    
    Returns
    -------
    response_df: pandas DataFrame
        DataFrame with stimulus presentations for all experiments in a session
    """
    oeids = np.sort(lamf_group.expt_table.query('session_name==@session_name').index.values)
    response_df = pd.DataFrame()
    for oeid in oeids:
        exp = lamf_group.experiments[oeid]
        response_df = response_df.append(get_event_annotated_response_df(exp, event_type='all', inter_image_interval=inter_image_interval, output_sampling_rate=output_sampling_rate))
    return response_df


def get_trace_df_event_from_all_response_df(response_df_session, lamf_group, event_type, image_order=3, inter_image_interval=0.75):
    """Get trace DataFrame for a particular event type from all experiments in a session

    Parameters
    ----------
    response_df_session: pandas DataFrame
        DataFrame with stimulus presentations for all experiments in a session
    lamf_group: Experiment group object
        Experiment group object for Learning and mFISH project
    event_type: str
        Event type, e.g., 'images', 'images-n-omissions', 'images-n-changes', 'images>n-changes'
    image_order: int, optional
        Image order for n in event_type, default 3
    
    Returns
    -------
    trace_df_event: pandas DataFrame
        DataFrame with traces for a particular event type from all experiments in a session
    """

    if event_type=='images-n-omissions':
        condition = (response_df_session['n_after_omission']==image_order) & \
                    (response_df_session['n_after_change'] > response_df_session['n_after_omission'])  # noqa E501
    elif event_type=='images-n-changes':
        condition = (response_df_session['n_after_change']==image_order) & \
                    ((response_df_session['n_after_omission'] > response_df_session['n_after_change']) | # noqa E501  
                        (response_df_session['n_after_omission'] == -1))  # for trials without omission
    elif event_type=='images>n-omissions':
        condition = (response_df_session['n_after_omission'] > image_order) & \
                    (response_df_session['n_after_change'] > response_df_session['n_after_omission'])
    elif event_type=='images>n-changes':
        condition = (response_df_session['n_after_change'] > image_order) & \
                    ((response_df_session['n_after_omission'] > image_order) |  # noqa E501
                        (response_df_session['n_after_omission'] == -1))  # for trials without omission
    elif event_type=='images-n-before-changes':
        condition = (response_df_session['n_before_change'] == image_order) & \
                    (response_df_session['n_after_omission'] == -1)  # Get trials without omission only
    elif event_type=='changes':
        condition = (response_df_session['n_after_change'] == 0)
    elif event_type=='omissions':
        condition = (response_df_session['n_after_omission'] == 0)
    else:
        raise ValueError('event_type not recognized')
    conditioned_df_session = response_df_session[condition]

    trace_df = get_trace_df_from_response_df_session(conditioned_df_session, lamf_group, inter_image_interval=inter_image_interval)
          
    return trace_df


def get_trace_df_from_response_df_session(response_df_session, lamf_group, inter_image_interval=0.75):
    
    trace_df = pd.DataFrame(columns=['cell_specimen_id', 'trace', 'oeid', 'target_region', 'depth_order', 'bisect_layer']).set_index('cell_specimen_id')
    oeids = np.sort(response_df_session.oeid.unique())
    for oeid in oeids:
        response_df_oeid = response_df_session.query('oeid==@oeid')
        csids = response_df_oeid.cell_specimen_id.unique()
        
        first_timestamps = response_df_oeid.trace_timestamps.values[0]
        assert response_df_oeid.trace_timestamps.apply(lambda x: np.all(x==first_timestamps)).all()
        start_index = np.where(first_timestamps>=0)[0][0]
        end_index = np.where(first_timestamps<=inter_image_interval)[0][-1]

        trace_all = []
        for csid in csids:
            csid_df = response_df_oeid[response_df_oeid.cell_specimen_id==csid]
            csid_trace = np.concatenate(csid_df.trace.apply(lambda x: x[start_index:end_index]).values)
            trace_all.append(csid_trace)
        
        temp_df = pd.DataFrame(columns=['cell_specimen_id', 'trace', 'oeid', 'target_region', 'depth_order', 'bisect_layer'])
        
        temp_df['cell_specimen_id'] = csids
        temp_df['trace'] = trace_all
        temp_df['oeid'] = oeid
        temp_df['target_region'] = lamf_group.expt_table.loc[oeid].targeted_structure
        temp_df['depth_order'] = lamf_group.expt_table.loc[oeid].depth_order
        temp_df['bisect_layer'] = lamf_group.expt_table.loc[oeid].bisect_layer
        temp_df.set_index('cell_specimen_id', inplace=True)
        temp_df.sort_index(inplace=True)
        trace_df = pd.concat([trace_df, temp_df])
    return trace_df


# A function to plot running speed during gray periods
def plot_running_speed_gray_periods(session_name, lamf_group, gray_period=5 * 60):
    # Plot running speed during gray periods
    oeids = lamf_group.expt_table.query('session_name==@session_name').index.values
    exp = lamf_group.experiments[oeids[0]]
    
    timepoints = exp.inner.running_speed.timestamps.values
    speed = exp.inner.running_speed.speed.values
    pregray_inds = np.where(timepoints < gray_period)[0]

    stim_df = exp.stimulus_presentations
    last_stim_end_time = stim_df.stop_time.values[-1]
    post_gray_end_time = last_stim_end_time + gray_period
    # get timepoints after last stimulus presentation and before gray period ends
    postgray_inds = np.where((timepoints > last_stim_end_time) & (timepoints < post_gray_end_time))[0]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timepoints[pregray_inds] - timepoints[pregray_inds[0]], speed[pregray_inds], label='pre-task gray')
    ax.plot(timepoints[postgray_inds] - timepoints[postgray_inds[0]], speed[postgray_inds], label='post-task gray')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('speed (cm/s)')
    ax.set_title(f'{session_name}\nrunning speed during gray periods')
    ax.legend()
    return fig


# A function to get interpolated time series
def get_interpolated_time_series(timestamps, values, new_timepoints):
    # remove timestamps with nan values
    nan_inds = np.where(np.isnan(values))[0]
    timestamps = np.delete(timestamps, nan_inds)
    values = np.delete(values, nan_inds)
    # Match timestamp range to that of the new_timepoints
    if new_timepoints[0] < timestamps[0]:
        timestamps = np.insert(timestamps, 0, new_timepoints[0])
        values = np.insert(values, 0, values[0])
    if new_timepoints[-1] > timestamps[-1]:
        timestamps = np.append(timestamps, new_timepoints[-1])
        values = np.append(values, values[-1])
    f = scipy.interpolate.interp1d(timestamps, values)
    new_values = f(new_timepoints)
    return new_values


def plot_task_raster_with_behav_sort_by_corr(
        lamf_group, session_name, remove_auto_rewarded=True,
        lick_template_rate=100, lick_rate_window=1, sub_title_fontsize=10,
        num_cell_threshold=20, vmin=-3, vmax=5):
    fig, ax = plt.subplots(figsize=(12, 8))
    trace_df_task = get_trace_df_task(lamf_group, session_name, remove_auto_rewarded=remove_auto_rewarded)
    if len(trace_df_task) < num_cell_threshold:
        print('Too few cells to plot')
        return fig
    else:
        *_, sort_ind_task, remove_ind = get_correlation_matrices(trace_df_task)
        if remove_ind is not None:
            trace_df_task = trace_df_task.reset_index()
            trace_df_task = trace_df_task.drop(remove_ind)
            trace_df_task = trace_df_task.set_index('cell_specimen_id')
        task_traces_all = trace_df_task.trace.values
        task_traces_all_zscore = np.array([(trace - np.nanmean(trace)) / np.nanmean(trace) for trace in task_traces_all])
        task_trace_all_mean = np.nanmean(task_traces_all_zscore, axis=0)
        sub_title_fontsize=10
        
        # Plot z-scored traces
        ax.imshow(task_traces_all_zscore[sort_ind_task, :], aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax.set_ylabel('Cell # (sorted by global correlation)')
        ax.set_xticks([])
        divider = make_axes_locatable(ax)

        # Add mean z-score
        zax = divider.append_axes('bottom', size='20%', pad=0.2)
        zax.plot(task_trace_all_mean, color='C2')
        zax.set_xlim(0, len(task_trace_all_mean))
        zax.set_xticks([])
        zax.set_title('z-scored dF/F', loc='left', fontsize=sub_title_fontsize, color='C2')

        # Add running speed by interpolating to the timepoints
        oeid = lamf_group.expt_table.query('session_name == @session_name').index.values[0]  # First one can be the representative one
        exp = lamf_group.experiments[oeid]
        timepoints = trace_df_task.iloc[0].timepoints  # First one can be the representative one

        running_timestamps = exp.running_speed.timestamps.values
        running_speed = exp.running_speed.speed.values
        running_interp = get_interpolated_time_series(running_timestamps, running_speed, timepoints)
        rax = divider.append_axes('bottom', size='20%', pad=0.25, sharex=zax)
        rax.plot(running_interp, color='C3')
        rax.set_xlim(0, len(task_trace_all_mean))
        rax.set_xticks([])
        rax.set_title('Running speed (cm/s)', loc='left', fontsize=sub_title_fontsize, color='C3')

        # Add pupil diameter by interpolating to the timepoints
        eye_timestamps = exp.eye_tracking.timestamps.values
        pupil_diameter = exp.eye_tracking.pupil_diameter.values
        pupil_interp = get_interpolated_time_series(eye_timestamps, pupil_diameter, timepoints)
        pax = divider.append_axes('bottom', size='20%', pad=0.25, sharex=zax)
        pax.plot(pupil_interp, color='black')
        pax.set_xlim(0, len(task_trace_all_mean))
        pax.set_xticks([])
        pax.set_title('Pupil diameter (AU)', loc='left', fontsize=sub_title_fontsize, color='black')

        # # Add licks to the nearest timepoints
        # licktimes = exp.licks.timestamps.values
        # stim_df = data_formatting.annotate_stimuli(exp) # First experiment represents the session stimulus presentations
        # # Only works on old allensdk version (or lamf_hacks branch of MJD's fork)
        # # TODO: when using updated version of allensdk (>2.13.6), change the code accordingly:
        # if remove_auto_rewarded:
        #     start_times, stop_times = get_non_auto_rewarded_start_end_times(stim_df)  # across all experiments
        # else:
        #     start_times = stim_df.start_time.values[0]
        #     stop_times = stim_df.stop_time.values[-1]
        # non_auto_rewarded_licktimes = np.concatenate([licktimes[(licktimes>=start) & (licktimes<=stop)] for start, stop in zip(start_times, stop_times)])
        # licks = np.zeros(len(timepoints))
        # for licktime in non_auto_rewarded_licktimes:
        #     # Find the nearest timepoint in timepoints to the reward timestamp
        #     nearest_ind = np.argmin(np.abs(timepoints - licktime))
        #     licks[nearest_ind] = 1
        # lax = divider.append_axes('bottom', size='10%', pad=0.25, sharex=zax)
        # lax.plot(licks, linewidth=1, color='pink')
        # lax.set_xlim(0, len(task_trace_all_mean))
        # lax.set_xticks([])
        # lax.set_title('Licks', loc='left', fontsize=sub_title_fontsize, color='pink')

        # Add lick rates by interpolating to the timepoints
        lick_time_template = np.arange(0, timepoints[-1] + 1 / lick_template_rate, 
                                       1 / lick_template_rate)  # timestamps are in seconds
        licks = np.zeros(len(lick_time_template))
        licktimes = exp.licks.timestamps.values
        licktimes = licktimes[licktimes <= timepoints[-1]]
        for licktime in licktimes:
            # Find the nearest timepoint in timepoints to the lick timestamp
            nearest_ind = np.argmin(np.abs(lick_time_template - licktime))
            licks[nearest_ind] += 1
        # calculate rate from licks with a moving window
        lickrate = np.convolve(licks, np.ones((int(lick_template_rate * lick_rate_window),)) /
                               int(lick_template_rate * lick_rate_window), mode='same') *\
                               lick_template_rate / lick_rate_window  # in Hz  # noqa E127
        # interpolate to the timepoints
        f = scipy.interpolate.interp1d(lick_time_template, lickrate, kind='linear')
        lickrate_interp = f(timepoints)
        lax = divider.append_axes('bottom', size='20%', pad=0.25, sharex=zax)
        lax.plot(lickrate_interp, linewidth=1, color='C6')
        lax.set_xlim(0, len(task_trace_all_mean))
        lax.set_xticks([])
        lax.set_title('Lick rate (Hz)', loc='left', fontsize=sub_title_fontsize, color='C6')

        # Add rewards to the nearest timepoints
        if remove_auto_rewarded:
            rewards_df = exp.rewards.query('autorewarded==False')
        else:
            rewards_df = exp.rewards
        reward_timestamps = rewards_df.timestamps
        reward_timestamps = reward_timestamps[reward_timestamps <= timepoints[-1]]
        rewards = np.zeros(len(timepoints))
        for timestamp in reward_timestamps:
            # Find the nearest timepoint in timepoints to the reward timestamp
            nearest_ind = np.argmin(np.abs(timepoints - timestamp))
            rewards[nearest_ind] = +1
        rax = divider.append_axes('bottom', size='10%', pad=0.25, sharex=zax)
        rax.plot(rewards, linewidth=1, color='C9')
        rax.set_xlim(0, len(task_trace_all_mean))
        rax.set_xticks([])
        rax.set_xlabel('Frame #')
        rax.set_title('Rewards', loc='left', fontsize=sub_title_fontsize, color='C9')

        cax = divider.append_axes('right', size='2%', pad=0.05)
        cbar = fig.colorbar(ax.images[0], cax=cax)
        cbar.ax.set_ylabel('z-scored dF/F')
    fig.suptitle(session_name)
    fig.tight_layout()

    return fig


def plot_notask_raster_with_behav_sort_by_corr(
        lamf_group, session_name, remove_auto_rewarded=True,
        lick_template_rate=100, lick_rate_window=1,
        sub_title_fontsize=10, num_cell_threshold=20,
        vmin_gray=-20, vmax_gray=20, vmin_fingerprint=-3, vmax_fingerprint=5):

    trace_df_no_task_list = get_trace_df_no_task(lamf_group, session_name)
    epochs = ['gray_pre', 'gray_post', 'fingerprint']
    sub_title_fontsize=10
    fig, ax = plt.subplots(1, len(epochs), figsize=(12, 8))
    run_num = 0
    for ax_ind, epoch in enumerate(epochs):
        epoch_ind = epochs.index(epoch)
        trace_df = trace_df_no_task_list[epoch_ind]
        if len(trace_df) < num_cell_threshold:
            continue
        else:
            run_num += 1
            *_, sort_ind, remove_ind = get_correlation_matrices(trace_df)
            if remove_ind is not None:
                trace_df = trace_df.reset_index()
                trace_df = trace_df.drop(remove_ind)
                trace_df = trace_df.set_index('cell_specimen_id')
            traces_all = trace_df.trace.values
            traces_all_zscore = np.array([(trace - np.nanmean(trace)) / np.nanmean(trace) for trace in traces_all])
            trace_all_mean = np.nanmean(traces_all_zscore, axis=0)
            
            # Plot z-scored traces
            if 'gray' in epoch:
                vmin = vmin_gray
                vmax = vmax_gray
            else:
                vmin = vmin_fingerprint
                vmax = vmax_fingerprint
            ax[ax_ind].imshow(traces_all_zscore[sort_ind, :], aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
            ax[ax_ind].set_title(epoch)
            ax[ax_ind].set_xticks([])
            if ax_ind == 0:
                ax[ax_ind].set_ylabel('Cell # (sorted by global correlation)')
            divider = make_axes_locatable(ax[ax_ind])

            # Add mean z-score
            zax = divider.append_axes('bottom', size='20%', pad=0.2)
            zax.plot(trace_all_mean, color='C2')
            zax.set_xlim(0, len(trace_all_mean))
            zax.set_xticks([])
            zax.set_title('z-scored dF/F', loc='left', fontsize=sub_title_fontsize, color='C2')

            # Add running speed by interpolating to the timepoints
            oeid = lamf_group.expt_table.query('session_name == @session_name').index.values[0]  # First one can be the representative one
            exp = lamf_group.experiments[oeid]
            timepoints = trace_df.iloc[0].timepoints  # First one can be the representative one

            running_timestamps = exp.running_speed.timestamps.values
            running_speed = exp.running_speed.speed.values
            running_interp = get_interpolated_time_series(running_timestamps, running_speed, timepoints)
            rax = divider.append_axes('bottom', size='20%', pad=0.25, sharex=zax)
            rax.plot(running_interp, color='C3')
            rax.set_xlim(0, len(trace_all_mean))
            rax.set_xticks([])
            rax.set_title('Running speed (cm/s)', loc='left', fontsize=sub_title_fontsize, color='C3')

            # Add pupil diameter by interpolating to the timepoints
            eye_timestamps = exp.eye_tracking.timestamps.values
            pupil_diameter = exp.eye_tracking.pupil_diameter.values
            pupil_interp = get_interpolated_time_series(eye_timestamps, pupil_diameter, timepoints)
            pax = divider.append_axes('bottom', size='20%', pad=0.25, sharex=zax)
            pax.plot(pupil_interp, color='black')
            pax.set_xlim(0, len(trace_all_mean))
            pax.set_xticks([])
            pax.set_title('Pupil diameter (AU)', loc='left', fontsize=sub_title_fontsize, color='black')

            # # Add licks to the nearest timepoints
            # licktimes = exp.licks.timestamps.values
            # stim_df = data_formatting.annotate_stimuli(exp) # First experiment represents the session stimulus presentations
            # # Only works on old allensdk version (or lamf_hacks branch of MJD's fork)
            # # TODO: when using updated version of allensdk (>2.13.6), change the code accordingly:
            # if remove_auto_rewarded:
            #     start_times, stop_times = get_non_auto_rewarded_start_end_times(stim_df)  # across all experiments
            # else:
            #     start_times = stim_df.start_time.values[0]
            #     stop_times = stim_df.stop_time.values[-1]
            # non_auto_rewarded_licktimes = np.concatenate([licktimes[(licktimes>=start) & (licktimes<=stop)] for start, stop in zip(start_times, stop_times)])
            # licks = np.zeros(len(timepoints))
            # for licktime in non_auto_rewarded_licktimes:
            #     # Find the nearest timepoint in timepoints to the reward timestamp
            #     nearest_ind = np.argmin(np.abs(timepoints - licktime))
            #     licks[nearest_ind] = 1
            # lax = divider.append_axes('bottom', size='10%', pad=0.25, sharex=zax)
            # lax.plot(licks, linewidth=1, color='pink')
            # lax.set_xlim(0, len(trace_all_mean))
            # lax.set_xticks([])
            # lax.set_title('Licks', loc='left', fontsize=sub_title_fontsize, color='pink')

            # Add lick rates by interpolating to the timepoints
            lick_time_template = np.arange(0, timepoints[-1] + 1 / lick_template_rate, 
                                           1 / lick_template_rate)  # timestamps are in seconds
            licks = np.zeros(len(lick_time_template))
            licktimes = exp.licks.timestamps.values
            licktimes = licktimes[licktimes <= timepoints[-1]]
            for licktime in licktimes:
                # Find the nearest timepoint in timepoints to the lick timestamp
                nearest_ind = np.argmin(np.abs(lick_time_template - licktime))
                licks[nearest_ind] += 1
            # calculate rate from licks with a moving window
            lickrate = np.convolve(licks, np.ones((int(lick_template_rate * lick_rate_window),)) /
                                   int(lick_template_rate * lick_rate_window), mode='same') * \
                                   lick_template_rate / lick_rate_window  # in Hz  # noqa E127
            # interpolate to the timepoints
            f = scipy.interpolate.interp1d(lick_time_template, lickrate, kind='linear')
            lickrate_interp = f(timepoints)
            lax = divider.append_axes('bottom', size='20%', pad=0.25, sharex=zax)
            lax.plot(lickrate_interp, linewidth=1, color='C6')
            lax.set_xlim(0, len(trace_all_mean))
            lax.set_xticks([])
            lax.set_title('Lick rate (Hz)', loc='left', fontsize=sub_title_fontsize, color='C6')

            # Add rewards to the nearest timepoints
            if remove_auto_rewarded:
                rewards_df = exp.rewards.query('autorewarded==False')
            else:
                rewards_df = exp.rewards
            reward_timestamps = rewards_df.timestamps
            reward_timestamps = reward_timestamps[reward_timestamps <= timepoints[-1]]
            rewards = np.zeros(len(timepoints))
            for timestamp in reward_timestamps:
                # Find the nearest timepoint in timepoints to the reward timestamp
                nearest_ind = np.argmin(np.abs(timepoints - timestamp))
                rewards[nearest_ind] += 1
            rax = divider.append_axes('bottom', size='10%', pad=0.25, sharex=zax)
            rax.plot(rewards, linewidth=1, color='C9')
            rax.set_xlim(0, len(trace_all_mean))
            rax.set_xticks([])
            rax.set_xlabel('Frame #')
            rax.set_title('Rewards', loc='left', fontsize=sub_title_fontsize, color='C9')

            cax = divider.append_axes('right', size='2%', pad=0.05)
            cbar = fig.colorbar(ax[ax_ind].images[0], cax=cax)
            cbar.ax.set_ylabel('z-scored dF/F')

    fig.suptitle(session_name)
    fig.tight_layout()
    return fig
