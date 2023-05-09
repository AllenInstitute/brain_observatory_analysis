import numpy as np
import pandas as pd
from mindscope_utilities.visual_behavior_ophys import data_formatting
# TODO: make it independent of mindscope_utilities


# TODO: Make them work with events as well.
# Need to consider timepoint alignment between events.
# Functions to get trace_df and calculate correlations

def _default_column_names():
    """ Return default column names for trace_df
    Returns
    -------
    column_names : list
        list of column names
    """
    column_names = ['cell_specimen_id', 'cell_roi_id', 'trace', 'timepoints', 'oeid', 'targeted_structure', 'depth_order', 'bisect_layer']
    return column_names


def _default_column_base_names():
    """ Return default column base names for trace_df
    Returns
    -------
    column_base_names : list
        list of column base names
    """
    column_base_names = ['cell_specimen_id', 'cell_roi_id', 'trace', 'timepoints', 'oeid']
    return column_base_names


def _set_column_names(expt_group, column_names, column_base_names):
    """ Set column names for trace_df
    From the column_names, remove the ones that are not in expt_group.expt_table.columns.values
    Add the ones that are in column_base_names but not in column_names
    Also return the column names that are NOT in column_base_names as column_added_names

    Parameters
    ----------
    expt_group : ExperimentGroup
        ExperimentGroup object
    column_names : list
        list of column names
    column_base_names : list
        list of column base names

    Returns
    -------
    column_names : list
        list of column names
    column_added_names : list
        list of column names added
    """
    expt_column_names = expt_group.expt_table.columns.values
    for cn in column_names:
        if cn not in expt_column_names:
            column_names.remove(cn)
    for cbn in column_base_names:
        if cbn not in column_names:
            column_names = [cbn] + column_names
    column_added_names = [cn for cn in column_names if cn not in column_base_names]
    return column_names, column_added_names


def get_all_trace_df(expt_group, session_name, trace_type='dff', column_names=None):
    """Get a dataframe of all traces from a session (all ophys trace, either dff or events)

    Parameters
    ----------
    expt_group : ExperimentGroup
        ExperimentGroup object
    session_name : str
        session name
    trace_type : str, optional
        'dff' or 'events', by default 'dff'
    column_names : list, optional
        column names of the dataframe, by default None (will use _default_column_names())
    
    Returns
    -------
    trace_df : pd.DataFrame
        dataframe of all traces from a session
    
    """
    oeids = np.sort(expt_group.expt_table[expt_group.expt_table.session_name.str.lower() == session_name.lower()].index.values)
    if column_names is None:
        column_names = _default_column_names()
    column_base_names = _default_column_base_names()
    column_names, column_added_names = _set_column_names(expt_group, column_names, column_base_names)
    # load all the traces from this session
    trace_df = pd.DataFrame(columns=column_names).set_index('cell_specimen_id')
    for oeid in oeids:
        temp_df = pd.DataFrame(columns=column_names)
        if trace_type == 'dff':
            exp_trace_df = expt_group.experiments[oeid].dff_traces
            trace = exp_trace_df.dff.values
        elif trace_type == 'events':
            exp_trace_df = expt_group.experiments[oeid].events
            trace = exp_trace_df.events.values
        else:
            raise ValueError('trace_type must be either dff or events')
        csid = exp_trace_df.index.values
        crid = exp_trace_df.cell_roi_id.values
        temp_df['cell_specimen_id'] = csid
        temp_df['cell_roi_id'] = crid
        temp_df['trace'] = trace
        temp_df['timepoints'] = [expt_group.experiments[oeid].ophys_timestamps] * len(csid)
        temp_df['oeid'] = oeid
        for cn in column_added_names:
            temp_df[cn] = expt_group.expt_table.loc[oeid, cn]
        temp_df.set_index('cell_specimen_id', inplace=True)
        temp_df.sort_index(inplace=True)
        trace_df = pd.concat([trace_df, temp_df])
    return trace_df


def get_notask_trace_df(expt_group, session_name, trace_type='dff', column_names=None):
    """Get a dataframe of traces from a session (all ophys trace, either dff or events) with no task
    Assume 5 min gray screen before and after task
    and 5 min fingerprint (movie-watching; 30 sec 10 iterations) imaging at the end
    Some sessions don't have them.

    Parameters
    ----------
    expt_group : ExperimentGroup
        ExperimentGroup object
    session_name : str
        session name
    trace_type : str, optional
        'dff' or 'events', by default 'dff'
    column_names : list, optional
        column names of the dataframe, by default None
    
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
    oeids = np.sort(expt_group.expt_table[expt_group.expt_table.session_name.str.lower() == session_name.lower()].index.values)
    stim_df = data_formatting.annotate_stimuli(expt_group.experiments[oeids[0]])  # First experiment represents the session stimulus presentations
    
    # Match the # of indices from each experiment
    start_inds = []
    end_inds = []
    post_gray_end_inds = []
    for oeid in oeids:
        timestamps = expt_group.experiments[oeid].ophys_timestamps
        
        first_stim_start_time = stim_df.start_time.values[0]
        if first_stim_start_time > gray_period - 30:  # 30 is an arbitrary buffer
            # This buffer is to select the cases where there really was a gray screen before the task 
            # (same as in the other buffers below)
            # Updated allensdk has the table for these epochs, but currently we cannot use it due to loading time (2023/03/11; instead, we are using Matt's lamf_hacks branch)
            # TODO: use the table from allensdk when it is available
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

    # Set the column names
    if column_names is None:
        column_names = _default_column_names()
    column_base_names = _default_column_base_names()
    column_names, column_added_names = _set_column_names(expt_group, column_names, column_base_names)
    
    # Initialize
    trace_df_gray_pre_task = pd.DataFrame(columns=column_names).set_index('cell_specimen_id')
    trace_df_gray_post_task = pd.DataFrame(columns=column_names).set_index('cell_specimen_id')
    trace_df_fingerprint = pd.DataFrame(columns=column_names).set_index('cell_specimen_id')
    for oeid in oeids:
        # Initialize for each experiment
        temp_df = pd.DataFrame(columns=column_names).set_index('cell_specimen_id')
        if trace_type == 'dff':
            trace_df = expt_group.experiments[oeid].dff_traces
        elif trace_type == 'events':
            trace_df = expt_group.experiments[oeid].events
        else:
            raise ValueError('trace_type must be dff or events')
        csid = trace_df.index.values
        crid = trace_df.cell_roi_id.values
        
        timestamps = expt_group.experiments[oeid].ophys_timestamps
        # cell_specimen_id, oeid, targeted_structure, depth_order, bisect_layer are the same
        # across epochs in no-task window
        # Only switch trace and timepoints for each epoch
        
        temp_df['cell_specimen_id'] = csid
        temp_df['cell_roi_id'] = crid
        temp_df['oeid'] = oeid
        for cn in column_added_names:
            temp_df[cn] = expt_group.expt_table.loc[oeid][cn]
        temp_df.set_index('cell_specimen_id', inplace=True)
        if run_pre_gray:
            if trace_type == 'dff':
                temp_df['trace'] = trace_df.dff.apply(lambda x: x[:min_start_ind]).values
            else:  # Other types already dealt with from above
                temp_df['trace'] = trace_df.events.apply(lambda x: x[:min_start_ind]).values
            temp_df['timepoints'] = [timestamps[:min_start_ind]] * len(csid)
            sorted_temp_df = temp_df.sort_index(inplace=False)
            trace_df_gray_pre_task = pd.concat([trace_df_gray_pre_task, sorted_temp_df])
        if run_post_gray:
            if trace_type == 'dff':
                temp_df['trace'] = trace_df.dff.apply(lambda x: x[max_end_ind:min_post_gray_end_ind]).values
            else:
                temp_df['trace'] = trace_df.events.apply(lambda x: x[max_end_ind:min_post_gray_end_ind]).values
            temp_df['timepoints'] = [timestamps[max_end_ind:min_post_gray_end_ind]] * len(csid)
            sorted_temp_df = temp_df.sort_index(inplace=False)
            trace_df_gray_post_task = pd.concat([trace_df_gray_post_task, sorted_temp_df])
            if run_fingerprint:
                if trace_type == 'dff':
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
    if len(auto_rewarded_inds) > 0:
        diff_ar_inds = np.diff(auto_rewarded_inds)
        skip_ar_inds = np.where(diff_ar_inds > 1)[0]
        start_times = []  # inclusive
        stop_times = []  # inclusive
        for skip_ar_ind in skip_ar_inds:
            start_times.append(stim_df.iloc[auto_rewarded_inds[skip_ar_ind] + 1].start_time)  # removing those before the first auto-rewarded stim
            stop_times.append(stim_df.iloc[auto_rewarded_inds[skip_ar_ind + 1]].stop_time)
        if auto_rewarded_inds[-1] != len(stim_df) - 1:  # When the last stimulus was not auto-rewarded, add the segments after the last auto-rewarded stim till the last stim
            start_times.append(stim_df.iloc[auto_rewarded_inds[-1] + 1].start_time)
            stop_times.append(stim_df.iloc[-1].stop_time)
    else:
        start_times = [stim_df.iloc[0].start_time]
        stop_times = [stim_df.iloc[-1].stop_time]
    return start_times, stop_times


def get_start_end_inds(start_times, stop_times, expt_group, oeids):
    """ Get start and end indices matching to the start and stop times
    from expt_group ophys_experiment_ids

    Parameters
    ----------
    start_times: list
        A list of start times
    stop_times: list
        A list of stop times
    expt_group : ExperimentGroup
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
            timestamps = expt_group.experiments[oeid].ophys_timestamps
            start_inds.append(np.where(timestamps >= start_time)[0][0])
            end_inds.append(np.where(timestamps <= end_time)[0][-1])
        max_start_inds.append(np.max(start_inds))
        min_end_inds.append(np.min(end_inds))
    return max_start_inds, min_end_inds


def get_task_trace_df(expt_group, session_name, remove_auto_rewarded=True, column_names=None):
    """ Get trace_df for a given session

    Parameters
    ----------
    expt_group : ExperimentGroup
        ExperimentGroup object
    session_name: str
        A string of session name
    remove_auto_rewarded: bool, optional
        Whether to remove auto-rewarded stimuli, default True
    column_names: list, optional
        A list of column names, default None

    Returns
    -------
    trace_df: pandas.DataFrame
        A dataframe containing trace information during the whole task
    """
    
    oeids = np.sort(expt_group.expt_table[expt_group.expt_table.session_name.str.lower() == session_name.lower()].index.values)
    stim_df = data_formatting.annotate_stimuli(expt_group.experiments[oeids[0]])  # First experiment represents the session stimulus presentations
    # Only works on old allensdk version (or lamf_hacks branch of MJD's fork)
    # TODO: when using updated version of allensdk (>2.13.6), change the code accordingly:
    if remove_auto_rewarded:
        start_times, stop_times = get_non_auto_rewarded_start_end_times(stim_df)  # across all experiments
    else:
        start_times = stim_df.start_time.values[0]
        stop_times = stim_df.stop_time.values[-1]
    # To match the frame indices across experiments
    max_start_inds, min_end_inds = get_start_end_inds(start_times, stop_times, expt_group, oeids)

    # Set the column names
    if column_names is None:
        column_names = _default_column_names()
    column_base_names = _default_column_base_names()
    column_names, column_added_names = _set_column_names(expt_group, column_names, column_base_names)
  
    # Initialize the dataframe
    trace_df = pd.DataFrame(columns=column_names).set_index('cell_specimen_id')
    for oeid in oeids:
        temp_df = pd.DataFrame(columns=column_names)

        dff_df = expt_group.experiments[oeid].dff_traces
        csid = dff_df.index.values
        crid = dff_df.cell_roi_id.values

        test_dff = dff_df.iloc[0].dff
        timestamps = expt_group.experiments[oeid].ophys_timestamps
        assert len(test_dff) == len(timestamps)

        try:
            test_dff_crop = np.concatenate([test_dff[msi: mei] for msi, mei in zip(max_start_inds, min_end_inds)])
            timepoints = np.concatenate([timestamps[msi: mei] for msi, mei in zip(max_start_inds, min_end_inds)])
            assert len(timepoints) == len(test_dff_crop)

            temp_df['cell_specimen_id'] = csid
            temp_df['cell_roi_id'] = crid
            temp_df['trace'] = dff_df.dff.apply(lambda x: np.concatenate([x[msi: mei] for msi, mei in zip(max_start_inds, min_end_inds)])).values
            temp_df['timepoints'] = [timepoints] * len(csid)
            temp_df['oeid'] = oeid
            for cn in column_added_names:
                temp_df[cn] = expt_group.expt_table.loc[oeid][cn]
            temp_df.set_index('cell_specimen_id', inplace=True)
            # sort temp_df by cell_specimen_id
            temp_df.sort_index(inplace=True)
            trace_df = pd.concat([trace_df, temp_df])
        except:  # noqa: E722
            continue
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
    if exp.cell_specimen_table.index.unique()[0] is not None:
        response_df = data_formatting.get_stimulus_response_df(exp, data_type=data_type, event_type=event_type, image_order=image_order,
                                                               time_window=[0, inter_image_interval], output_sampling_rate=output_sampling_rate)
        stim_df = get_all_annotated_stimulus_presentations(exp)
        response_df = response_df.merge(stim_df, how='left', on='stimulus_presentations_id', validate='m:1')
        response_df['oeid'] = exp.ophys_experiment_id
    else:
        response_df = None

    return response_df


def get_trace_df_event(expt_group, session_name, event_type, data_type='dff', image_order=3,
                       inter_image_interval=0.75, output_sampling_rate=20, remove_auto_rewarded=True, column_names=None):
    """ Get trace dataframe for a given session and event type
    Parameters
    ----------
    expt_group: ExperimentGroup
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
    column_names: list, optional
        Column names, default None
    
    Returns
    -------
    trace_df: pandas.DataFrame
        A dataframe containing traces for a given session and event type
    """

    oeids = np.sort(expt_group.expt_table[expt_group.expt_table.session_name.str.lower() == session_name.lower()].index.values)
    # Set the column names
    if column_names is None:
        column_names = _default_column_names()
    column_base_names = _default_column_base_names()
    column_names, column_added_names = _set_column_names(expt_group, column_names, column_base_names)
  
    # load all the traces from this session
    trace_df = pd.DataFrame(columns=column_names).set_index('cell_specimen_id')
    for oeid in oeids:
        csids = expt_group.experiments[oeid].cell_specimen_table.index.values
        if event_type == 'images>n-changes':
            response_df = get_event_annotated_response_df(expt_group.experiments[oeid], data_type=data_type, event_type=event_type, image_order=image_order,
                                                          inter_image_interval=inter_image_interval, output_sampling_rate=output_sampling_rate)
        else:
            response_df = get_event_annotated_response_df(expt_group.experiments[oeid], data_type=data_type, event_type=event_type,
                                                          inter_image_interval=inter_image_interval, output_sampling_rate=output_sampling_rate)
        
        if len(response_df) > 0:
            if remove_auto_rewarded:
                response_df = response_df[response_df.auto_rewarded == False]  # noqa: E712

            first_timestamps = response_df.trace_timestamps.values[0]
            assert response_df.trace_timestamps.apply(lambda x: np.all(x == first_timestamps)).all()
            start_index = np.where(first_timestamps >= 0)[0][0]  
            end_index = np.where(first_timestamps <= inter_image_interval)[0][-1]  # The last index is going to be not included

            csids = response_df.cell_specimen_id.unique()
            crids = []
            trace_all = []
            for csid in csids:
                csid_df = response_df[response_df.cell_specimen_id == csid]
                crids.append(csid_df.cell_roi_id.values[0])
                csid_trace = np.concatenate(csid_df.trace.apply(lambda x: x[start_index:end_index]).values)
                trace_all.append(csid_trace)
            assert np.diff([len(trace) for trace in trace_all]).any() == False  # noqa: E712
            timepoints = np.concatenate([st + tts[1:] for st, tts in zip(csid_df.start_time.values, csid_df.trace_timestamps.values)])
            assert len(timepoints) == len(trace_all[0])
            
            temp_df = pd.DataFrame(columns=column_names)
            temp_df['cell_specimen_id'] = csids
            temp_df['cell_roi_id'] = crids
            temp_df['trace'] = trace_all
            temp_df['timepoints'] = [timepoints] * len(csids)
            temp_df['oeid'] = oeid
            for cn in column_added_names:
                temp_df[cn] = expt_group.expt_table.loc[oeid, cn]
            temp_df.set_index('cell_specimen_id', inplace=True)
            temp_df.sort_index(inplace=True)
            trace_df = pd.concat([trace_df, temp_df])
    return trace_df


def get_all_annotated_session_response_df(expt_group, session_name, inter_image_interval=0.75, output_sampling_rate=20):
    """Get all response_df for a session from all experiments in a expt_group

    Parameters
    ----------
    expt_group: Experiment group object
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
    
    oeids = np.sort(expt_group.expt_table[expt_group.expt_table.session_name.str.lower() == session_name.lower()].index.values)
    response_df = pd.DataFrame()
    for oeid in oeids:
        exp = expt_group.experiments[oeid]
        temp_df = get_event_annotated_response_df(exp, event_type='all', inter_image_interval=inter_image_interval, output_sampling_rate=output_sampling_rate)
        if temp_df is not None:
            response_df = response_df.append(temp_df)
    return response_df


def get_trace_df_event_from_all_response_df(response_df_session, expt_group, event_type, image_order=3, inter_image_interval=0.75):
    """Get trace DataFrame for a particular event type from all experiments in a session

    Parameters
    ----------
    response_df_session: pandas DataFrame
        DataFrame with stimulus presentations for all experiments in a session
    expt_group: Experiment group object
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

    if event_type == 'images-n-omissions':
        condition = (response_df_session['n_after_omission']==image_order) & \
                    (response_df_session['n_after_change'] > response_df_session['n_after_omission'])  # noqa E501
    elif event_type == 'images-n-changes':
        condition = (response_df_session['n_after_change']==image_order) & \
                    ((response_df_session['n_after_omission'] > response_df_session['n_after_change']) | # noqa E501  
                        (response_df_session['n_after_omission'] == -1))  # for trials without omission
    elif event_type == 'images>n-omissions':
        condition = (response_df_session['n_after_omission'] > image_order) & \
                    (response_df_session['n_after_change'] > response_df_session['n_after_omission'])
    elif event_type == 'images>n-changes':
        condition = (response_df_session['n_after_change'] > image_order) & \
                    ((response_df_session['n_after_omission'] > image_order) |  # noqa E501
                        (response_df_session['n_after_omission'] == -1))  # for trials without omission
    elif event_type == 'images-n-before-changes':
        condition = (response_df_session['n_before_change'] == image_order) & \
                    (response_df_session['n_after_omission'] == -1)  # Get trials without omission only
    elif event_type == 'changes':
        condition = (response_df_session['n_after_change'] == 0)
    elif event_type == 'omissions':
        condition = (response_df_session['n_after_omission'] == 0)
    else:
        raise ValueError('event_type not recognized')
    conditioned_df_session = response_df_session[condition]

    trace_df = get_trace_df_from_response_df_session(conditioned_df_session, expt_group, inter_image_interval=inter_image_interval)
          
    return trace_df


def get_trace_df_from_response_df_session(response_df_session, expt_group, inter_image_interval=0.75, column_names=None):
    """Get trace DataFrame from all experiments in a session

    Parameters
    ----------
    response_df_session: pandas DataFrame
        DataFrame with stimulus presentations for all experiments in a session
    expt_group: Experiment group object
        Experiment group object for Learning and mFISH project
    inter_image_interval: float, optional
        Inter image interval in seconds, default 0.75
    column_names: list, optional
        List of column names to add to the trace DataFrame, default None
    
    Returns
    -------
    trace_df: pandas DataFrame
        DataFrame with traces for all experiments in a session
    """
    # Set the column names
    if column_names is None:
        column_names = _default_column_names()
    column_base_names = _default_column_base_names()
    column_names, column_added_names = _set_column_names(expt_group, column_names, column_base_names)
  
    # Initialize the trace DataFrame
    trace_df = pd.DataFrame(columns=column_names).set_index('cell_specimen_id')
    oeids = np.sort(response_df_session.oeid.unique())
    for oeid in oeids:
        response_df_oeid = response_df_session.query('oeid==@oeid')
        csids = response_df_oeid.cell_specimen_id.unique()
        
        first_timestamps = response_df_oeid.trace_timestamps.values[0]
        assert response_df_oeid.trace_timestamps.apply(lambda x: np.all(x == first_timestamps)).all()
        start_index = np.where(first_timestamps >= 0)[0][0]
        end_index = np.where(first_timestamps <= inter_image_interval)[0][-1]

        crids = []
        trace_all = []
        for csid in csids:
            csid_df = response_df_oeid[response_df_oeid.cell_specimen_id == csid]
            crids.append(csid_df.cell_roi_id.values[0])
            csid_trace = np.concatenate(csid_df.trace.apply(lambda x: x[start_index:end_index]).values)
            trace_all.append(csid_trace)
        
        temp_df = pd.DataFrame(columns=column_names)
        
        temp_df['cell_specimen_id'] = csids
        temp_df['cell_roi_id'] = crids
        temp_df['trace'] = trace_all
        temp_df['oeid'] = oeid
        for cn in column_added_names:
            temp_df[cn] = expt_group.expt_table.loc[oeid][cn]
        temp_df.set_index('cell_specimen_id', inplace=True)
        temp_df.sort_index(inplace=True)
        trace_df = pd.concat([trace_df, temp_df])
    return trace_df


def get_concatenated_mean_response_df_from_response_df(response_df, csid_list=None):
    """Get concatenated mean response DataFrame from response DataFrame

    Parameters
    ----------
    response_df: pandas DataFrame
        DataFrame with stimulus presentations for all experiments in a session
    csid_list: list, optional
        List of cell_specimen_ids to include in the concatenated mean response DataFrame, default None

    Returns
    -------
    mean_response_df: pandas DataFrame
        DataFrame with concatenated mean responses for all experiments in a session
    """

    non_auto_rewarded_response_df = response_df.query('auto_rewarded == False')
    change_response_df = non_auto_rewarded_response_df.query('is_change==True and auto_rewarded == False')
    change_response_df['trace_norm'] = change_response_df.apply(lambda x: x.trace - x.trace[0], axis=1)
    image_response_df = non_auto_rewarded_response_df.query('n_after_change > 3 and (n_after_omission > 3 or n_after_omission < 0)')
    image_response_df['trace_norm'] = image_response_df.apply(lambda x: x.trace - x.trace[0], axis=1)
    omission_response_df = non_auto_rewarded_response_df.query('omitted==True')
    omission_response_df['trace_norm'] = omission_response_df.apply(lambda x: x.trace - x.trace[0], axis=1)

    cell_specimen_ids = np.sort(response_df.cell_specimen_id.unique())
    temp_df = response_df[['cell_specimen_id', 'trace']].groupby('cell_specimen_id').mean()
    assert (temp_df.index.values - cell_specimen_ids).any() == False  # noqa: E712

    trace_timestamps = response_df.trace_timestamps.values[0]
    mean_response_change_df = pd.DataFrame({'cell_specimen_id': cell_specimen_ids})
    mean_response_change_df['trace'] = change_response_df[['cell_specimen_id', 'trace']].groupby('cell_specimen_id').apply(lambda x: np.vstack(x['trace'])).apply(lambda x: np.nanmean(x, axis=0)).values
    mean_response_change_df['trace_std'] = change_response_df[['cell_specimen_id', 'trace']].groupby('cell_specimen_id').apply(lambda x: np.vstack(x['trace'])).apply(lambda x: np.nanstd(x, axis=0)).values
    mean_response_change_df['trace_norm'] = change_response_df[['cell_specimen_id', 'trace_norm']].groupby('cell_specimen_id').apply(lambda x: np.vstack(x['trace_norm'])).apply(lambda x: np.nanmean(x, axis=0)).values
    mean_response_change_df['trace_std_norm'] = change_response_df[['cell_specimen_id', 'trace_norm']].groupby('cell_specimen_id').apply(lambda x: np.vstack(x['trace_norm'])).apply(lambda x: np.nanstd(x, axis=0)).values
    mean_response_change_df['time_stamps'] = [trace_timestamps] * len(mean_response_change_df)

    mean_response_image_df = pd.DataFrame({'cell_specimen_id': cell_specimen_ids})
    mean_response_image_df['trace'] = image_response_df[['cell_specimen_id', 'trace']].groupby('cell_specimen_id').apply(lambda x: np.vstack(x['trace'])).apply(lambda x: np.nanmean(x, axis=0)).values
    mean_response_image_df['trace_std'] = image_response_df[['cell_specimen_id', 'trace']].groupby('cell_specimen_id').apply(lambda x: np.vstack(x['trace'])).apply(lambda x: np.nanstd(x, axis=0)).values
    mean_response_image_df['trace_norm'] = image_response_df[['cell_specimen_id', 'trace_norm']].groupby('cell_specimen_id').apply(lambda x: np.vstack(x['trace_norm'])).apply(lambda x: np.nanmean(x, axis=0)).values
    mean_response_image_df['trace_std_norm'] = image_response_df[['cell_specimen_id', 'trace_norm']].groupby('cell_specimen_id').apply(lambda x: np.vstack(x['trace_norm'])).apply(lambda x: np.nanstd(x, axis=0)).values

    mean_response_omission_df = pd.DataFrame({'cell_specimen_id': cell_specimen_ids})
    mean_response_omission_df['trace'] = omission_response_df[['cell_specimen_id', 'trace']].groupby('cell_specimen_id').apply(lambda x: np.vstack(x['trace'])).apply(lambda x: np.nanmean(x, axis=0)).values
    mean_response_omission_df['trace_std'] = omission_response_df[['cell_specimen_id', 'trace']].groupby('cell_specimen_id').apply(lambda x: np.vstack(x['trace'])).apply(lambda x: np.nanstd(x, axis=0)).values
    mean_response_omission_df['trace_norm'] = omission_response_df[['cell_specimen_id', 'trace_norm']].groupby('cell_specimen_id').apply(lambda x: np.vstack(x['trace_norm'])).apply(lambda x: np.nanmean(x, axis=0)).values
    mean_response_omission_df['trace_std_norm'] = omission_response_df[['cell_specimen_id', 'trace_norm']].groupby('cell_specimen_id').apply(lambda x: np.vstack(x['trace_norm'])).apply(lambda x: np.nanstd(x, axis=0)).values

    assert (mean_response_change_df.index.values - mean_response_image_df.index.values).any() == False  # noqa: E712
    assert (mean_response_change_df.index.values - mean_response_omission_df.index.values).any() == False  # noqa: E712

    concat_mean_trace = np.hstack([np.vstack(mean_response_change_df.trace.values), np.vstack(mean_response_image_df.trace.values), np.vstack(mean_response_omission_df.trace.values)])
    concat_mean_trace_norm = np.hstack([np.vstack(mean_response_change_df.trace_norm.values), np.vstack(mean_response_image_df.trace_norm.values), np.vstack(mean_response_omission_df.trace_norm.values)])
    concat_std_trace = np.hstack([np.vstack(mean_response_change_df.trace_std.values), np.vstack(mean_response_image_df.trace_std.values), np.vstack(mean_response_omission_df.trace_std.values)])
    concat_std_trace_norm = np.hstack([np.vstack(mean_response_change_df.trace_std_norm.values), np.vstack(mean_response_image_df.trace_std_norm.values), np.vstack(mean_response_omission_df.trace_std_norm.values)])

    max_timestamp = np.max(trace_timestamps)
    timestamp_interval = np.diff(trace_timestamps)[0]
    concat_timeseries = np.hstack([trace_timestamps, trace_timestamps + max_timestamp + timestamp_interval, trace_timestamps + 2 * max_timestamp + 2 * timestamp_interval])
    assert len(np.where(np.diff(np.unique(np.diff(concat_timeseries))) > 1e-5)[0]) == 0

    concat_trace_df = pd.DataFrame({'cell_specimen_id': cell_specimen_ids,
                                    'trace': list(concat_mean_trace),
                                    'trace_std': list(concat_std_trace),
                                    'trace_norm': list(concat_mean_trace_norm),
                                    'trace_std_norm': list(concat_std_trace_norm),
                                    'time_stamps': [concat_timeseries] * len(cell_specimen_ids)})
    
    concat_trace_df = concat_trace_df.set_index('cell_specimen_id')
    if csid_list is not None:
        concat_trace_df = concat_trace_df.loc[csid_list]

    return concat_trace_df