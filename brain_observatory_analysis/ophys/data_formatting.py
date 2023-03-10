import numpy as np
import pandas as pd
from mindscope_utilities.visual_behavior_ophys import data_formatting
# TODO: make it independent of mindscope_utilities


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