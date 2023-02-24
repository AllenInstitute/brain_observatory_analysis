
import numpy as np
import pandas as pd
import logging
from functools import partial

logger = logging.getLogger(__name__)


def get_extended_stimulus_presentations_df(stimulus_presentations_df: pd.DataFrame, 
                                           licks: pd.DataFrame,
                                           rewards: pd.DataFrame, 
                                           running_speed: pd.DataFrame, 
                                           eye_tracking: bool = None, 
                                           behavior_session_id: int = None):
    """
    Takes SDK stimulus presentations table and adds a bunch of useful columns by incorporating data from other tables
    and reformatting existing column data
    Additional columns include epoch #s for 10 minute bins in the session, whether a flash was a pre or post change or omission,
    the mean running speed per flash, mean pupil area per flash, licks per flash, rewards per flash, lick rate, reward rate,
    time since last change, time since last omission, time since last lick

    Set eye_tracking to None by default so that things still run for behavior only sessions
    If behavior_session_id is provided, will load metrics from behavior model outputs file
    """
    spdf = stimulus_presentations_df

    if 'time' in licks.keys():
        licks = licks.rename(columns={'time': 'timestamps'})

    if 'orientation' in spdf.columns:
        spdf = spdf.drop(columns=['orientation', 'image_set',
                                  'phase', 'spatial_frequency'])

    spdf = add_change_each_flash(spdf)
    spdf['pre_change'] = spdf['change'].shift(-1)
    spdf['pre_omitted'] = spdf['omitted'].shift(-1)

    # spdf = add_epoch_times(spdf) # MJD EDIT

    spdf = add_mean_running_speed(spdf, running_speed)

    if eye_tracking is not None:
        try:  # if eye tracking data is not present or cant be loaded
            spdf = add_mean_pupil_area(spdf, eye_tracking)
        except BaseException:  # set to NaN
            spdf['mean_pupil_area'] = np.nan

    spdf = add_licks_each_flash(spdf, licks)
    spdf = add_response_latency(spdf)
    spdf = add_rewards_each_flash(spdf, rewards)
    spdf['licked'] = [True if len(licks) > 0 else False for licks in
                      spdf.licks.values]
    # lick rate per second
    spdf['lick_rate'] = spdf['licked'].rolling(window=320, min_periods=1,
                                               win_type='triang').mean() / .75

    spdf['rewarded'] = [True if len(rewards) > 0 else False for rewards in spdf.rewards.values]

    # (rewards/stimulus)*(1 stimulus/.750s) = rewards/second
    spdf['reward_rate_per_second'] = spdf['rewarded'].rolling(window=320, min_periods=1,
                                                              win_type='triang').mean() / .75  # units of rewards per second
    # (rewards/stimulus)*(1 stimulus/.750s)*(60s/min) = rewards/min
    spdf['reward_rate'] = spdf['rewarded'].rolling(window=320, min_periods=1, 
                                                   win_type='triang').mean() * (60 / .75)  # units of rewards/min

    reward_threshold = 2 / 3  # 2/3 rewards per minute = 1/90 rewards/second
    spdf['engaged'] = [x > reward_threshold for x in spdf['reward_rate']]
    spdf['engagement_state'] = ['engaged' if True else 'disengaged' for engaged in spdf['engaged'].values]

    spdf = add_response_latency(spdf)
    # spdf = reformat.add_image_contrast_to_stimulus_presentations(spdf)
    spdf = add_time_from_last_lick(spdf, licks)
    spdf = add_time_from_last_reward(spdf, rewards)
    spdf = add_time_from_last_change(spdf)
    try:  # behavior only sessions dont have omissions
        spdf = add_time_from_last_omission(spdf)
        spdf['flash_after_omitted'] = spdf['omitted'].shift(1)
    except BaseException:
        pass

    # spdf['flash_after_change'] = spdf['change'].shift(1)
    # spdf['image_name_next_flash'] = spdf['image_name'].shift(-1)
    # spdf['image_index_next_flash'] = spdf['image_index'].shift(-1)
    # spdf['image_name_previous_flash'] = spdf['image_name'].shift(1)
    # spdf['image_index_previous_flash'] = spdf['image_index'].shift(1)
    # spdf['lick_on_next_flash'] = spdf['licked'].shift(-1)
    # spdf['lick_rate_next_flash'] = spdf['lick_rate'].shift(-1)
    # spdf['lick_on_previous_flash'] = spdf['licked'].shift(1)
    # spdf['lick_rate_previous_flash'] = spdf['lick_rate'].shift(1)
    # if behavior_session_id:
    #     if check_if_model_output_available(behavior_session_id):
    #         spdf = add_model_outputs_to_stimulus_presentations(
    #             spdf, behavior_session_id)
    #     else:
    #         print('model outputs not available')
    return spdf


####################################################################################################
# REFORMAT/ANNOTATIONS: Functions to add columns to stimulus_presentations_df
####################################################################################################

# TODO: maybe general tool
def convert_running_speed(running_speed_obj):
    '''
    running speed is returned as a custom object, inconsistent with other attrs.
    should be a dataframe with cols for timestamps and speed.

    ARGS: running_speed object
    RETURNS: dataframe with columns timestamps and speed
    '''
    return pd.DataFrame({
        'timestamps': running_speed_obj.timestamps,
        'speed': running_speed_obj.values
    })


def add_change_each_flash(stimulus_presentations_df):
    '''Add change flash column to stimulus_presentations_table

        Parameters
        ----------
        stimulus_presentations_df : pd.DataFrame
            stimulus_presentations_df

        Returns
        -------
        stimulus_presentations_df : pd.DataFrame
            stimulus_presentations_table with ['change'] column added

    '''

    changes = find_change(stimulus_presentations_df['image_index'], 
                          get_omitted_index(stimulus_presentations_df))
    stimulus_presentations_df['change'] = changes
    return stimulus_presentations_df


def add_mean_running_speed(stimulus_presentations_df, running_speed, range_relative_to_stimulus_start=[0, 0.75]):
    '''
    Append a column to stimulus_presentations_df which contains the mean running speed between

    Args:
        stimulus_presentations_df(pd.DataFrame): dataframe of stimulus presentations.
                Must contain: 'start_time'
        running_speed (pd.DataFrame): dataframe of running speed.
            Must contain: 'speed', 'timestamps'
        range_relative_to_stimulus_start (list with 2 elements): start and end of the range
            relative to the start of each stimulus to average the running speed.
    Returns:
        nothing, modifies session in place. Same as the input, but with 'mean_running_speed' column added
    '''
    if isinstance(running_speed, pd.DataFrame):
        mean_running_speed_df = mean_running_speed(stimulus_presentations_df,
                                                       running_speed,
                                                       range_relative_to_stimulus_start)
    else:
        mean_running_speed_df = mean_running_speed(stimulus_presentations_df,
                                                       convert_running_speed(running_speed),
                                                       range_relative_to_stimulus_start)

    stimulus_presentations_df["mean_running_speed"] = mean_running_speed_df
    return stimulus_presentations_df


def add_mean_pupil_area(stimulus_presentations, eye_tracking, range_relative_to_stimulus_start=[0, 0.75]):
    '''
    Append a column to stimulus_presentations which contains the mean pupil area (in pixels^2) in the window provided.

    Args:
        stimulus_presentations(pd.DataFrame): dataframe of stimulus presentations.
                Must contain: 'start_time'
        eye_tracking (pd.DataFrame): dataframe of eye tracking data.
            Must contain: 'pupil_area', 'timestamps'
        range_relative_to_stimulus_start (list with 2 elements): start and end of the range
            relative to the start of each stimulus to average the pupil area.
    Returns:
        nothing, modifies session in place. Same as the input, but with 'mean_pupil_area' column added
    '''
    mean_pupil_area_df = mean_pupil_area(stimulus_presentations,
                                             eye_tracking,
                                             range_relative_to_stimulus_start)

    stimulus_presentations["mean_pupil_area"] = mean_pupil_area_df
    return stimulus_presentations


def add_licks_each_flash(stimulus_presentations, licks, range_relative_to_stimulus_start=[0, 0.75]):
    '''
    Append a column to stimulus_presentations which contains the timestamps of licks that occur
    in a range relative to the onset of the stimulus.

    Args:
        stimulus_presentations (pd.DataFrame): dataframe of stimulus presentations.
            Must contain: 'start_time'
        licks (pd.DataFrame): lick dataframe. Must contain 'timestamps'
        range_relative_to_stimulus_start (list with 2 elements): start and end of the range
            relative to the start of each stimulus to average the running speed.
    Returns:
        nothing, modifies session in place. Same as the input, but with 'licks' column added
    '''

    result = licks_each_flash(stimulus_presentations,
                                        licks,
                                        range_relative_to_stimulus_start)
    stimulus_presentations['licks'] = result
    return stimulus_presentations


def add_rewards_each_flash(stimulus_presentations, rewards, range_relative_to_stimulus_start=[0, 0.75]):
    '''
    Append a column to stimulus_presentations which contains the timestamps of rewards that occur
    in a range relative to the onset of the stimulus.

    Args:
        stimulus_presentations (pd.DataFrame): dataframe of stimulus presentations.
            Must contain: 'start_time'
        rewards (pd.DataFrame): rewards dataframe. Must contain 'timestamps'
        range_relative_to_stimulus_start (list with 2 elements): start and end of the range
            relative to the start of each stimulus to average the running speed.
    Returns:
        nothing. session.stimulus_presentations is modified in place with 'rewards' column added
    '''

    result = rewards_each_flash(stimulus_presentations,
                                                rewards,
                                                range_relative_to_stimulus_start)
    stimulus_presentations['rewards'] = result
    return stimulus_presentations


def add_time_from_last_lick(stimulus_presentations, licks):
    '''
        Adds a column in place to session.stimulus_presentations['time_from_last_lick'], which is the time, in seconds
        since the last lick

        Args:
        stimulus_presentations (pd.DataFrame): dataframe of stimulus presentations.
            Must contain: 'start_time'
        licks (pd.DataFrame): lick dataframe. Must contain 'timestamps'
        range_relative_to_stimulus_start (list with 2 elements): start and end of the range
            relative to the start of each stimulus to average the running speed.

        Returns:
            modified stimulus_presentations table
    '''
    lick_times = licks['timestamps'].values
    flash_times = stimulus_presentations["start_time"].values
    if len(lick_times) < 5:  # Passive sessions
        time_from_last_lick = np.full(len(flash_times), np.nan)
    else:
        time_from_last_lick = time_from_last(flash_times, lick_times)
    stimulus_presentations["time_from_last_lick"] = time_from_last_lick
    return stimulus_presentations


def add_time_from_last_reward(stimulus_presentations, rewards):
    '''
        Adds a column to stimulus_presentations['time_from_last_reward'], which is the time, in seconds
        since the last reward

        Args:
        stimulus_presentations (pd.DataFrame): dataframe of stimulus presentations.
            Must contain: 'start_time'
        rewards (pd.DataFrame): rewards dataframe. Must contain 'timestamps'
        range_relative_to_stimulus_start (list with 2 elements): start and end of the range
            relative to the start of each stimulus to average the running speed.
        Returns:
            modified stimulus_presentations table
    '''
    reward_times = rewards['timestamps'].values
    flash_times = stimulus_presentations["start_time"].values

    if len(reward_times) < 1:  # Sometimes mice are bad
        time_from_last_reward = np.full(len(flash_times), np.nan)
    else:
        time_from_last_reward = time_from_last(flash_times, reward_times)
    stimulus_presentations["time_from_last_reward"] = time_from_last_reward
    return stimulus_presentations


def add_time_from_last_change(stimulus_presentations):
    '''
        Adds a column to session.stimulus_presentations, 'time_from_last_change', which is the time, in seconds
        since the last image change

        ARGS: SDK session object
        MODIFIES: session.stimulus_presentations
        RETURNS: stimulus_presentations
    '''
    flash_times = stimulus_presentations["start_time"].values
    change_times = stimulus_presentations.query('is_change')['start_time'].values
    time_from_last_change = time_from_last(flash_times, change_times)
    stimulus_presentations["time_from_last_change"] = time_from_last_change
    return stimulus_presentations


def add_time_from_last_omission(stimulus_presentations):
    '''
        Adds a column to session.stimulus_presentations, 'time_from_last_omission', which is the time, in seconds
        since the last stimulus omission

        ARGS: SDK session object
        MODIFIES: session.stimulus_presentations
        RETURNS: stimulus_presentations
    '''
    flash_times = stimulus_presentations["start_time"].values
    omission_times = stimulus_presentations.query('omitted')['start_time'].values
    time_from_last_omission = time_from_last(flash_times, omission_times, side='left')
    stimulus_presentations["time_from_last_omission"] = time_from_last_omission
    return stimulus_presentations


# TODO: maybe general utils? need to fix
def add_epoch_times(df, time_column='start_time', epoch_duration_mins=10):
    """
    Add column called 'epoch' with values as an index for the epoch within a session, for a given epoch duration.

    :param df: dataframe with a column indicating event start times. Can be stimulus_presentations or trials table.
    :param time_column: name of column in dataframe indicating event times
    :param epoch_duration_mins: desired epoch length in minutes
    :return: input dataframe with epoch column added
    """
    start_time = df[time_column].values[0]
    stop_time = df[time_column].values[-1]
    epoch_times = np.arange(start_time, stop_time, epoch_duration_mins * 60)
    df['epoch'] = None
    for i, time in enumerate(epoch_times):
        if i < len(epoch_times) - 1:
            indices = df[(df[time_column] >= epoch_times[i]) & (df[time_column] < epoch_times[i + 1])].index.values
        else:
            indices = df[(df[time_column] >= epoch_times[i])].index.values
        df.at[indices, 'epoch'] = i
    return df


####################################################################################################
# UTILS
####################################################################################################

# TODO: could be moved to utils
def time_from_last(timestamps, event_times, side='right'):
    '''
    For each timestamp, returns the time from the most recent other time (in event_times)

    Args:
        timestamps (np.array): array of timestamps for which the 'time from last event' will be returned
        event_times (np.array): event timestamps
    Returns
        time_from_last_event (np.array): the time from the last event for each timestamp

    '''
    last_event_index = np.searchsorted(a=event_times, v=timestamps, side=side) - 1
    time_from_last_event = timestamps - event_times[last_event_index]

    # flashes that happened before the other thing happened should return nan
    time_from_last_event[last_event_index == -1] = np.nan

    return time_from_last_event


# TODO: could be moved to utils
def apply_to_window(values, timestamps, start_time, stop_time, func):
    '''
    Apply a function to an array of values that fall within a time window.

    '''
    if len(values) != len(timestamps):
        raise ValueError('values and timestamps must be the same length')
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError('timestamps must be monotonically increasing')
    if start_time < timestamps[0]:
        raise ValueError('start time must be within range of timestamps')
    if stop_time > timestamps[-1]:
        raise ValueError('stop time must be within range of timestamps')

    values_this_range = values[((timestamps >= start_time) & (timestamps < stop_time))]
    return func(values_this_range)


# TODO: remove?
# Applying np.mean to values in the window is what we usually want to do.
trace_average = partial(apply_to_window, func=np.mean)

####################################################################################################
# PROCESSING: Functions to process stimulus_presentations_df
####################################################################################################


def find_change(image_index, omitted_index=None):
    '''
    Get a boolean indicating whether each flash was a change flash.
    Args:
        image_index (pd.Series): The index of the presented image for each flash
        omitted_index (int): The index value for omitted stimuli
    Returns:
        change (np.array of bool): Whether each flash was a change flash
    '''
    change = np.diff(image_index) != 0
    change = np.concatenate([np.array([False]), change])  # First flash not a change
    if omitted_index is not None:
        omitted = image_index == omitted_index
        omitted_inds = np.flatnonzero(omitted)
        change[omitted_inds] = False
        if image_index.iloc[-1] == omitted_index:
            # If the last flash is omitted we can't set the +1 for that omitted idx
            change[omitted_inds[:-1] + 1] = False
        else:
            change[omitted_inds + 1] = False
    return change


def get_omitted_index(stimulus_presentations_df):
    '''
    Get the image index for omitted stimuli
    Args:
        stimulus_presentations_df (pd.DataFrame): dataframe containting 'image_name' and 'image_index' columns
    Returns:
        omitted_index (int): index corresponding to stimulus with name 'omitted'
    '''
    if 'omitted' in stimulus_presentations_df['image_name'].unique():
        omitted_indices = np.unique(stimulus_presentations_df.query('image_name == "omitted"')['image_index'].values)
        assert len(omitted_indices) == 1
        omitted_index = omitted_indices[0]
    else:
        omitted_index = None
    return omitted_index


def mean_running_speed(stimulus_presentations_df, running_speed_df,
                       range_relative_to_stimulus_start=[0, 0.25]):
    '''
    Append a column to stimulus_presentations_df which contains the mean running speed in a range relative to
    the stimulus start time.

    Args:
        stimulus_presentations_df (pd.DataFrame): dataframe of stimulus presentations.
            Must contain: 'start_time'
        running_speed_df (pd.DataFrame): dataframe of running speed.
            Must contain: 'speed', 'timestamps'
        range_relative_to_stimulus_start (list with 2 elements): start and end of the range
            relative to the start of each stimulus to average the running speed.
    Returns:
        flash_running_speed (pd.Series): mean running speed for each stimulus presentation.
    '''
    flash_running_speed = stimulus_presentations_df.apply(
        lambda row: trace_average(
            running_speed_df['speed'].values,
            running_speed_df['timestamps'].values,
            row["start_time"] + range_relative_to_stimulus_start[0],
            row["start_time"] + range_relative_to_stimulus_start[1],
        ),
        axis=1,
    )
    return flash_running_speed


def mean_pupil_area(stimulus_presentations_df, eye_tracking,
                    range_relative_to_stimulus_start=[0, 0.25]):
    '''
    Append a column to stimulus_presentations_df which contains the mean pupil area in a range relative to
    the stimulus start time.

    Args:
        stimulus_presentations_df (pd.DataFrame): dataframe of stimulus presentations.
            Must contain: 'start_time'
        eye_tracking (pd.DataFrame): dataframe of eye tracking data.
            Must contain: 'pupil_area', 'timestamps'
        range_relative_to_stimulus_start (list with 2 elements): start and end of the range
            relative to the start of each stimulus to average the pupil area.
    Returns:
        flash_running_speed (pd.Series): mean running speed for each stimulus presentation.
    '''
    flash_pupil_area = stimulus_presentations_df.apply(
        lambda row: trace_average(
            eye_tracking['pupil_area'].values,
            eye_tracking['timestamps'].values,
            row["start_time"] + range_relative_to_stimulus_start[0],
            row["start_time"] + range_relative_to_stimulus_start[1],
        ),
        axis=1,
    )
    return flash_pupil_area


def licks_each_flash(stimulus_presentations_df, licks_df,
                     range_relative_to_stimulus_start=[0, 0.75]):
    '''
    Append a column to stimulus_presentations_df which contains the timestamps of licks that occur
    in a range relative to the onset of the stimulus.

    Args:
        stimulus_presentations_df (pd.DataFrame): dataframe of stimulus presentations.
            Must contain: 'start_time'
        licks_df (pd.DataFrame): lick dataframe. Must contain 'timestamps'
        range_relative_to_stimulus_start (list with 2 elements): start and end of the range
            relative to the start of each stimulus to average the running speed.
    Returns:
        licks_each_flash (pd.Series): lick times that fell within the window relative to each stim time
    '''

    lick_times = licks_df['timestamps'].values
    stimulus_presentations_df['next_start'] = stimulus_presentations_df['start_time'].shift(-1)
    stimulus_presentations_df.at[stimulus_presentations_df.index[-1], 'next_start'] = stimulus_presentations_df.iloc[-1]['start_time'] + .75
    licks_each_flash = stimulus_presentations_df.apply(
        lambda row: lick_times[
            ((
                lick_times > row["start_time"]
            ) & (
                lick_times <= row["next_start"]
            ))
        ],
        axis=1,
    )
    stimulus_presentations_df.drop(columns=['next_start'], inplace=True)
    return licks_each_flash


def rewards_each_flash(stimulus_presentations_df, rewards_df,
                       range_relative_to_stimulus_start=[0, 0.75]):
    '''
    Append a column to stimulus_presentations_df which contains the timestamps of rewards that occur
    in a range relative to the onset of the stimulus.

    Args:
        stimulus_presentations_df (pd.DataFrame): dataframe of stimulus presentations.
            Must contain: 'start_time'
        rewards_df (pd.DataFrame): rewards dataframe. Must contain 'timestamps'
        range_relative_to_stimulus_start (list with 2 elements): start and end of the range
            relative to the start of each stimulus to average the running speed.
    Returns:
        rewards_each_flash (pd.Series): reward times that fell within the window relative to each stim time
    '''

    reward_times = rewards_df['timestamps'].values
    stimulus_presentations_df['next_start'] = stimulus_presentations_df['start_time'].shift(-1)
    stimulus_presentations_df.at[stimulus_presentations_df.index[-1], 'next_start'] = stimulus_presentations_df.iloc[-1]['start_time'] + .75
    rewards_each_flash = stimulus_presentations_df.apply(
        lambda row: reward_times[
            ((
                reward_times > row["start_time"]
            ) & (
                reward_times <= row["next_start"]
            ))
        ],
        axis=1,
    )
    stimulus_presentations_df.drop(columns=['next_start'], inplace=True)

    return rewards_each_flash


def get_block_index(image_index, omitted_index):
    '''
    A block is defined as a continuous epoch of presentation of the same stimulus.
    This func gets the block index for each stimulus presentation.

    e.g.:
    block_index:  |     0     |        1      |        2      |    3  ...
    stim ID:      |1| |1| |1| |2| |2| |2| |2| |3| |3| |3| |3| |1| |1| ...

    Args:
        image_index (np.array): Index of image for each stimulus presentation
        omitted_index (int): Index of omitted stimuli

    Returns:
        block_inds (np.array): Index of the current block for each stimulus
    '''
    changes = find_change(image_index, omitted_index)

    # Include the first presentation as a 'change'
    changes[0] = 1
    change_indices = np.flatnonzero(changes)

    flash_inds = np.arange(len(image_index))
    block_inds = np.searchsorted(a=change_indices, v=flash_inds, side="right") - 1
    return block_inds


def get_block_repetition_number(image_index, omitted_index):
    '''
    For each image block, is this the first time the image has been presented? second? etc.
    This function gets the repetition number (0 for first block of an image, 1 for second block..)
    for each stimulus presentation.

    e.g.:
    block_repetition_number:  |     0     |        0      |        0      |    1      |    1  ...
    stim ID:                  |1| |1| |1| |2| |2| |2| |2| |3| |3| |3| |3| |1| |1| |1| |3| |3| ...

    Args:
        image_index (np.array): Index of image for each stimulus presentation
        omitted_index (int): Index of omitted stimuli

    Returns:
        block_repetition_number (np.array): Repetition number for the current block
    '''
    block_inds = get_block_index(image_index, omitted_index)
    temp_df = pd.DataFrame({'image_index': image_index, 'block_index': block_inds})

    # For each image, what are the block inds where it was presented?
    blocks_per_image = temp_df.groupby("image_index").apply(
        lambda group: np.unique(group["block_index"])
    )

    # Go through each image, then enum the blocks where it was presented and return the rep
    block_repetition_number = np.empty(block_inds.shape)
    for image_index, image_blocks in blocks_per_image.iteritems():
        if image_index != omitted_index:
            for block_rep, block_ind in enumerate(image_blocks):
                block_repetition_number[np.where(block_inds == block_ind)] = block_rep

    return block_repetition_number


def get_repeat_within_block(image_index, omitted_index):
    '''
    Within each block, what repetition of the stimulus is this (0 = change flash)
    Returns NaN for omitted flashes

        e.g.:
    repeat_within_block:   0   1   2   0   1  NaN  2   0   1   2   3   0   1
    stim ID:              |1| |1| |1| |2| |2|     |2| |3| |3| |3| |3| |1| |1| ...

    Args:
        image_index (np.array): Index of image for each stimulus presentation
        omitted_index (int): Index of omitted stimuli

    Returns:
        repeat_within_block (np.array): Repetition number within the block for this image
    '''
    block_inds = get_block_index(image_index, omitted_index)
    temp_df = pd.DataFrame({'image_index': image_index, 'block_index': block_inds})
    repeat_number = np.full(len(image_index), np.nan)
    for ind_group, group in temp_df.groupby("block_index"):
        repeat = 0
        for ind_row, row in group.iterrows():
            if row["image_index"] != omitted_index:
                repeat_number[ind_row] = repeat
                repeat += 1
    return repeat_number


def add_response_latency(stimulus_presentations_df):
    logger.warning('Untested extended stimulus processing function (add_response_latency). Use at your own risk.')
    st = stimulus_presentations_df.copy()
    st['response_latency'] = st['licks'] - st['start_time']
    # st = st[st.response_latency.isnull()==False] #get rid of random NaN values
    st['response_latency'] = [response_latency[0] if len(response_latency) > 0 else np.nan for response_latency in st['response_latency'].values ]
    st['response_binary'] = [True if np.isnan(response_latency) == False else False for response_latency in st.response_latency.values]
    st['early_lick'] = [True if response_latency < 0.15 else False for response_latency in st['response_latency'].values ]
    return st


def add_inter_flash_lick_diff_to_stimulus_presentations(stimulus_presentations_df):
    logger.warning('Untested extended stimulus processing function (add_inter_flash_lick_diff_to_stimulus_presentations). Use at your own risk.')
    st = stimulus_presentations_df.copy()
    st['first_lick'] = [licks[0] if len(licks) > 0 else np.nan for licks in st['licks'].values]
    st['last_lick'] = [licks[-1] if len(licks) > 0 else np.nan for licks in st['licks'].values]
    st['previous_trial_last_lick'] = np.hstack((np.nan, st.last_lick.values[:-1]))
    st['inter_flash_lick_diff'] = st['previous_trial_last_lick'] - st['first_lick']
    return st


def add_first_lick_in_bout_to_stimulus_presentations(stimulus_presentations_df):
    logger.warning('Untested extended stimulus processing function (add_first_lick_in_bout_to_stimulus_presentations). Use at your own risk.')
    st = stimulus_presentations_df.copy()  # get median inter lick interval to threshold
    lick_times = st[st.licks.isnull() == False].licks.values
    median_inter_lick_interval = np.median(np.diff(np.hstack(list(lick_times))))
    # create first lick in bout boolean
    st['first_lick_in_bout'] = False  # set all to False
    indices = st[st.response_binary].index
    st.at[indices, 'first_lick_in_bout'] = True  # set stimulus_presentations_df with a lick to True
    indices = st[(st.response_binary) & (st.inter_flash_lick_diff < median_inter_lick_interval * 3)].index
    st.at[indices, 'first_lick_in_bout'] = False  # set stimulus_presentations_df with low inter lick interval back to False
    return st


def get_consumption_licks(stimulus_presentations_df):
    logger.warning('Untested extended stimulus processing function (get_consumption_licks). Use at your own risk.')
    st = stimulus_presentations_df.copy()
    lick_times = st[st.licks.isnull() == False].licks.values
    # need to make this a hard threshold
    median_inter_lick_interval = np.median(np.diff(np.hstack(list(lick_times))))
    st['consumption_licks'] = False
    for row in range(len(st)):
        row_data = st.iloc[row]
        if (row_data.change == True) and (row_data.first_lick_in_bout == True):
            st.loc[row, 'consumption_licks'] = True
        if (st.iloc[row - 1].consumption_licks == True) & (st.iloc[row].inter_flash_lick_diff < median_inter_lick_interval * 3):
            st.loc[row, 'consumption_licks'] = True
    return st


def add_prior_image_to_stimulus_presentations(stimulus_presentations_df):
    logger.warning('Untested extended stimulus processing function (add_prior_image_to_stimulus_presentations). Use at your own risk.')
    prior_image_name = [None]
    prior_image_name = prior_image_name + list(stimulus_presentations_df.image_name.values[:-1])
    stimulus_presentations_df['prior_image_name'] = prior_image_name
    return stimulus_presentations_df
