import numpy as np
import pandas as pd


def parse_lick_type(lick_type: str = "lick"):
    if lick_type == "lick" or lick_type == "licks":
        lick_key = "lick_times"
    elif lick_type == "bout" or lick_type == "bouts":
        lick_key = "lick_bout_times"
    elif lick_type == "lick_times" or lick_type == "lick_bout_times":
        lick_key = lick_type
    else:
        raise ValueError("lick_type must be 'lick', 'bout', 'lick_times', or 'lick_bout_times")

    return lick_key


def get_session_licks_from_change(ext_trials_df: pd.DataFrame, lick_type: str = "lick") -> np.ndarray:
    """Get lick times for a session, relative to change time

    Parameters
    ----------
    ext_trials_df : pd.DataFrame
        dataframe with lick times
    lick_type : str, optional
        lick type, by default "lick"

    Returns
    -------
        np.ndarray
        lick times for a session
    """
    lick_key = parse_lick_type(lick_type)

    # check that lick_key is in the dataframe
    if lick_key not in ext_trials_df.columns:
        raise ValueError("lick_key not in dataframe")

    # TODO: uuid function
    if len(ext_trials_df['behavior_session_uuid'].unique()) > 1:
        print("Warning: more than 1 unique ['behavior_session_uuid'] in dataframe")

    licks = []

    for _, row in ext_trials_df.iterrows():

        lt = np.array(row[lick_key]) - row['change_time']

        # the last trial is buggy, and has very long duration,so we'll just ignore it
        if np.any(lt > 60):
            # if np.any(lt > row['endtime'] - row['change_time']):

            # print(row['endtime'], row['change_time'])
            # print("Warning: licks past trial end")
            # print(row['behavior_session_uuid'], row['trial'])
            # print()
            continue

        else:
            # build lt array
            licks.append(lt)

    session_licks = np.concatenate(licks)

    return session_licks
