import numpy as np
from scipy.ndimage import median_filter as medfilt
import uuid
from dateutil import tz, parser
import datetime
import logging
from functools import wraps


def catch_trials_pkl_hack(pkl_dict):
    """Hack to get around pkl error when loading catch trials

    TODO: exand doc
    Basically, for associative learning pilots in 2022/2023, there are no
    catch trials.So this inserts the correct key and sets them all to false

    Parameters
    ----------
    pkl_dict : dict
        pkl_file dict with catch trials
    
    Returns
    -------
    dict
        pkl_file dict with catch trials set to false
    """
    for t in pkl_dict['items']['behavior']["trial_log"]:
        t["trial_params"]["catch"] = False
    return pkl_dict



# VBA
# vba.analyze
# TODO UPDATE
def calc_deriv(x, time):
    dx = np.diff(x)
    dt = np.diff(time)
    dxdt_rt = np.hstack((np.nan, dx / dt))
    dxdt_lt = np.hstack((dx / dt, np.nan))

    dxdt = np.vstack((dxdt_rt, dxdt_lt))

    dxdt = np.nanmean(dxdt, axis=0)

    return dxdt


# VBA
# vba.analyze
# TODO UPDATE
def deg_to_dist(speed_deg_per_s):
    '''
    takes speed in degrees per second
    converts to radians
    multiplies by radius (in cm) to get linear speed in cm/s
    '''
    wheel_diameter = 6.5 * 2.54  # 6.5" wheel diameter
    running_radius = 0.5 * (
        2.0 * wheel_diameter / 3.0)  # assume the animal runs at 2/3 the distance from the wheel center
    running_speed_cm_per_sec = np.pi * speed_deg_per_s * running_radius / 180.
    return running_speed_cm_per_sec


# VBA
# vba.analyze
# TODO UPDATE
def compute_running_speed(dx_raw, time, v_sig, v_in, smooth=False):
    """Calculate running speed

    #FROM VBA
    #TODO UPDATE

    Parameters
    ----------
    dx_raw: numpy.ndarray
        dx values for each stimulus frame
    time: numpy.ndarray
        timestamps for each stimulus frame
    v_sig: numpy.ndarray
        v_sig for each stimulus frame: currently unused
    v_in: numpy.ndarray
        v_in for each stimulus frame: currently unused
    smooth: boolean, default=False
        flag to smooth output: not implemented

    Returns
    -------
    numpy.ndarray
        Running speed (cm/s)
    """
    dx = medfilt(dx_raw, size=5)  # remove big, single frame spikes in encoder values
    dx = np.cumsum(dx)  # wheel rotations
    speed = calc_deriv(dx, time)  # speed in degrees/s
    speed = deg_to_dist(speed)

    if smooth:
        raise NotImplementedError

    return speed


###############################################################################
# vba.uuid_utils
###############################################################################


NAMESPACE_VISUAL_BEHAVIOR = uuid.UUID('a4b1bc02-4490-4a61-82db-f5c274a77080')


# TODO UPDATE
def create_mouse_namespace(mouse_id):
    return uuid.uuid5(NAMESPACE_VISUAL_BEHAVIOR, mouse_id)


# TODO UPDATE
def create_session_uuid(mouse_id, session_datetime_iso_utc):
    mouse_namespace = create_mouse_namespace(mouse_id)
    return uuid.uuid5(mouse_namespace, session_datetime_iso_utc)


# TODO UPDATE
def make_deterministic_session_uuid(mouse_id, startdatetime):
    start_time_datetime = parser.parse(startdatetime)
    start_time_datetime_utc = start_time_datetime.astimezone(tz.gettz("UTC")).isoformat()
    behavior_session_uuid = create_session_uuid(str(mouse_id), start_time_datetime_utc)
    return behavior_session_uuid


###############################################################################
# vba.utilities
###############################################################################


# TODO UPDATE
def local_time(iso_timestamp, timezone=None):
    if isinstance(iso_timestamp, datetime.datetime):
        dt = iso_timestamp
    else:
        dt = parser.parse(iso_timestamp)

    if not dt.tzinfo:
        dt = dt.replace(tzinfo=tz.gettz('America/Los_Angeles'))
    return dt.isoformat()


# TODO UPDATE
class ListHandler(logging.Handler):
    """docstring for ListHandler."""

    def __init__(self, log_list):
        super(ListHandler, self).__init__()
        self.log_list = log_list

    def emit(self, record):
        entry = self.format(record)
        self.log_list.append(entry)


# TODO UPDATE
DoubleColonFormatter = logging.Formatter(
    "%(levelname)s::%(name)s::%(message)s",
)

# TODO UPDATE
def inplace(func):
    """ decorator which allows functions that modify a dataframe inplace
    to use a copy instead
    """

    @wraps(func)
    def df_wrapper(df, *args, **kwargs):

        try:
            inplace = kwargs.pop('inplace')
        except KeyError:
            inplace = False

        if inplace is False:
            df = df.copy()

        func(df, *args, **kwargs)

        if inplace is False:
            return df
        else:
            return None

    return df_wrapper