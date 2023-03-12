import numpy as np
import scipy


# A function to get interpolated time series
def get_interpolated_time_series(timestamps, values, new_timepoints):
    """Interpolate time series to new timepoints.
    Remove timestamps with nan values and pad with first and last values
    when new_timepoints are outside the range of timestamps.
    
    Parameters
    ----------
    timestamps : np.ndarray
        Timestamps of the original time series.
    values : np.ndarray
        Values of the original time series.
    new_timepoints : np.ndarray
        Timestamps of the new time series.

    Returns
    -------
    np.ndarray
        Values of the new time series.
    """

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