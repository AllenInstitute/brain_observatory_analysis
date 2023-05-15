import numpy as np
from matplotlib import pyplot as plt


# A function to plot running speed during gray periods
def plot_running_speed_gray_periods(expt_group, session_name, gray_period=5 * 60):
    """Plot running speed during gray periods.
    
    Parameters
    ----------
    expt_group : ExperimentGroup
        ExperimentGroup object.
    ession_name : str
        Name of the session, as in the Experiment Object
    gray_period : int, optional
        Length of gray period in seconds, by default 5 * 60

    Returns
    -------
    matplotlib.figure.Figure
        Figure object.
    """
    # Plot running speed during gray periods
    oeids = expt_group.expt_table.query('session_name==@session_name').index.values
    exp = expt_group.experiments[oeids[0]]  # Use the first experiment for the session (running is session-specific, same across experiments)
    
    timepoints = exp.running_speed.timestamps.values
    speed = exp.running_speed.speed.values
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