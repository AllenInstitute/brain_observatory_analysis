import numpy as np
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from brain_observatory_analysis.ophys import correlation_analysis as ca
from brain_observatory_analysis.ophys import data_formatting as df
from brain_observatory_analysis.utilities import data_utils as du


def _get_pupil_diameter(exp):
    if len(exp.eye_tracking) > 0:  # Sometimes, eye_tracking is empty
        eye_timestamps = exp.eye_tracking.timestamps.values
        if 'pupil_diameter' in exp.eye_tracking.columns:
            pupil_diameter = exp.eye_tracking.pupil_diameter.values
        elif 'pupil_area' in exp.eye_tracking.columns:
            pupil_diameter = np.sqrt(exp.eye_tracking.pupil_area.values / np.pi)
        else:
            pupil_diameter = None
    else:
        pupil_diameter = None
        eye_timestamps = None
    return pupil_diameter, eye_timestamps


def _get_interpolated_running(exp, timepoints):
    running_timestamps = exp.running_speed.timestamps.values
    running_speed = exp.running_speed.speed.values
    running_interp = du.get_interpolated_time_series(running_timestamps, running_speed, timepoints)
    return running_interp


def _get_interpolated_pupil_diameter(exp, timepoints):
    pupil_diameter, eye_timestamps = _get_pupil_diameter(exp)
    if pupil_diameter is not None:
        pupil_diameter_interp = du.get_interpolated_time_series(eye_timestamps, pupil_diameter, timepoints)
    else:
        pupil_diameter_interp = None
    return pupil_diameter_interp


def _get_interpolated_lickrate(exp, timepoints, lick_template_rate=100, lick_rate_window=1):
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
    return lickrate_interp


def plot_task_raster_with_behav_sort_by_corr(
        lamf_group, session_name, remove_auto_rewarded=True,
        lick_template_rate=100, lick_rate_window=1, sub_title_fontsize=10,
        num_cell_threshold=20, vmin=-3, vmax=5):
    fig, ax = plt.subplots(figsize=(12, 8))
    trace_df_task = df.get_trace_df_task(lamf_group, session_name, remove_auto_rewarded=remove_auto_rewarded)
    if len(trace_df_task) < num_cell_threshold:
        print('Too few cells to plot')
        return fig
    else:
        *_, sort_ind_task, remove_ind = ca.get_correlation_matrices(trace_df_task)
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

        running_interp = _get_interpolated_running(exp, timepoints)
        rax = divider.append_axes('bottom', size='20%', pad=0.25, sharex=zax)
        rax.plot(running_interp, color='C3')
        rax.set_xlim(0, len(task_trace_all_mean))
        rax.set_xticks([])
        rax.set_title('Running speed (cm/s)', loc='left', fontsize=sub_title_fontsize, color='C3')

        # Add pupil diameter by interpolating to the timepoints
        pupil_diameter, eye_timestamps = _get_pupil_diameter(exp)
        pax = divider.append_axes('bottom', size='20%', pad=0.25, sharex=zax)
        if pupil_diameter is not None:
            pupil_interp = du.get_interpolated_time_series(eye_timestamps, pupil_diameter, timepoints)
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
        lickrate_interp = _get_interpolated_lickrate(exp, timepoints)
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

    trace_df_no_task_list = df.get_trace_df_no_task(lamf_group, session_name)
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
            *_, sort_ind, remove_ind = ca.get_correlation_matrices(trace_df)
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
            running_interp = du.get_interpolated_time_series(running_timestamps, running_speed, timepoints)
            rax = divider.append_axes('bottom', size='20%', pad=0.25, sharex=zax)
            rax.plot(running_interp, color='C3')
            rax.set_xlim(0, len(trace_all_mean))
            rax.set_xticks([])
            rax.set_title('Running speed (cm/s)', loc='left', fontsize=sub_title_fontsize, color='C3')

            # Add pupil diameter by interpolating to the timepoints
            pupil_diameter, eye_timestamps = _get_pupil_diameter(exp)
            if pupil_diameter is not None:
                pupil_interp = du.get_interpolated_time_series(eye_timestamps, pupil_diameter, timepoints)
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


def get_traces_from_trace_df_task(exp_group, session_name, trace_df_task):
    # Get traces and behavioral variables
    trace_array, remove_ind, nan_frame_ind = ca.get_trace_array_from_trace_df(trace_df_task)
    # standardize trace_array
    # In each row, subtract the mean of the row
    trace_array_t = trace_array.T
    trace_array_std_t = (trace_array_t - np.mean(trace_array_t, axis=0)) / np.std(trace_array_t, axis=0)
    trace_array_std = trace_array_std_t.T

    oeid = exp_group.expt_table.query('session_name == @session_name').index.values[0]  # First one can be the representative one
    exp = exp_group.experiments[oeid]
    timepoints = trace_df_task.iloc[0].timepoints  # First one can be the representative one

    running_interp = _get_interpolated_running(exp, timepoints)
    pupil_interp = _get_interpolated_pupil_diameter(exp, timepoints)
    lickrate_interp = _get_interpolated_lickrate(exp, timepoints)

    finite_frame_ind = np.setdiff1d(range(len(running_interp)), nan_frame_ind)
    running_interp_finite_frame = running_interp[finite_frame_ind]
    pupil_interp_finite_frame = pupil_interp[finite_frame_ind]
    lickrate_interp_finite_frame = lickrate_interp[finite_frame_ind]

    return trace_array, trace_array_std, running_interp_finite_frame, pupil_interp_finite_frame, lickrate_interp_finite_frame, remove_ind