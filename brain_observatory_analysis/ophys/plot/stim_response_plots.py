import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from allensdk.brain_observatory.behavior.behavior_ophys_experiment import \
    BehaviorOphysExperiment
# from allensdk.brain_observatory.behavior.stimulus_processing import \
#     get_stimulus_epoch_table



def plot_mean_trace(traces, timestamps, ylabel='dF/F', legend_label=None, color='k', interval_sec=1, xlim_seconds=[-2, 2],
                    plot_sem=True, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    trace = np.mean(traces, axis=0)
    sem = (np.std(traces)) / np.sqrt(float(len(traces)))
    ax.plot(timestamps, trace, label=legend_label, linewidth=2, color=color)
    if plot_sem:
        ax.fill_between(timestamps, trace + sem, trace - sem, alpha=0.5, color=color)
    ax.set_xticks(np.arange(int(timestamps[0]), int(timestamps[-1]) + 1, interval_sec))
    ax.set_xlim(xlim_seconds)
    ax.set_xlabel('time (sec)')
    ax.set_ylabel(ylabel)
    
    sns.despine(ax=ax)
    return ax


def plot_flashes_on_trace(ax, timestamps, change=None, omitted=False, alpha=0.075, facecolor='gray'):
    """
    plot stimulus flash durations on the given axis according to the provided timestamps
    """
    stim_duration = 0.2502
    blank_duration = 0.5004
    change_time = 0
    start_time = timestamps[0]
    end_time = timestamps[-1]
    interval = (blank_duration + stim_duration)
    # after time 0
    if omitted:
        array = np.arange((change_time + interval), end_time, interval)  # image array starts at the next interval
        # plot a dashed line where the stimulus time would have been
        ax.axvline(x=change_time, ymin=0, ymax=1, linestyle='--', color=sns.color_palette()[9], linewidth=1.5)
    else:
        array = np.arange(change_time, end_time, interval)
    for i, vals in enumerate(array):
        amin = array[i]
        amax = array[i] + stim_duration
        if change and (i == 0):
            change_color = sns.color_palette()[0]
            ax.axvspan(amin, amax, facecolor=change_color, edgecolor='none', alpha=alpha * 1.5, linewidth=0, zorder=1)
        else:
            ax.axvspan(amin, amax, facecolor=facecolor, edgecolor='none', alpha=alpha, linewidth=0, zorder=1)
    # if change == True:
    #     alpha = alpha / 2.
    else:
        alpha
    # before time 0
    array = np.arange(change_time, start_time - interval, -interval)
    array = array[1:]
    for i, vals in enumerate(array):
        amin = array[i]
        amax = array[i] + stim_duration
        ax.axvspan(amin, amax, facecolor=facecolor, edgecolor='none', alpha=alpha, linewidth=0, zorder=1)
    return ax


def plot_stimulus_response_df_trace(stimulus_response_df, time_window=[-1, 1], change=True, omitted=False,
                                    ylabel=None, legend_label=None, title=None, color='k', ax=None):
    """
    Plot average +/- sem trace for a subset of a stimulus_response_df, loaded via loading.get_stimulus_response_df()
    or directly from mindscope_utilities.visual_behavior_ophys.data_formatting.get_stimulus_response_df()
    If the provided stimulus_response_df only has a single row, with a single trace, that trace will be plotted without sem
    :param stimulus_response_df:
    :param time_window:
    :param change:
    :param omitted:
    :param ylabel:
    :param legend_label:
    :param title:
    :param color:
    :param ax:
    :return:
    """
    if 'trace' not in stimulus_response_df.keys():
        print('something other than a stimulus response df was provided, must include trace column')
    
    # get traces and timestamps
    traces = np.vstack(stimulus_response_df.trace.values)
    timestamps = stimulus_response_df.trace_timestamps.values[0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    ax = plot_mean_trace(traces, timestamps, ylabel=ylabel, legend_label=legend_label, color=color,
                         interval_sec=1, xlim_seconds=time_window, plot_sem=True, ax=ax)

    ax = plot_flashes_on_trace(ax, timestamps, change=change, omitted=omitted, alpha=0.15, facecolor='gray')

    if title:
        ax.set_title(title)

    return ax



def plot_mean_stimulus_response_df_trace(mean_stimulus_response_df, time_window=[-1, 1], change=True, omitted=False,
                                    ylabel=None, legend_label=None, title=None, color='k', ax=None):
    """
    Plot the mean +/- sem trace for a single row, or set of rows, from a mean_stimulus_response_df, 
    created via ophys.stimulus_response_df.get_mean_df() or .get_standard_mean_df()
    If a single row of mean_stimulus_response_df is provided, 
    function will plot the trace contained in the 'mean_trace' column with sem from 'sem_trace' column. 
    If multiple rows are provided, the average and SEM of the mean traces will be computed and plotted.

    :param mean_stimulus_response_df:
    :param time_window:
    :param change:
    :param omitted:
    :param ylabel:
    :param legend_label:
    :param title:
    :param color:
    :param ax:
    :return:
    """
    # check that its actually an averaged stim response df
    if 'mean_trace' not in mean_stimulus_response_df:
        print('something other than a mean stimulus response df was provided, must include mean_trace column')
    
    # timestamps should be the same for all rows of mean_stimulus_response_df
    timestamps = mean_stimulus_response_df.trace_timestamps.values[0]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    # for single row (i.e. cell) in mean_stimulus_response_df, plot the mean_trace and sem_trace columns
    if len(mean_stimulus_response_df) == 1:
        trace = mean_stimulus_response_df.mean_trace.values[0]
        sem = mean_stimulus_response_df.sem_trace.values[0]
        ax.plot(timestamps, trace, label=legend_label, linewidth=2, color=color)
        ax.fill_between(timestamps, trace + sem, trace - sem, alpha=0.5, color=color)
        ax.set_xlim(time_window)
        ax.set_xlabel('time (sec)')
        ax.set_ylabel(ylabel)
    else: # if there are multiple rows of mean_stimulus_response_df provided, plot the average
        traces = mean_stimulus_response_df.mean_trace.values
        ax = plot_mean_trace(traces, timestamps, ylabel=ylabel, legend_label=legend_label, color=color,
                            interval_sec=1, xlim_seconds=time_window, plot_sem=True, ax=ax)

    ax = plot_flashes_on_trace(ax, timestamps, change=change, omitted=omitted, alpha=0.15, facecolor='gray')

    if title:
        ax.set_title(title)

    return ax


def get_experience_level_colors():
    """
    get color map corresponding to Familiar, Novel 1 and Novel >1
    Familiar = red
    Novel 1 = blue
    Novel >1 = lighter blue
    """
    import seaborn as sns

    reds = sns.color_palette('Reds_r', 6)[:5][::2]
    blues = sns.color_palette('Blues_r', 6)[:5][::2]
    purples = sns.color_palette('Purples_r', 6)[:5][::2]

    colors = [reds[0], blues[0], purples[0]]

    return colors