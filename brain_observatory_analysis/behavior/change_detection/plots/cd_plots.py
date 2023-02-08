import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from ..analysis.licks import parse_lick_type


def get_reward_window(extended_trials: pd.DataFrame):
    # TODO move to utils
    try:
        reward_window = extended_trials.iloc[0].response_window
    except Exception:
        reward_window = [0.15, 1]
    return reward_window


def lick_raster(extended_trials: pd.DataFrame,
                ax: plt.Axes = None,
                lick_type: str = 'lick',
                xlims: tuple = (-1, 5),
                ymax: int = None,
                reward_window: bool = None,
                show_reward_window: bool = True,
                palette='trial_types',
                color_trials: bool = True):

    from visual_behavior.utilities import flatten_list
    from visual_behavior.translator.core.annotate import \
        colormap, trial_translator

    lick_key = parse_lick_type(lick_type)
    print(lick_key)

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 6))

    if reward_window is None:
        reward_window = get_reward_window(extended_trials)

    if show_reward_window == True:  # NOQA E712
        ax.axvspan(reward_window[0], reward_window[1],
                   facecolor='k', alpha=0.5)

    lick_x = []
    lick_y = []
    reward_x = []
    reward_y = []

    # check for trial key
    if 'trial' not in extended_trials.columns:
        extended_trials['trial'] = extended_trials.index

    for ii, idx in enumerate(extended_trials['trial']):
        y_index = idx # ii is original
        if len(extended_trials.loc[idx][lick_key]) > 0:
            lt = np.array(extended_trials.loc[idx][lick_key]) - \
                extended_trials.loc[idx]['change_time']
            lick_x.append(lt)
            lick_y.append(np.ones_like(lt) * y_index)

        if len(extended_trials.loc[idx]['reward_times']) > 0:
            rt = np.array(extended_trials.loc[idx]['reward_times']) - \
                extended_trials.loc[idx]['change_time']
            reward_x.append(rt)
            reward_y.append(np.ones_like(rt) * y_index)

        trial_type = trial_translator(extended_trials.loc[idx]['trial_type'],
                                      extended_trials.loc[idx]['response'])

        if color_trials:
            colors = colormap(trial_type, palette)
        else:
            colors = 'white'

        # ax.axhspan(ii - 0.5, ii + 0.5,
        #            facecolor=colors, alpha=0.5,
        #            edgecolor='lightgray')

    # ax.plot(flatten_list(lick_x), flatten_list(lick_y), '.k',
    #          marker = '|', markersize = 5, markeredgewidth = 2)

    ax.plot(flatten_list(lick_x), flatten_list(lick_y), '.k',
            marker='o', markersize=2)

    plot_reward = False
    if plot_reward:
        ax.plot(flatten_list(reward_x), flatten_list(reward_y),
                'o', color='blue', alpha=0.5, markersize=5)

    ax.set_xlim(xlims[0], xlims[1])

    if ymax is None or ymax is False:
        ax.set_ylim(-0.5, y_index + 0.5)
    else:
        ax.set_ylim(-0.5, ymax + 0.5)

    ax.invert_yaxis()

    ax.set_title('Lick Raster', fontsize=16)
    ax.set_ylabel('Trial Number', fontsize=14)
    ax.set_xlabel('Time from \nstimulus onset (s)', fontsize=14)
