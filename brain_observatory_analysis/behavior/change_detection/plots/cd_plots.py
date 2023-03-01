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


def lick_raster_1(extended_trials: pd.DataFrame,
                ax: plt.Axes = None,
                lick_type: str = 'lick',
                xlims: tuple = (-1, 5),
                ymax: int = None,
                reward_window: bool = None,
                show_reward_window: bool = True,
                palette='trial_types',
                color_trials: bool = True,
                plot_aborted_trials: bool = False):


    # keys added
    # + 'last_rewarded_trial'
    # + 'rewarded' FIXED

    from visual_behavior.utilities import flatten_list
    from visual_behavior.translator.core.annotate import \
        colormap, trial_translator

    lick_key = parse_lick_type(lick_type)

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 6))

    if reward_window is None:
        reward_window = get_reward_window(extended_trials)

    if show_reward_window == True:  # NOQA E712
        ax.axvspan(reward_window[0], reward_window[1],
                   facecolor='k', alpha=0.3)


    # show change stimulus time with dash linbe
    ax.axvline(0, color='k', linestyle='--', linewidth=1, alpha = 0.5)


    lick_x = []
    lick_y = []
    reward_x = []
    reward_y = []

    last_change_index = 0 

    # check for trial key
    if 'trial' not in extended_trials.columns:
        extended_trials['trial'] = extended_trials.index

    for ii, idx in enumerate(extended_trials['trial']):
        y_index = idx # ii is original

        # all other trial type are change
        if extended_trials.loc[y_index]['trial_type'] != 'aborted':
            last_change_index = ii
            plot_y_index = extended_trials.loc[y_index]['cumulative_reward_number']
        if len(extended_trials.loc[y_index][lick_key]) > 0:

            if plot_aborted_trials and extended_trials.loc[y_index]['trial_type'] == 'aborted':
                    lt = np.array(extended_trials.loc[y_index][lick_key]) - \
                                  extended_trials.loc[last_change_index]['change_time']

            else: 

                # regular change trials
                lt = np.array(extended_trials.loc[idx][lick_key]) - \
                    extended_trials.loc[last_change_index]['change_time']

            lick_x.append(lt)
            lick_y.append(np.ones_like(lt) * last_change_index)

            

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

    if plot_aborted_trials:
        # get max of lick_x, ignore inf
        max_lick_x = np.nanmax(flatten_list(lick_x))

        # if above 500 s (8.3 min), set to 500
        if max_lick_x > 50:
            # warn user
            print('max lick time is above 500 s, setting to 500 s')
            max_lick_x = 50
        xlims = (xlims[0], max_lick_x)

    ax.set_xlim(xlims[0], xlims[1])

    if ymax is None or ymax is False:
        ax.set_ylim(-0.5, y_index + 0.5)
    else:
        ax.set_ylim(-0.5, ymax + 0.5)

    ax.invert_yaxis()

    ax.set_title('Lick Raster', fontsize=16)
    ax.set_ylabel('Trial Number', fontsize=14)
    ax.set_xlabel('Time from \nstimulus onset (s)', fontsize=14)



def lick_raster(extended_trials: pd.DataFrame,
                ax: plt.Axes = None,
                lick_type: str = 'lick',
                xlims: tuple = (-1, 5),
                ymax: int = None,
                reward_window: bool = None,
                show_reward_window: bool = True,
                palette='trial_types',
                color_trials: bool = True,
                plot_aborted_trials: bool = False,
                first_lick_only: bool = False):


    # keys added
    # + 'last_rewarded_trial'
    # + 'rewarded' FIXED

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
                   facecolor='purple', alpha=0.2)


    # show change stimulus time with dash linbe
    ax.axvline(0, color='k', linestyle='--', linewidth=1, alpha = 1)


    lick_x = []
    lick_y = []
    reward_x = []
    reward_y = []
    lick_types_list = []

    y_plot_index = np.nan
    y_df_index = np.nan

    # check for trial key
    if 'trial' not in extended_trials.columns:
        extended_trials['trial'] = extended_trials.index

    for ii, idx in enumerate(extended_trials['trial']):

        # all other trial type are change
        if extended_trials.loc[ii]['not_aborted_bool'] is True:
            y_df_index = ii
            y_plot_index = extended_trials.loc[ii]['not_aborted_trial_index']

        if len(extended_trials.loc[ii][lick_key]) > 0:

            if plot_aborted_trials and extended_trials.loc[ii]['trial_type'] == 'aborted':
                if np.isnan(y_df_index) or np.isnan(y_plot_index):
                    continue
                else:
                    lt = np.array(extended_trials.loc[ii][lick_key]) - \
                                    extended_trials.loc[y_df_index]['change_time']
                
                    # make an array of lick types for each lick, first lick 'aborted', the rest 'post-aborted'
                    lick_types = np.array(["post-aborted"] * lt.size)
                    lick_types[0] = "aborted"

            else:

                # regular change trials
                lt = np.array(extended_trials.loc[ii][lick_key]) - \
                    extended_trials.loc[ii]['change_time']

                # set lick types to color later
                if lt.size != 0:
                    # make lick_types array
                    lick_types = np.array(['post-change'] * lt.size)

                    # get index of first lick after .15
                    if np.any(lt > .15):
                        first_lick_after_change = np.where(lt > .15)[0][0]

                        # set lick_types for first lick after change to 'first-lick'
                        # TODO: may have trial translator
                        response_type = extended_trials.loc[ii]['response_type']
                        if extended_trials.loc[ii]["trial_type"] == "go" and response_type == "HIT":
                            lick_types[first_lick_after_change] = '1st-go'
                        elif extended_trials.loc[ii]["trial_type"] == "go" and response_type == "MISS":
                            lick_types[first_lick_after_change] = '1st-miss'
                        elif extended_trials.loc[ii]["trial_type"] == "catch":
                            lick_types[first_lick_after_change] = '1st-catch'

                    # get index of all licks before .15
                    if np.any(lt < .15):
                        licks_before_change = np.where(lt < .15)[0]

                        # set lick_types for licks before change to 'pre-change'
                        lick_types[licks_before_change] = 'pre-change'

            lick_x.append(lt)
            lick_y.append(np.ones_like(lt) * y_plot_index)

            lick_types_list.append(lick_types)


        if len(extended_trials.loc[idx]['reward_times']) > 0:
            rt = np.array(extended_trials.loc[idx]['reward_times']) - \
                extended_trials.loc[idx]['change_time']
            reward_x.append(rt)
            reward_y.append(np.ones_like(rt) * y_plot_index)

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
    
    # lick type colormap
    lick_type_colors = {'pre-change': 'gray',
                        'post-change': 'gray',
                        '1st-go': 'blue',
                        '1st-catch': 'green',
                        '1st-miss': 'red',
                        'aborted': 'orange',
                        'post-aborted': 'gray'}

    # plot licks
    lick_x = flatten_list(lick_x)
    lick_y = flatten_list(lick_y)
    lick_types_list = flatten_list(lick_types_list)


    # reverse all list orders (helps plotting of aborted trials)
    lick_x = lick_x[::-1]
    lick_y = lick_y[::-1]
    lick_types_list = lick_types_list[::-1]

    for x, y, lick_type in zip(lick_x, lick_y, lick_types_list):
            color = lick_type_colors[lick_type]
            if color == 'gray':
                alpha = 0.7
            else:
                alpha = 1
            ax.plot(x, y, '.', color=lick_type_colors[lick_type], markersize=5, alpha=alpha)

    #ax.plot(flatten_list(lick_x), flatten_list(lick_y), '.k',
    #        marker='o', markersize=2, )

    plot_reward = False
    if plot_reward:
        ax.plot(flatten_list(reward_x), flatten_list(reward_y),
                'o', color='blue', alpha=0.5, markersize=5)

    if plot_aborted_trials:
        # get max of lick_x, ignore inf
        max_lick_x = np.nanmax(flatten_list(lick_x))

        # if above 500 s (8.3 min), set to 500
        # if max_lick_x > 60:
        #     # warn user
        #     print('max lick time is above 500 s, setting to 500 s')
        #     max_lick_x = 60
        # xlims = (xlims[0], max_lick_x)

    ax.set_xlim(xlims[0], xlims[1])

    if ymax is None or ymax is False:
        ax.set_ylim(-0.5, y_plot_index + 0.5)
    else:
        ax.set_ylim(-0.5, ymax + 0.5)

    ax.invert_yaxis()

    ax.set_title('Lick Raster', fontsize=16)
    ax.set_ylabel('Trial Number', fontsize=14)
    ax.set_xlabel('Time from \nstimulus onset (s)', fontsize=14)

