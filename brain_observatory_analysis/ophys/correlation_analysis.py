import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import brain_observatory_analysis.ophys.data_formatting as df


def get_trace_array_from_trace_df(trace_df, nan_frame_prop_threshold=0.2, nan_cell_prop_threshold=0.2):
    trace_array = np.vstack(trace_df.trace.values)
    
    # Check if there are too many nan frames or cells with too many nan frames
    num_cell = len(trace_df)
    num_nan_frames_threshold = int(trace_array.shape[1] * nan_frame_prop_threshold)
    nan_frames = np.where(np.isnan(trace_array).sum(axis=0)>0)[0]
    num_nan_frames = len(nan_frames)

    remove_ind = np.zeros(0,)
    if num_nan_frames > num_nan_frames_threshold:
        num_nan_frames_each = np.isnan(trace_array).sum(axis=1)
        ind_many_nan_frames = np.where(num_nan_frames_each > num_nan_frames_threshold)[0]
        num_nan_cells = len(ind_many_nan_frames)
        if num_nan_cells / num_cell > nan_cell_prop_threshold:
            raise ValueError(f"Too many cells with nan frames > threshold {nan_frame_prop_threshold}: {num_nan_cells} out of {num_cell}")
        else:
            print(f"Removing {num_nan_cells} cells with nan frames proportion > threshold {nan_frame_prop_threshold}")
            remove_ind = ind_many_nan_frames
            trace_array = np.delete(trace_array, remove_ind, axis=0)
            nan_frames = np.where(np.isnan(trace_array).sum(axis=0)>0)[0]
            num_nan_frames = len(nan_frames)
            print(f"Removing {num_nan_frames} frames with nan values")
            trace_array = np.delete(trace_array, nan_frames, axis=1)
    else:
        print(f"Removing {num_nan_frames} frames with nan values")
        trace_array = np.delete(trace_array, nan_frames, axis=1)
    return trace_array, remove_ind, nan_frames


def get_correlation_matrices(trace_df, nan_frame_prop_threshold=0.2, nan_cell_prop_threshold=0.2):
    """ Get correlation matrices for a given trace dataframe
    If number of frames with nan values is small, it's okay to just remove those frames
    If number of frames with nan values is large, something is wrong with the data
        If number of cells with large number of nan frames is small, it's okay to just remove those cells

    Parameters
    ----------
    trace_df: pandas.DataFrame
        A dataframe containing traces for a given session and event type
    nan_frame_prop_threshold: float, optional
        Threshold for the proportion of nan frames in a cell, default 0.2
    nan_cell_prop_threshold: float, optional
        Threshold for the proportion of nan cells in a session, default 0.2
    
    Returns
    -------
    corr: np.ndarray (2d)
        Correlation matrix (No sorting, i.e., sorted by oeid and cell specimen id)
    corr_ordered: np.ndarray (2d)
        Correlation matrix (Sorted by mean correlation coefficient)    
    corr_ordered_by_region: np.ndarray (2d)
        Correlation matrix (Sorted by region and depth)
    xy_labels: list
        List of labels for corr_ordered_by_region
    xy_label_pos: list
        List of label positions for corr_ordered_by_region
    mean_corr_sorted_ind: np.ndarray (1d)
        Sorted indices of mean correlation coefficients (applied to corr to produce corr_ordered)
    remove_ind: np.ndarray (1d)
        Indices of cells to be removed due to large number of nan frames
    """
    trace_array, remove_ind, _ = get_trace_array_from_trace_df(trace_df, nan_frame_prop_threshold, nan_cell_prop_threshold)
    
    corr = np.corrcoef(trace_array)

    # sort by global mean correlation
    mean_corr = np.nanmean(corr, axis=0)
    mean_corr_sorted_ind = np.argsort(mean_corr)[::-1]
    corr_ordered = corr[mean_corr_sorted_ind, :][:, mean_corr_sorted_ind]

    numcell_cumsum = np.cumsum([x[0] for x in trace_df.groupby(['oeid']).count().values])
    xy_ticks = np.insert(numcell_cumsum, 0, 0)
    xy_labels = [f"{x}-{y}" for x, y in zip(trace_df.groupby(['oeid']).first().targeted_structure.values, trace_df.groupby(['oeid']).first().bisect_layer.values)]
    xy_label_pos = xy_ticks[:-1] + np.diff(xy_ticks) / 2

    # order the corr matrix by the mean correlation of each cell within each oeid
    trace_array_ordered_by_region = np.zeros_like(trace_array)
    for i in range(len(numcell_cumsum)):
        # find the order within each oeid
        within_mean_corr = np.mean(corr[xy_ticks[i]: xy_ticks[i + 1], xy_ticks[i]: xy_ticks[i + 1]], axis=1)
        within_order = np.argsort(-within_mean_corr)  # descending order
        trace_array_ordered_by_region[xy_ticks[i]:xy_ticks[i + 1], :] = trace_array[xy_ticks[i]: xy_ticks[i + 1], :][within_order, :]
    corr_ordered_by_region = np.corrcoef(trace_array_ordered_by_region)

    return corr, corr_ordered, corr_ordered_by_region, xy_labels, xy_label_pos, mean_corr_sorted_ind, remove_ind


# Comparing between different epochs and events
# Functions to get all epoch trace dfs and correlation matrices
def _append_results(results, trace_df, epoch):
    """Append results from trace_df

    Parameters
    ----------
    results: dict
        Dictionary to store results
    trace_df: pd.DataFrame
        Dataframe containing trace and other information
    epoch: str
        Epoch name
    
    Returns
    -------
    results: dict
        Dictionary to store results
    """
    results['epochs'].append(epoch)
    results['trace_dfs'].append(trace_df)
    corr, corr_ordered, corr_ordered_by_region, xy_labels, xy_label_pos, sorted_ind, remove_ind = get_correlation_matrices(trace_df)
    results['corr_matrices'].append(corr)
    results['corr_ordered_matrices'].append(corr_ordered)
    results['corr_ordered_by_region_matrices'].append(corr_ordered_by_region)
    results['xy_label_pos'].append(xy_label_pos)
    results['xy_labels'].append(xy_labels)
    results['sorted_inds'].append(sorted_ind)
    results['remove_inds'].append(remove_ind)
    return results


def get_all_epoch_trace_df_and_correlation_matrices(lamf_group, session_name, image_order=3,
                                                    inter_image_interval=0.75, output_sampling_rate=20):
    """Get all epoch trace dfs and correlation matrices

    Parameters
    ----------
    lamf_group: Experiment group
        Experiment group object
    session_name: str
        Session name
    image_order: int, optional
        Image order for 'images>n-change' event type, default 3
    inter_image_interval: float, optioanl
        Inter image interval, default 0.75
    output_sampling_rate: float
        Output sampling rate, default 20 (Hz)
    
    Returns
    -------
    epochs: list
        List of epochs
    trace_dfs: list
        List of trace dataframes
    corr_matrices: list
        List of correlation matrices
    corr_ordered_matrices: list
        List of correlation matrices ordered by global mean correlation
    corr_ordered_by_region_matrices: list
        List of correlation matrices ordered by mean correlation within each region and depth
    xy_label_pos_list: list
        List of x and y label positions for corr_ordered_by_region_matrices
    xy_labels_list: list
        List of x and y labels for corr_ordered_by_region_matrices
    sorted_inds_list: list
        List of sorted indices for corr_ordered_by_region_matrices
    remove_inds_list: list
        List of indices removed because of NaN frames
    """

    results = {'epochs': [],
               'trace_dfs': [],
               'corr_matrices': [],
               'corr_ordered_matrices': [],
               'corr_ordered_by_region_matrices': [],
               'xy_label_pos': [],
               'xy_labels': [],
               'sorted_inds': [],
               'remove_inds': []}

    trace_task_df = df.get_trace_df_task(lamf_group, session_name)
    if len(trace_task_df) > 0:
        results = _append_results(results, trace_task_df, 'task')
        
        trace_graypre_df, trace_graypost_df, trace_fingerprint_df = df.get_trace_df_no_task(lamf_group, session_name)
        if len(trace_graypre_df) > 0:
            assert (trace_task_df.index.values - trace_graypre_df.index.values).any() == False  # noqa: E712
            results = _append_results(results, trace_graypre_df, 'graypre')

        if len(trace_graypost_df) > 0:
            assert (trace_task_df.index.values - trace_graypost_df.index.values).any() == False  # noqa: E712
            results = _append_results(results, trace_graypost_df, 'graypost')
        
        if len(trace_fingerprint_df) > 0:
            assert (trace_task_df.index.values - trace_fingerprint_df.index.values).any() == False  # noqa: E712
            results = _append_results(results, trace_fingerprint_df, 'fingerprint')
        
        events = ['images>n-changes', 'changes', 'omissions']
        for event in events:
            trace_event_df = df.get_trace_df_event(lamf_group, session_name=session_name, event_type=event,
                                                   image_order=image_order, inter_image_interval=inter_image_interval, 
                                                   output_sampling_rate=output_sampling_rate)
            if len(trace_event_df) > 0:
                assert (trace_task_df.index.values - trace_event_df.index.values).any() == False  # noqa: E712
                if event == 'images>n-changes':
                    event = 'images'
                results = _append_results(results, trace_event_df, event)
        
    epochs, trace_dfs, corr_matrices, corr_ordered_matrices, corr_ordered_by_region_matrices, \
        xy_label_pos_list, xy_labels_list, sorted_inds_list, remove_inds_list = results.values()

    return epochs, trace_dfs, corr_matrices, corr_ordered_matrices, corr_ordered_by_region_matrices, \
        xy_label_pos_list, xy_labels_list, sorted_inds_list, remove_inds_list


def compare_correlation_matrices(compare_epochs, epochs, corr_matrices, corr_ordered_matrices, corr_ordered_by_region_matrices,
                                 xy_label_pos_list, xy_labels_list, sorted_inds_list, remove_inds_list, session_name,
                                 vmin=-0.4, vmax=0.8, cb_shrink_factor=0.7):
    """Compare correlation matrices
    Plot correlation matrices for different epochs and events
    along with their sorted correlation matrices by one correlation matrix (Reference matrix)

    Parameters
    ----------
    compare_epochs: list
        List of epochs to compare
    epochs: list
        List of epochs (results from get_all_epoch_trace_df_and_correlation_matrices)
    corr_matrices: list
        List of corr (results from get_all_epoch_trace_df_and_correlation_matrices)
    corr_ordered_matrices: list
        List of corr_ordered (results from get_all_epoch_trace_df_and_correlation_matrices)
    corr_ordered_by_region_matrices: list
        List of corr_ordered_by_region (results from get_all_epoch_trace_df_and_correlation_matrices)
    xy_label_pos_list: list
        List of xy_label_pos (results from get_all_epoch_trace_df_and_correlation_matrices)
    xy_labels_list: list
        List of xy_labels (results from get_all_epoch_trace_df_and_correlation_matrices)
    sorted_inds_list: list
        List of sorted_inds (results from get_all_epoch_trace_df_and_correlation_matrices)
    session_name: str
        Session name
    vmin: float, optional
        Minimum value for correlation matrix plot colorbar, default -0.4
    vmax: float, optional
        Maximum value for correlation matrix plot colorbar, default 0.8
    cb_shrink_factor: float, optional
        Colorbar shrink factor for sns.heatmap, default 0.7
    """

    num_rows = len(compare_epochs)
    inds = []
    for i in range(num_rows):
        inds.append(epochs.index(compare_epochs[i]))

    # Set remove_inds as the union of remove_inds_list
    remove_inds = remove_inds_list[inds[0]]
    for i in range(1, num_rows):
        remove_inds = np.union1d(remove_inds, remove_inds_list[inds[i]])
    # Only show the cell indice that are not removed in any of the compared matrices
    ref_sort_ind = [si for si in sorted_inds_list[inds[0]] if si not in remove_inds]
    comp_sorted_by_ref = []
    num_indice = []
    for i in range(1, num_rows):
        # For each compared matrix, reduce the cell indice by the number of removed cells
        # Because each matrix has different number of removed cells
        # However, the resulting compared matrices will all have the same number of cells
        temp_sort_ind = ref_sort_ind.copy()
        for j in range(len(temp_sort_ind)):
            reduction = np.where(remove_inds_list[inds[i]] < temp_sort_ind[j])[0].shape[0]
            temp_sort_ind[j] -= reduction
        assert len(temp_sort_ind) == len(np.unique(temp_sort_ind))
        num_indice.append(len(temp_sort_ind))
        comp_sorted_by_ref.append(corr_matrices[inds[i]][temp_sort_ind, :][:, temp_sort_ind])
    assert len(np.unique(num_indice)) == 1

    fig, ax = plt.subplots(num_rows, 3, figsize=(15, num_rows * 5))
    # Plot reference matrices
    sns.heatmap(corr_matrices[inds[0]], cmap='RdBu_r', square=True, cbar_kws={'shrink': cb_shrink_factor},
                vmin=vmin, vmax=vmax, ax=ax[0, 0])
    ax[0, 0].set_title(compare_epochs[0])
    sns.heatmap(corr_ordered_matrices[inds[0]], cmap='RdBu_r', square=True, cbar_kws={'shrink': cb_shrink_factor},
                vmin=vmin, vmax=vmax, ax=ax[0, 1])
    ax[0, 1].set_title(compare_epochs[0] + ' sorted')
    sns.heatmap(corr_ordered_by_region_matrices[inds[0]], cmap='RdBu_r', square=True, cbar_kws={'shrink': cb_shrink_factor}, 
                vmin=vmin, vmax=vmax, ax=ax[0, 2])
    ax[0, 2].set_title(compare_epochs[0] + ' sorted within region')

    # Plot comparison matrices
    for i in range(1, num_rows):
        sns.heatmap(corr_matrices[inds[i]], cmap='RdBu_r', square=True, cbar_kws={'shrink': cb_shrink_factor},
                    vmin=vmin, vmax=vmax, ax=ax[i, 0])
        ax[i, 0].set_title(compare_epochs[i])
        sns.heatmap(comp_sorted_by_ref[i - 1], cmap='RdBu_r', square=True, cbar_kws={'shrink': cb_shrink_factor},
                    vmin=vmin, vmax=vmax, ax=ax[i, 1])
        ax[i, 1].set_title(compare_epochs[i] + ' sorted by ' + compare_epochs[0])
        sns.heatmap(corr_ordered_by_region_matrices[inds[i]], cmap='RdBu_r', square=True, cbar_kws={'shrink': cb_shrink_factor},
                    vmin=vmin, vmax=vmax, ax=ax[i, 2])
        ax[i, 2].set_title(compare_epochs[i] + ' sorted within region')

    # Label axes
    num_cell = corr_matrices[inds[0]].shape[0]
    interval = 100 if num_cell>300 else 50
    xy_label_pos_ref = xy_label_pos_list[inds[0]]
    xy_labels_ref = xy_labels_list[inds[0]]
    for i in range(num_rows):
        for j in range(2):
            ax[i, j].set_xticks(range(0, num_cell, interval))
            ax[i, j].set_yticks(range(0, num_cell, interval))
            ax[i, j].set_xticklabels(range(0, num_cell, interval))
            ax[i, j].set_yticklabels(range(0, num_cell, interval))
        ax[i, 2].set_xticks(xy_label_pos_ref)
        ax[i, 2].set_xticklabels(xy_labels_ref, rotation=90)
        ax[i, 2].set_yticks(xy_label_pos_ref)
        ax[i, 2].set_yticklabels(xy_labels_ref, rotation=0)
    fig.suptitle(f'{session_name}\n{compare_epochs[0]} vs {compare_epochs[1]}')
    fig.tight_layout()

    return fig

