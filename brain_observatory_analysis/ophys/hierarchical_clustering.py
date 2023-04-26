import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.cluster.hierarchy as hierarchy
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from brain_observatory_analysis.ophys import correlation_analysis as ca
from brain_observatory_analysis.ophys import raster_plot as rp
from brain_observatory_analysis.ophys import data_formatting as df
from brain_observatory_analysis.utilities import data_utils as du


def get_task_df_list(base_dir, mouse_id, project_code=None):
    df_fn_list = glob.glob(str(base_dir / mouse_id / f'*_task_trace_df.pkl'))
    trace_df_fn_list = []
    concat_df_fn_list = []
    for df_fn in df_fn_list:
        if 'concat' in df_fn:
            concat_df_fn_list.append(df_fn)
        else:
            trace_df_fn_list.append(df_fn)
    concat_df_fn_list = np.sort(concat_df_fn_list)
    trace_df_fn_list = np.sort(trace_df_fn_list)
    
    trace_df_list = []
    for fn in trace_df_fn_list:
        trace_df = pd.read_pickle(fn)
        trace_df_list.append(trace_df)
    session_names = ['_'.join(Path(fn).stem.split('_')[:3]) for fn in trace_df_fn_list]

    if project_code is not None:
        training_types = [f'training{x}' for x in range(6)]
        project_code = str(project_code).lower()
        session_types = []
        for sn in session_names:
            sn = sn.lower()
            if 'visualbehavior' in project_code:
                if ('ophys3' in sn) or ('ophys1' in sn) or ('1images' in sn) or ('3images' in sn):
                    session_types.append('Familiar')
                elif ('ophys4' in sn) or ('4images' in sn):
                    session_types.append('Novel')
                elif ('ophys6' in sn) or ('6images' in sn):
                    session_types.append('Novel+')
                elif ('ophys2' in sn) or ('2images' in sn):
                    session_types.append('Passive_novel')
                elif ('ophys5' in sn) or ('5images' in sn):
                    session_types.append('Passive_novel')
                else:
                    session_types.append('Unknown')
            if 'learningmfish' in project_code:
                if 'ophys1' in sn:
                    session_types.append('Familiar')
                elif 'ophys4' in sn:
                    session_types.append('Novel')
                elif 'ophys6' in sn:
                    session_types.append('Extinction')
                elif 'ophys' in sn:
                    session_types.append('Other Ophys')
                elif 'training' in sn:
                    for tt in training_types:
                        if tt in sn:
                            session_types.append(tt)
                else:
                    session_types.append('Unknown')
        
    # Concatenated traces
    concat_df_list = []
    for fn in concat_df_fn_list:
        concat_df = pd.read_pickle(fn)
        concat_df_list.append(concat_df)
    
    return trace_df_list, concat_df_list, session_names, session_types


def draw_session_trace_clustering(trace_array_std, cre_line, mouse_id, session_type,
                                  method='ward', metric='euclidean', num_clusters=10):
    fig, ax = plt.subplots(3,1, figsize=(6, 12))
    Z_raster = linkage(trace_array_std, method=method, metric=metric)
    ddata_raster = dendrogram(Z_raster, ax=ax[0])
    leaves_raster = ddata_raster['leaves']

    dendrogram(Z_raster, p=num_clusters, truncate_mode='lastp', ax=ax[1]);

    cluster_raster = hierarchy.fcluster(Z_raster, num_clusters, criterion='maxclust') - 1
    num_cell = len(cluster_raster)
    ordered_cluster_raster = cluster_raster[leaves_raster]
    vmin=-2
    vmax=3
    ax[2].imshow(trace_array_std[leaves_raster,:], aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax[2].set_yticks([])
    ax[2].set_xlabel('Frame #')
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("left", size="5%", pad=0)
    # bar showing the cluster in cax
    for ci in range(0, num_clusters):
        cax.bar(0, num_cell - len(np.where(ordered_cluster_raster<ci)[0]), color='C{}'.format(ci), width=1)
        ax[2].axhline(y=len(np.where(ordered_cluster_raster<ci)[0]), color='k', linestyle='--', linewidth=0.5)
    cax.set_yticks([])
    cax.set_xticks([])
    cax.set_ylim(0, len(ordered_cluster_raster))
    # remove bounding box of cax
    cax.spines['top'].set_visible(False)
    cax.spines['right'].set_visible(False)
    cax.spines['bottom'].set_visible(False)
    cax.spines['left'].set_visible(False)
    fig.suptitle(f'{mouse_id}({cre_line}) {session_type} {method} {metric} {num_clusters} clusters')
    fig.tight_layout()
    return fig, cluster_raster, leaves_raster


def get_dendogram_and_fig(trace_df, cre_line, mouse_id, session_type):
    trace_array, remove_ind, nan_frame_ind = ca.get_trace_array_from_trace_df(trace_df, nan_frame_prop_threshold=0)
    if len(remove_ind) > 0:
        trace_array_nonan = np.delete(trace_array, remove_ind, axis=0)
    else:
        trace_array_nonan = trace_array.copy()
    if len(nan_frame_ind) > 0:
        trace_array_nonan = np.delete(trace_array_nonan, nan_frame_ind, axis=1)
    else:
        trace_array_nonan = trace_array_nonan.copy()
    keep_ind = np.setdiff1d(np.arange(len(trace_df)), remove_ind)
    keep_frame = np.setdiff1d(np.arange(len(trace_df.timepoints.values[0])), nan_frame_ind)
    # standardize trace_array
    # In each row, subtract the mean of the row
    trace_array_t = trace_array.T
    trace_array_std_t = (trace_array_t - np.nanmean(trace_array_t, axis=0)) / np.nanstd(trace_array_t, axis=0)
    trace_array_std = trace_array_std_t.T

    if len(remove_ind) > 0:
        trace_array_std_nonan = np.delete(trace_array_std, remove_ind, axis=0)
    else:
        trace_array_std_nonan = trace_array_std.copy()
    if len(nan_frame_ind) > 0:
        trace_array_std_nonan = np.delete(trace_array_std_nonan, nan_frame_ind, axis=1)
    else:
        trace_array_std_nonan = trace_array_std_nonan.copy()

    fig, cluster, leaves = draw_session_trace_clustering(trace_array_std_nonan, cre_line, mouse_id, session_type, num_clusters=10)
    return fig, cluster, leaves, trace_array_std, trace_array_std_nonan, keep_ind, keep_frame


def plot_cluster_distribution(trace_df, keep_ind, cluster, region_depth_list_template, cre_line, mouse_id, session_type):
    region_depth = trace_df.apply(lambda x: f'{x.targeted_structure}_{int(x.depth_order):02d}', axis=1)
    region_depth = region_depth.iloc[keep_ind]
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    if len(region_depth.unique()) > 2:
        rd_ind = [region_depth_list_template.index(rd) for rd in region_depth.unique()]
        region_depth_list = np.sort(region_depth.unique())[np.argsort(rd_ind)]
        
        for ci in range(0, len(np.unique(cluster))):
            cluster_ind = np.where(cluster == ci)[0]
            cluster_region_depth = region_depth.iloc[cluster_ind]
            cluster_region_depth_count = cluster_region_depth.value_counts()
            cluster_region_depth_count = cluster_region_depth_count.reindex(region_depth_list)
            cluster_region_depth_count.plot(kind='line', ax=ax, color='C{}'.format(ci), label='Cluster {}'.format(ci))
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax.set_ylabel('Number of cells')
        ax.set_xlabel('Region_depth')
        ax.set_title(f'{mouse_id}({cre_line}) {session_type} clusters distribution by region_depth')
        ax.set_xticks(np.arange(len(region_depth_list)))
        ax.set_xticklabels(region_depth_list, rotation=45, ha='right')

    return fig


def get_response_fig(cluster, concat_df_sorted_by_trace_df, keep_ind, trace_array_std,
                     cre_line, mouse_id, session_type):
    mean_concat_trace = []
    mean_cluster_trace = []
    for i in range(np.max(cluster) + 1):
        inds = np.where(cluster == i)[0]
        cluster_mean_concat = concat_df_sorted_by_trace_df.iloc[keep_ind[inds]].trace_norm.mean()
        mean_concat_trace.append(cluster_mean_concat)
        cluster_mean_trace = np.nanmean(trace_array_std[keep_ind[inds]], axis=0)
        mean_cluster_trace.append(cluster_mean_trace)
    fig, ax = plt.subplots()
    concat_time_stamps = concat_df_sorted_by_trace_df.time_stamps.values[0]

    for i, trace in enumerate(mean_concat_trace):
        ax.plot(concat_time_stamps, trace, label=f'C{i}')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., prop={'size': 8})
    ax.fill_betweenx(y=ax.get_ylim(), x1=0.8, x2=1.6, alpha=0.2, color='gray')
    ax.set_xticks([0.75 / 2, 0.75 * 3 / 2 + 0.05, 0.75 * 5 / 2 + 0.1])
    ax.set_xticklabels(['Changes', 'Images', 'Omissions'])
    ax.set_ylabel('DF/F response')
    ax.set_title(f'{mouse_id}({cre_line}) {session_type} - Responses of each cluster')

    return fig, mean_concat_trace, mean_cluster_trace


def get_cluster_correlation_fig(mean_concat_trace, mean_cluster_trace, cre_line, mouse_id, session_type):
    cc_mean_concat_trace = np.corrcoef(np.asarray(mean_concat_trace))
    cc_mean_cluster_trace = np.corrcoef(np.asarray(mean_cluster_trace))
    # select upper triangle of the correlation matrix
    cc_mean_concat_trace_values = cc_mean_concat_trace[np.triu_indices_from(cc_mean_concat_trace, k=1)]
    cc_mean_cluster_trace_values = cc_mean_cluster_trace[np.triu_indices_from(cc_mean_cluster_trace, k=1)]
    vmin = np.min([cc_mean_concat_trace_values, cc_mean_cluster_trace_values])
    vmax = np.max([cc_mean_concat_trace_values, cc_mean_cluster_trace_values])

    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].imshow(cc_mean_cluster_trace, vmin=vmin, vmax=vmax)
    ax[0].set_title('Between mean traces', fontsize=10)
    ax[0].set_ylabel('Cluster #')
    ax[0].set_xlabel('Cluster #')
    ax[1].imshow(cc_mean_concat_trace, vmin=vmin, vmax=vmax)
    ax[1].set_title('Between mean responses', fontsize=10)
    ax[1].set_xlabel('Cluster #')
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    # cbar = plt.colorbar(ax[1].images[0], cax=cax)
    fig.suptitle(f'{mouse_id}({cre_line}) {session_type} cluster correlations')
    fig.tight_layout()

    return fig


def get_behavior(trace_df, cache):
    oeid = trace_df.oeid.values[0]
    exp = cache.get_behavior_ophys_experiment(oeid)
    # trace_timepoints = trace_df.timepoints.values[0][keep_frame]
    trace_timepoints = trace_df.timepoints.values[0]
    lickrate_interp = rp._get_interpolated_lickrate(exp, trace_timepoints)
    running_interp = rp._get_interpolated_running(exp, trace_timepoints)
    pupil_diameter, eye_timestamps = rp._get_pupil_diameter(exp)
    if pupil_diameter is not None:
        pupil_interp = du.get_interpolated_time_series(eye_timestamps, pupil_diameter, trace_timepoints)
    else:
        pupil_interp = None
    rewards_df = exp.rewards.query('autorewarded==False')
    reward_timestamps = rewards_df.timestamps
    reward_timestamps = reward_timestamps[reward_timestamps <= trace_timepoints[-1]]
    rewards = np.zeros(len(trace_timepoints))
    for timestamp in reward_timestamps:
        # Find the nearest timepoint in timepoints to the reward timestamp
        nearest_ind = np.argmin(np.abs(trace_timepoints - timestamp))
        rewards[nearest_ind] += 1
    return trace_timepoints, lickrate_interp, running_interp, pupil_interp, rewards


def get_clutstered_trace(cluster, leaves, trace_array_std_nonan, trace_timepoints,
                         cre_line, mouse_id, session_type,
                         num_clusters=10, vmin=-2, vmax=3, sub_title_fontsize=10):
    fig, ax = plt.subplots(figsize=(12, 8))
    num_cell = len(cluster)
    ordered_cluster_raster = cluster[leaves]
    ax.imshow(trace_array_std_nonan[leaves, :], aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax.set_yticks([])
    ax.set_xlabel('Frame #')
    ax.set_title(f'{mouse_id}({cre_line}) {session_type}')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("left", size="2%", pad=0)
    # bar showing the cluster in cax
    for ci in range(0, num_clusters):
        cax.bar(0, num_cell - len(np.where(ordered_cluster_raster < ci)[0]), color='C{}'.format(ci), width=1)
        ax.axhline(y=len(np.where(ordered_cluster_raster < ci)[0]), color='k', linestyle='--', linewidth=0.5)
    cax.set_yticks([])
    cax.set_xticks([])
    cax.set_ylim(0, len(ordered_cluster_raster))
    # remove bounding box of cax
    cax.spines['top'].set_visible(False)
    cax.spines['right'].set_visible(False)
    cax.spines['bottom'].set_visible(False)
    cax.spines['left'].set_visible(False)
    return fig


def get_clustered_trace_with_behavior(cluster, leaves, trace_array_std_nonan, trace_timepoints,
                                      lickrate_interp, running_interp, pupil_interp, rewards,
                                      cre_line, mouse_id, session_type,
                                      num_clusters=10, vmin=-2, vmax=3, sub_title_fontsize=10):
    fig, ax = plt.subplots(figsize=(12, 8))
    num_cell = len(cluster)
    ordered_cluster_raster = cluster[leaves]
    ax.imshow(trace_array_std_nonan[leaves, :], aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax.set_yticks([])
    ax.set_xlabel('Frame #')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("left", size="2%", pad=0)
    # bar showing the cluster in cax
    for ci in range(0, num_clusters):
        cax.bar(0, num_cell - len(np.where(ordered_cluster_raster < ci)[0]), color='C{}'.format(ci), width=1)
        ax.axhline(y=len(np.where(ordered_cluster_raster < ci)[0]), color='k', linestyle='--', linewidth=0.5)
    cax.set_yticks([])
    cax.set_xticks([])
    cax.set_ylim(0, len(ordered_cluster_raster))
    # remove bounding box of cax
    cax.spines['top'].set_visible(False)
    cax.spines['right'].set_visible(False)
    cax.spines['bottom'].set_visible(False)
    cax.spines['left'].set_visible(False)

    # Add mean z-score
    zax = divider.append_axes('bottom', size='20%', pad=0.5)
    zax.plot(trace_timepoints, np.mean(trace_array_std_nonan, axis=0), color='C2')
    zax.set_xlim(trace_timepoints[0], trace_timepoints[-1])
    zax.set_xticks(trace_timepoints[ax.get_xticks()[1:-1].astype(int)])
    zax.set_xticklabels([])
    # zax.plot(np.mean(trace_array_std_nonan,axis=0), color='C2')
    # zax.set_xlim(0, trace_array_std_nonan.shape[1])
    # zax.set_xticks([])
    zax.set_title('z-scored dF/F', loc='left', fontsize=sub_title_fontsize, color='C2')

    # Add running speed
    rax = divider.append_axes('bottom', size='20%', pad=0.25, sharex=zax)
    rax.plot(trace_timepoints, running_interp, color='C3')
    # rax.plot(running_interp, color='C3')
    rax.set_title('Running speed (cm/s)', loc='left', fontsize=sub_title_fontsize, color='C3')

    # Add pupil diameter
    pax = divider.append_axes('bottom', size='20%', pad=0.25, sharex=zax)
    if pupil_interp is not None:
        pax.plot(trace_timepoints, pupil_interp, color='black')
        # pax.plot(pupil_interp, color='black')
    pax.set_title('Pupil diameter (AU)', loc='left', fontsize=sub_title_fontsize, color='black')

    # Add lick rates
    lax = divider.append_axes('bottom', size='20%', pad=0.25, sharex=zax)
    lax.plot(trace_timepoints, lickrate_interp, linewidth=1, color='C6')
    # lax.plot(lickrate_interp, linewidth=1, color='C6')
    lax.set_title('Lick rate (Hz)', loc='left', fontsize=sub_title_fontsize, color='C6')

    # Add rewards
    rax = divider.append_axes('bottom', size='10%', pad=0.25)
    rax.plot(trace_timepoints, rewards, linewidth=1, color='C9')
    rax.set_xticks(trace_timepoints[ax.get_xticks()[1:-1].astype(int)])
    rax.set_xticklabels((trace_timepoints[ax.get_xticks()[1:-1].astype(int)] - trace_timepoints[0]).astype(int))
    rax.set_xlim(trace_timepoints[0], trace_timepoints[-1])
    rax.set_xlabel('Time (s)')
    # rax = divider.append_axes('bottom', size='10%', pad=0.25, sharex=zax)
    # rax.plot(rewards, linewidth=1, color='C9')
    # rax.set_xlabel('Frame #')

    rax.set_title('Rewards', loc='left', fontsize=sub_title_fontsize, color='C9')

    # Add colorbar
    cax = divider.append_axes('right', size='2%', pad=0.05)
    cbar = fig.colorbar(ax.images[0], cax=cax)
    cbar.ax.set_ylabel('z-scored dF/F')

    fig.suptitle(f'{mouse_id}({cre_line}) {session_type}')
    fig.tight_layout()

    return fig


def calculate_correlation_between_cluster_and_behavior(trace_array_std_nonan, num_clusters, trace_timepoints,
                                                       mean_cluster_trace, running_interp, pupil_interp, lickrate_interp,
                                                       sliding_window_minute=5, interval_min=0.5, imaging_freq=10.7):
    sliding_window = int(sliding_window_minute * 60 * imaging_freq)  # in frames (about 5 min)
    interval = int(interval_min * 60 * imaging_freq)  # in frames (about 30 sec)
    mean_trace = np.mean(trace_array_std_nonan, axis=0)
    corr_with_running = np.zeros((num_clusters, (len(trace_timepoints) - sliding_window) // interval))
    corr_with_pupil = np.zeros((num_clusters, (len(trace_timepoints) - sliding_window) // interval))
    corr_with_lick = np.zeros((num_clusters, (len(trace_timepoints) - sliding_window) // interval))
    corr_running_pupil = np.zeros((len(trace_timepoints) - sliding_window) // interval)
    corr_running_lick = np.zeros((len(trace_timepoints) - sliding_window) // interval)
    corr_pupil_lick = np.zeros((len(trace_timepoints) - sliding_window) // interval)
    corr_mean_running = np.zeros((len(trace_timepoints) - sliding_window) // interval)
    corr_mean_pupil = np.zeros((len(trace_timepoints) - sliding_window) // interval)
    corr_mean_lick = np.zeros((len(trace_timepoints) - sliding_window) // interval)
    for ci in range(0, num_clusters):
        for ti in range(0, (len(trace_timepoints) - sliding_window) // interval):
            start_ind = ti * interval
            end_ind = start_ind + sliding_window
            corr_with_running[ci, ti] = np.corrcoef(mean_cluster_trace[ci][start_ind:end_ind], running_interp[start_ind:end_ind])[0, 1]
            if pupil_interp is not None:
                corr_with_pupil[ci, ti] = np.corrcoef(mean_cluster_trace[ci][start_ind:end_ind], pupil_interp[start_ind:end_ind])[0, 1]
            corr_with_lick[ci, ti] = np.corrcoef(mean_cluster_trace[ci][start_ind:end_ind], lickrate_interp[start_ind:end_ind])[0, 1]
    for ti in range(0, (len(trace_timepoints) - sliding_window) // interval):
        start_ind = ti * interval
        end_ind = start_ind + sliding_window
        if pupil_interp is not None:
            corr_running_pupil[ti] = np.corrcoef(running_interp[start_ind:end_ind], pupil_interp[start_ind:end_ind])[0, 1]
        corr_running_lick[ti] = np.corrcoef(running_interp[start_ind:end_ind], lickrate_interp[start_ind:end_ind])[0, 1]
        if pupil_interp is not None:
            corr_pupil_lick[ti] = np.corrcoef(pupil_interp[start_ind:end_ind], lickrate_interp[start_ind:end_ind])[0, 1]
        corr_mean_running[ti] = np.corrcoef(mean_trace[start_ind:end_ind], running_interp[start_ind:end_ind])[0, 1]
        if pupil_interp is not None:
            corr_mean_pupil[ti] = np.corrcoef(mean_trace[start_ind:end_ind], pupil_interp[start_ind:end_ind])[0, 1]
        corr_mean_lick[ti] = np.corrcoef(mean_trace[start_ind:end_ind], lickrate_interp[start_ind:end_ind])[0, 1]
    
    corr_timepoints = trace_timepoints[sliding_window // 2:-sliding_window // 2 - interval:interval]
    return corr_timepoints, corr_with_running, corr_with_pupil, corr_with_lick, corr_running_pupil, corr_running_lick, corr_pupil_lick, corr_mean_running, corr_mean_pupil, corr_mean_lick


def plot_all_correlations(corr_timepoints, num_clusters, corr_with_running, corr_with_pupil, corr_with_lick,
                          corr_running_pupil, corr_running_lick, corr_pupil_lick, pupil_interp,
                          corr_mean_running, corr_mean_pupil, corr_mean_lick,
                          cre_line, mouse_id, session_type):
    fig, ax = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    for ci in range(num_clusters):
        ax[0].plot(corr_timepoints, corr_with_running[ci], label=f'cluster {ci}')
        ax[1].plot(corr_timepoints, corr_with_pupil[ci])
        ax[2].plot(corr_timepoints, corr_with_lick[ci])
    # Set the size of the legend box

    ax[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., prop={'size': 8})

    ax[0].set_title('Correlation to running')
    ax[0].set_ylabel('Correlation')
    ax[1].set_title('Correlation to pupil')
    ax[1].set_ylabel('Correlation')
    ax[2].set_title('Correlation to lickrate')
    ax[2].set_ylabel('Correlation')
    # ax[2].set_xlabel('Time (s)')
    ax[3].plot(corr_timepoints, corr_mean_running, label='mean trace vs running', color='C3')
    ax[3].plot(corr_timepoints, corr_mean_lick, label='mean trace vs lick', color='C6')
    if pupil_interp is not None:
        ax[3].plot(corr_timepoints, corr_mean_pupil, label='mean trace vs pupil', color='k')
    ax[3].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    ax[3].set_title('Correlation to mean trace')
    ax[3].set_ylabel('Correlation')
    if pupil_interp is not None:
        ax[4].plot(corr_timepoints, corr_running_pupil, label='running vs pupil')
    ax[4].plot(corr_timepoints, corr_running_lick, label='running vs lick')
    if pupil_interp is not None:
        ax[4].plot(corr_timepoints, corr_pupil_lick, label='pupil vs lick')
    ax[4].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    ax[4].set_xlabel('Time (s)')
    ax[4].set_ylabel('Correlation')
    ax[4].set_title('Correlation between running, pupil, and lickrate')

    fig.suptitle(f'{mouse_id}({cre_line}) {session_type}')
    fig.tight_layout()
    return fig


def plot_pca_with_corr(pca_coord, pca_axes, corr_with_behav, ax, colorbar_label):
    im = ax.scatter(pca_coord[:, pca_axes[0]], pca_coord[:, pca_axes[1]], c=corr_with_behav, cmap='coolwarm', 
               vmin=np.nanmin(corr_with_behav), vmax=np.nanmax(corr_with_behav))
    ax.set_xlabel(f'PC{pca_axes[0]+1}')
    ax.set_ylabel(f'PC{pca_axes[1]+1}')
    ax.set_aspect('equal')
    ax.set_title(f'{colorbar_label}')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label=f'{colorbar_label}')


def plot_pca_with_corr_behavior(pca_coord, pca_axes_ind,
                       mean_corrcoef, corr_with_running, corr_with_pupil, corr_with_lick,
                       cre_line, mouse_id, session_type):
    fig, ax = plt.subplots(2,2, figsize=(12,10))
    plot_pca_with_corr(pca_coord, pca_axes_ind, mean_corrcoef, ax[0,0], 'Mean correlation coefficient')
    plot_pca_with_corr(pca_coord, pca_axes_ind, corr_with_running, ax[0,1], 'Correlation with running')
    if corr_with_pupil is not None:
        plot_pca_with_corr(pca_coord, pca_axes_ind, corr_with_pupil, ax[1,0], 'Correlation with pupil')
    plot_pca_with_corr(pca_coord, pca_axes_ind, corr_with_lick, ax[1,1], 'Correlation with lickrate')
    fig.suptitle(f'{mouse_id}({cre_line}) {session_type} PCA')
    fig.tight_layout()
    return fig


def plot_varexp(trace_array_std_nonan, pca, cre_line, mouse_id, session_type):
    num_pc_limit = min(300, trace_array_std_nonan.shape[0] - trace_array_std_nonan.shape[0]//10)
    inset_num_pc = min(30, trace_array_std_nonan.shape[0]//10)
    varexp_pc = pca.explained_variance_ratio_
    pcs = np.arange(1, len(varexp_pc)+1)
    fig, ax = plt.subplots(1,3, figsize=(10,3))
    ax[0].plot(pcs, np.cumsum(varexp_pc))
    ax[0].set_xlabel('# PCs')
    ax[0].set_ylabel('Cumulative variance explained')
    ax[1].plot(pcs[:inset_num_pc], np.cumsum(varexp_pc)[:inset_num_pc])
    ax[1].set_xlabel('# PCs')
    ax[1].set_ylabel('Cumulative variance explained')
    ax[2].loglog(pcs[:num_pc_limit], varexp_pc[:num_pc_limit])
    ax[2].set_aspect('equal')
    ax[2].set_ybound(upper=1/10)
    ax[2].set_xlabel('# PCs')
    ax[2].set_ylabel('Variance explained')
    fig.suptitle(f'{mouse_id}({cre_line}) {session_type} PCA Variance Explained')
    fig.tight_layout()
    return fig


def plot_pca_loading_corr_with_behavior(pca, running_interp, pupil_interp, lickrate_interp,
                                        cre_line, mouse_id, session_type, num_pc=10):
    # Calculate correlation with running, pupil, lickrate, for each PC loading
    corr_pc_vs_running = np.zeros(num_pc)
    corr_pc_vs_pupil = np.zeros(num_pc)
    corr_pc_vs_lickrate = np.zeros(num_pc)
    
    for i in range(num_pc):
        corr_pc_vs_running[i] = np.corrcoef(pca.components_[i, :], running_interp)[0, 1]
        if pupil_interp is not None:
            corr_pc_vs_pupil[i] = np.corrcoef(pca.components_[i, :], pupil_interp)[0, 1]
        corr_pc_vs_lickrate[i] = np.corrcoef(pca.components_[i, :], lickrate_interp)[0, 1]
        
    fig, ax = plt.subplots(1,2,figsize=(15,5))
    ax[0].plot(range(1,11), corr_pc_vs_running, label='running')
    ax[0].plot(range(1,11), corr_pc_vs_pupil, label='pupil')
    ax[0].plot(range(1,11), corr_pc_vs_lickrate, label='lickrate')
    ax[0].set_xlabel('PC #')
    ax[0].set_ylabel('Correlation')

    ax[1].plot(range(1,11), np.abs(corr_pc_vs_running), label='running')
    ax[1].plot(range(1,11), np.abs(corr_pc_vs_pupil), label='pupil')
    ax[1].plot(range(1,11), np.abs(corr_pc_vs_lickrate), label='lickrate')
    ax[1].legend()
    ax[1].set_xlabel('PC #')
    ax[1].set_ylabel('|Correlation|')
    fig.suptitle(f'{mouse_id}({cre_line}) {session_type} PCA loadings correlation with running, pupil, lickrate')
    fig.tight_layout()
    return fig
