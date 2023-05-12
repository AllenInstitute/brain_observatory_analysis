# core
import warnings
from pathlib import Path
import pickle as pkl
import os
import argparse
import multiprocessing as mp
from typing import Union

# 3rd party standard
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 3rd party plus
import tifffile
import skimage

# brain observatory
from allensdk.brain_observatory.behavior.behavior_ophys_experiment import \
    BehaviorOphysExperiment
from brain_observatory_qc.data_access.behavior_ophys_experiment_dev import \
    BehaviorOphysExperimentDev
from brain_observatory_analysis.ophys.experiment_group import ExperimentGroup
from brain_observatory_analysis.ophys.roi_filter import filter_rois_with_with_nrsac_classifier
from brain_observatory_analysis.ophys.experiment_loading import get_ophys_expt, start_lamf_analysis
from brain_observatory_qc.pipeline_dev.scripts.depth_estimation_module import image_normalization_uint16

# optional import
try:
    from adjust_text import adjust_text
except ImportError:
    adjust_text = None

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# args
parser = argparse.ArgumentParser()
parser.add_argument("--oeid", type=int, required=True, help="ophys experiment id")
parser.add_argument("--cz_stack_path", type=str, required=True, help="Path to cortical zstack tiff file")

# not necessary?
# from brain_observatory_analysis.ophys.cells import container_cells_table_plus_metadata
# from brain_observatory_analysis.ophys.cells import get_cell_specimens_matched_across_sessions
# import brain_observatory_qc.projects.LearningmFISHTask1A as constants
# from scipy.signal import find_peaks


def matched_cz_to_fov_plane():
    """relace with import to jinhos functions"""
    pass


def register_fov_to_cz_rois(expt: Union[BehaviorOphysExperiment, BehaviorOphysExperimentDev],
                            cz_stack: np.ndarray,
                            cz_stack_masks: np.ndarray,
                            fov_cz_match_df: pd.DataFrame,
                            gene_table: pd.DataFrame,
                            deoverlap=False,
                            iou_tresh=0.1):
    """Get 2P FOV rois that are matched to the corresponding plane from cortical zstack. Adds additional info
    about whether the matching ROIs are coregistred in the light sheet volume.

    Parameters
    ----------
    expt: BehaviorOphysExperiment
        2P experiment object
    cz_stack: np.ndarray
        3D array of cortical zstack images
    cz_stack_masks: np.ndarray
        3D array of cortical zstack masks. Each mask is a unique integer value.
    fov_cz_match_df: pd.DataFrame
        DataFrame with cortical zstack plane matches for each 2P experiment (jinho's output).
        Requires columns:
            + "match_plane_ind": which is the best matching cortical plane for each 2P FOV
            + "ophys_experiment_id": which is the 2P FOV experiment id
    deoverlap: bool
        Whether to deoverlap the 2P FOV ROIs
        (Note: no longer needed, this was used to remove overlapping ROIs in the 2P FOV, which is now done in the
        pipeline)
    gene_table: pd.DataFrame
        DataFrame with gene info for each cell
        Fairly extensive table with many columns (TODO: update key columns)
    iou_tresh: float
        IOU threshold for filtering 2P FOV ROIs that are matched to the corresponding plane from cortical zstack

    Returns
    -------
    expt_rois_cz_filt: pd.DataFrame
        DataFrame with 2P FOV ROIs that are matched to the corresponding plane from cortical zstack. Adds additional info
        about whether the matching ROIs are coregistred in the light sheet volume. 
        Key columns:
            + "roi_mask_shifted": 2P FOV ROI mask shifted to the cortical zstack plane
    """
    try:
        oeid = expt.ophys_experiment_id
        print(f"Processing: {oeid}")
        expt_fov_img = expt.average_projection.data
        match_plane_ind = fov_cz_match_df[fov_cz_match_df.ophys_experiment_id ==
                                          oeid].matched_plane_index.values[0]
        print(f"Best cortical stack match plane: {match_plane_ind}")

        if deoverlap:
            expt_rois_df = deoverlap_expt_rois(expt)
        else:
            expt_rois_df = expt.roi_masks

        cz_stack_match_plane_expt = cz_stack[match_plane_ind, :, :]
        cz_stack_match_masks_expt = cz_stack_masks[match_plane_ind, :, :]

        expt_rois_df_filt = register_fov_and_matched_cz_stack_plane(
            expt_fov_img, cz_stack_match_plane_expt, expt_rois_df)

        explode_cz_stack_masks_expt, explode_cz_stack_ids_expt = explode_masks(
            cz_stack_match_masks_expt)

        # only keep rois that have a match in the cortical zstack
        expt_rois_cz_filt = filter_fov_masks_by_cz_stack_overlap_iou(fov_mask_df=expt_rois_df_filt,
                                                                     mask_key="roi_mask_shifted",
                                                                     stack_masks=explode_cz_stack_masks_expt,
                                                                     stack_ids=explode_cz_stack_ids_expt,
                                                                     iou_thresh=iou_tresh)
        if gene_table is not None:
            right_merge_key = "2p_mask_id" if "2p_mask_id" in gene_table.columns else "cz_stack_id"
            expt_rois_cz_filt = expt_rois_cz_filt.merge(
                gene_table, left_on="cz_stack_id", right_on=right_merge_key, how="left")

        expt_rois_cz_filt["ophys_experiment_id"] = oeid
        expt_rois_cz_filt["cz_match_plane_index"] = match_plane_ind

        # don't need to keep OG masks and shifted, extra space
        expt_rois_cz_filt = expt_rois_cz_filt.drop(columns=["roi_mask"])

    except Exception as e:
        print(f"ERROR: failed to processes {oeid}")
        print(e)
        return None

    return expt_rois_cz_filt


def register_fov_to_cz_rois_for_expt_grp(expt_grp,
                                         cz_stack,
                                         cz_stack_masks,
                                         fov_cz_match_df,
                                         ls_cz_match_df,
                                         roi_filter=False,
                                         gene_table=None):
    """See register_fov_to_cz_rois() for details. Parallelize wrapper"""

    n_cores = mp.cpu_count()
    pool = mp.Pool(n_cores)
    args = []

    for name, expt in expt_grp.experiments.items():
        if roi_filter:
            expt = filter_rois_with_with_nrsac_classifier(expt)
        args.append((expt, cz_stack, cz_stack_masks,
                    fov_cz_match_df, ls_cz_match_df, gene_table))
    
    results = pool.starmap(register_fov_to_cz_rois, args)
    pool.close()

    all_cz_filt_dfs = pd.concat(results, axis=0)

    return all_cz_filt_dfs


def filter_fov_masks_by_cz_stack_overlap_iou(fov_mask_df: pd.DataFrame,
                                             stack_masks: list,
                                             stack_ids: list,
                                             mask_key: str = "roi_mask_shifted",
                                             iou_thresh: float = .1):

    """
    Filter 2P FOV ROIs by overlap with cortical zstack plane ROIs.

    Parameters
    ----------
    fov_mask_df: pd.DataFrame
        DataFrame with 2P FOV ROIs
        Key columns:
            + "roi_mask_shifted": 2P FOV ROI mask shifted to the cortical zstack plane
    mask_key: str
        Name of column in fov_mask_df that contains the mask to use for overlap calculation
    stack_masks: list
        List of cortical zstack plane masks
    stack_ids: list
        List of cortical zstack plane mask ids
    iou_thresh: float
        IOU threshold for filtering rois

    Returns
    -------
    fov_mask_df: pd.DataFrame
        DataFrame with 2P FOV ROIs that are matched to the corresponding plane from cortical zstack.
    """

    # stack ids and mask should be same length
    assert len(stack_ids) == len(stack_masks)

    fov_mask_df = fov_mask_df.copy()

    # if index nan, reset, no csids, but no probalemif fake them for thos
    if fov_mask_df.index.isna().all():
        fov_mask_df.reset_index(inplace=True)

    overlap_zstack_csids = []
    final_ious = []
    matched_stack_ids = []
    matched_stack_masks = []
    for i in range(len(stack_masks)):
        ious = []
        for j in range(len(fov_mask_df)):

            iou = get_iou(stack_masks[i], fov_mask_df.iloc[j][mask_key])
            ious.append(iou)

        # retain the max iou for each mask
        max_iou = np.max(ious)
        max_iou_index = np.argmax(ious)

        # if above 0, then save cell_specimen_id of expt2_masks[max_iou_index]
        if max_iou > iou_thresh:
            csid = fov_mask_df.iloc[max_iou_index].name
            matched_stack_ids.append(stack_ids[i])
            matched_stack_masks.append(stack_masks[i])
            overlap_zstack_csids.append(csid)
            final_ious.append(max_iou)

    print("Number of masks in FOV plane: ", len(fov_mask_df))
    print("Number of masks in CZ stack plane: ", len(overlap_zstack_csids))
    # filter expt2_masks by overlap_zstack_csids, keep order of overlap_zstack_csids
    expt_masks_cz_filt = fov_mask_df.loc[overlap_zstack_csids]
    # expt_masks_cz_filt = fov_mask_df[fov_mask_df.index.isin(overlap_zstack_csids)]

    expt_masks_cz_filt["max_iou"] = final_ious
    expt_masks_cz_filt["cz_stack_id"] = matched_stack_ids
    expt_masks_cz_filt["cz_stack_mask"] = matched_stack_masks

    return expt_masks_cz_filt


def explode_masks(mask_plane):
    """Explode masks from 2d plane into list of 2d arrays, each array is a mask.

    Parameters
    ----------
    mask_plane : np.ndarray
        2D array of masks, where each mask is a unique integer

    Returns
    -------
    explode_masks : list
        list of 2D arrays, where each array is a mask
    unique_masks_ids : list
        list of unique mask ids
    """
    unique_masks = np.unique(mask_plane)

    # drop 0
    unique_masks_ids = unique_masks[1:]

    explode_masks = []
    for mask_id in unique_masks_ids:
        mask = mask_plane == mask_id
        # convert mask to all 0s and 1s
        mask = mask.astype(int)
        explode_masks.append(mask)

    return explode_masks, unique_masks_ids





# def blank_image(img: np.ndarray = None, ax = None):
#     if not img:
#         img = np.zeros((512, 512))
#     else:
#         img = np.zeros_like(img)
#         # make img light gray
#         img += 1050
#     if not ax:
#         fig, ax = plt.subplots(figsize=(10, 10))
#     ax.imshow(img, cmap='gray', vmin=0, vmax=255)
#     return ax

def blank_image(img: np.ndarray = None, ax=None):
    if not ax:
        fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(np.zeros((512, 512)) + 190, cmap='gray', vmin=0, vmax=255)
    return ax


def add_cols_to_rois_df(df):

    df = df.copy()

    # add column last two digits of cell_specimen_id
    df['csid_last_digits'] = df.index % 100

    # add column x, and y of upper left corner of roi_mask
    df['x'] = [np.where(roi_mask)[1].min() for roi_mask in df.roi_mask]
    df['y'] = [np.where(roi_mask)[0].min() for roi_mask in df.roi_mask]

    # add column width and height of roi_mask
    df['width'] = [np.where(roi_mask)[1].max(
    ) - np.where(roi_mask)[1].min() for roi_mask in df.roi_mask]
    df['height'] = [np.where(roi_mask)[0].max(
    ) - np.where(roi_mask)[0].min() for roi_mask in df.roi_mask]

    # add column for area of roi_mask
    df['area'] = [roi_mask.sum() for roi_mask in df.roi_mask]

    # calculate the center of the roi_mask
    df['center_x'] = df.x + df.width / 2
    df['center_y'] = df.y + df.height / 2

    # as int
    df['x'] = df.x.astype(int)
    df['y'] = df.y.astype(int)
    df['center_x'] = df.center_x.astype(int)
    df['center_y'] = df.center_y.astype(int)

    return df


def pairwise_overlap_matrix(rois_df):
    # get pairwise overlap % for all rois

    rois = rois_df.roi_mask.values
    pairwise_overlap = np.zeros((len(rois), len(rois)))
    for i, roi1 in enumerate(rois):
        for j, roi2 in enumerate(rois):
            if i == j:
                continue
            # overlap, normalize to area of smallest roi
            overlap = np.sum(roi1 & roi2) / \
                min(rois_df.iloc[i].area, rois_df.iloc[i].area)
            pairwise_overlap[i, j] = overlap

    return pairwise_overlap


def retain_largest_overlap_rois(rois_df, pairwise_overlap):

    # get csid pairs that have overlap > 0.1, only look at upper triangle
    csid_pairs = np.argwhere(pairwise_overlap > 0.1)
    csid_pairs = csid_pairs[csid_pairs[:, 0] < csid_pairs[:, 1]]

    rois_df_large = []
    cell_specimen_id_smalls = []
    area_smalls = []
    # for each pair, iterate through rois_df, get csid of roi with largest area
    for csid_pair in csid_pairs:
        area_0 = rois_df.iloc[csid_pair[0]].area
        area_1 = rois_df.iloc[csid_pair[1]].area
        if area_0 > area_1:
            rois_df_large.append(rois_df.iloc[csid_pair[0]])
            cell_specimen_id_smalls.append(rois_df.iloc[csid_pair[1]].name)
            area_smalls.append(rois_df.iloc[csid_pair[1]].area)
        else:
            rois_df_large.append(rois_df.iloc[csid_pair[1]])
            cell_specimen_id_smalls.append(rois_df.iloc[csid_pair[0]].name)
            area_smalls.append(rois_df.iloc[csid_pair[0]].area)

    # make df of large rois, csid of small rois, and area of small rois
    rois_df_large = pd.DataFrame(rois_df_large)
    # index name
    rois_df_large.index.name = 'cell_specimen_id'
    rois_df_large['cell_specimen_id_small'] = cell_specimen_id_smalls
    rois_df_large['area_small'] = area_smalls

    return rois_df_large


def image_clahe(img):
    return image_normalization_uint16(skimage.exposure.equalize_adapthist(img.astype(np.uint16)))


def plot_roi_df_contours_colored(df, color: Union[str, list] = 'random', ax=None):

    if ax is None:
        fig, ax = plt.subplots()

    assert 'roi_mask' in df.columns, "roi_mask column not found in df"

    for csid, row in df.iterrows():
        mask = row.roi_mask

        contours = skimage.measure.find_contours(mask, 0.5)

        # strange with this check
        # if color == 'random':
        # generate random color from 'tab10' colormap
        color = plt.cm.gist_rainbow(np.random.rand(1))

        for contour in contours:

            ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color=color)

    return ax


def deoverlap_expt_rois(expt):
    """Use mjd functions to deoverlap an experiment ROIS


    """

    rois_df = expt.roi_masks
    rois_df = add_cols_to_rois_df(rois_df)
    pairwise_overlap = pairwise_overlap_matrix(rois_df)
    rois_df_large = retain_largest_overlap_rois(rois_df, pairwise_overlap)
    # filter out cell_specimen_ids_small from masks
    csids_overlap_small = rois_df_large.cell_specimen_id_small.values
    rois_df_filt = rois_df.loc[~rois_df.index.isin(csids_overlap_small)]
    return rois_df_filt


def register_fov_and_matched_cz_stack_plane(expt_fov_img, cz_match_plane_img, expt_fov_masks):
    """Register FOV plane to matched CZ stack plane

    Parameters
    ----------
    expt_fov_img : np.ndarray
        2D array of FOV plane image, usually average projection
    cz_match_plane_img : np.ndarray
        2D array of CZ stack plane image, matched to the FOV plane
    expt_fov_masks : pd.DataFrame
        dataframe of masks for the FOV plane

    Returns
    -------
    expt_fov_masks : pd.DataFrame
        dataframe of masks for the FOV plane, translated to match the CZ stack plane
    """
    import skimage.registration as reg
    import skimage.transform

    expt_fov_masks = expt_fov_masks.copy()

    max_cc = reg.phase_cross_correlation(cz_match_plane_img, expt_fov_img)[0]
    print(f"max cc: {max_cc}")
    # sign flip, and reverse, was necessary for some reason
    max_cc = max_cc[::-1] * -1

    # IMAGE warp: translate expt2_fov_plane to match cz_stack_match_plane_expt2
    # expt2_fov_plane_translated = skimage.transform.warp(expt2_fov_plane, skimage.transform.AffineTransform(translation=max_cc))
    # plt.imshow(expt2_fov_plane_translated)

    # apply translation to expt2_masks
    shifted_masks = []
    for name, row in expt_fov_masks.iterrows():
        mask = skimage.transform.warp(
            row.roi_mask, skimage.transform.AffineTransform(translation=max_cc))
        shifted_masks.append(mask)

    expt_fov_masks["roi_mask_shifted"] = shifted_masks
    # make max_cc into list of lists length df

    expt_fov_masks['xy_translation'] = [max_cc] * len(expt_fov_masks)

    return expt_fov_masks


def get_iou(mask1, mask2):
    iou = np.sum(np.logical_and(mask1, mask2)) / \
        np.sum(np.logical_or(mask1, mask2))
    return iou





####################################################################################################
# Loading Data
####################################################################################################

def load_stacks_and_matches_for_copper():
    """Load CZ stacks and FOV to CZ matches for copper"""
    # Pre NRSAC
    # # hand labeled for old segmentations
    # gene_table_path =  "/home/matt.davis/code/mjd_dev/scripts/gene_table/copper_genes_r1.txt"
    # gene_table = pd.read_csv(gene_table_path, sep='\t')

    # # old segmentation match dict
    # pf = "//allen/programs/mindscope/workgroups/learning/pipeline_validation/coregistration/copper_r1_2p_ls_matched_labels.pkl"
    # with open(pf, 'rb') as f:
    #     ls_cz_match_dict = pkl.load(f)

    # jinhos fov to cz matches
    coreg_dir = Path(
        r'\\allen\programs\mindscope\workgroups\learning\pipeline_validation\coregistration'.replace('\\', '/'))
    save_fn = 'copperV1_exp_cortical_zstack_match.pkl'
    with open(coreg_dir / save_fn, 'rb') as f:
        fov_cz_match_df = pkl.load(f)

    # add row to fov_cz_match_df, oeid = 1193675218, matched_plane_index = 33
    fov_cz_match_df = fov_cz_match_df.append(
        {'ophys_experiment_id': 1193675218, 'matched_plane_index': 33}, ignore_index=True)
    fov_cz_match_df = fov_cz_match_df.append(
        {'ophys_experiment_id': 1193675220, 'matched_plane_index': 57}, ignore_index=True)

    # CZ stack data
    cz_path = "/allen/programs/mindscope/workgroups/omfish/coreg/z_stacks/qcathon/629294_1188097724_cortical_z_stack0/629294_1188097724_cortical_z_stack0/629294_1188097724_cortical_z_stack0.tif"
    cz_stack = tifffile.imread(cz_path)

    # CZ stack masks
    copper_v1_stack_loc = Path(
        '/allen/programs/mindscope/workgroups/omfish/coreg/z_stacks/qcathon/629294_1188097724_cortical_z_stack0/629294_1188097724_cortical_z_stack0_stardist_2d_rays_128_epochs_5000')
    copper_v1_stack = zutil.ImageStack().load(copper_v1_stack_loc)
    # mask_stack = zutil.ImageStack().load(stardist_label_path).to_numpy()
    cz_stack_masks = copper_v1_stack.img

    cfr1 = "/allen/programs/mindscope/workgroups/learning/pipeline_validation/coregistration/copper_r1_2p_ls_matched_labels_df_2d_ls_seg.pkl"
    cfr2 = "/allen/programs/mindscope/workgroups/learning/pipeline_validation/coregistration/copper_r2_2p_ls_matched_labels_df_2d_ls_seg.pkl"

    with open(cfr1, "rb") as f:
        ls_cz_match_df_r1 = pkl.load(f)
        ls_cz_match_df_r1.reset_index(inplace=True)
    with open(cfr2, "rb") as f:
        ls_cz_match_df_r2 = pkl.load(f)
        ls_cz_match_df_r2.reset_index(inplace=True)

    return fov_cz_match_df, cz_stack, cz_stack_masks, ls_cz_match_df_r1, ls_cz_match_df_r2


def copper_2_expts():
    expt_table = start_lamf_analysis(verbose=False)

    filters = {"mouse_name": "Copper",
               "bisect_layer": "upper",
               "targeted_structure": "VISp",
               "tiny_session_type_num": "FIO2"}

    expt_group_copper1 = ExperimentGroup(expt_table_to_load=expt_table,
                                         filters=filters,
                                         dev=True,
                                         verbose=False)

    expt_group_copper1.load_experiments()

    expt1 = expt_group_copper1.sample_experiment()
    fov_plane_expt1 = expt1.average_projection.data

    filters = {"mouse_name": "Copper",
               "bisect_layer": "lower",
               "targeted_structure": "VISp",
               "tiny_session_type_num": "FIO2"}
    expt_group_copper2 = ExperimentGroup(expt_table_to_load=expt_table,
                                         filters=filters,
                                         dev=True,
                                         verbose=False)
    expt_group_copper2.load_experiments()

    expt2 = expt_group_copper2.sample_experiment()

    fov_plane_expt2 = expt2.average_projection.data

    return expt1, expt2, fov_plane_expt1, fov_plane_expt2


####################################################################################################
# Gene Table
####################################################################################################
def wrangle_r1_gene_table(gene_table):
    """Wrangle gene table from round 1"""

    # convert nan in gene table to 0
    gene_table = gene_table.fillna(0)
    # covert SST, PV, Gad2 to bool
    gene_table["SST"] = gene_table["SST"].astype(bool)
    gene_table["PV"] = gene_table["PV"].astype(bool)
    gene_table["Gad2"] = gene_table["Gad2"].astype(bool)

    gene_table["is_pv"] = (gene_table["PV"] == True) & (
        gene_table["SST"] == False)
    gene_table["is_sst"] = (gene_table["SST"] == True) & (
        gene_table["PV"] == False)
    gene_table["is_pv_sst"] = (gene_table["SST"] == True) & (
        gene_table["PV"] == True)

    return gene_table


def load_copper_cellxgene_data():
    def fix_df(df):
        df = df.copy()
        # drop possible_end_feet, Notes? columns
        # for col in ["possible_end_feet", "notes"]:
        for col in ["notes"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        # onyly keep unresolved_type = 0
        # df = df[df["unresolved_type"] == 0]

        # put ["2p_mask_id", "ls_mask_id"] first columns
        cols = df.columns.tolist()
        cols = cols[-2:] + cols[:-2]
        df = df[cols]

        # # set all nummeric columns to int
        # for col in df.columns:
        #     if df[col].dtype == "float64":
        #         df[col] = df[col].astype(int)
        return df

    # Load data
    cxgfile = Path(
        "/home/matt.davis/code/mjd_dev/scripts/gene_table/copper_gene_expression_r1_r2_2dsac_segmentation_label1.xlsx")

    # read R1, R2 sheet into separate dataframes
    df_r1 = pd.read_excel(cxgfile, sheet_name="R1")
    df_r2 = pd.read_excel(cxgfile, sheet_name="R2")

    # ls 2p match
    cfr1 = "/allen/programs/mindscope/workgroups/learning/pipeline_validation/coregistration/copper_r1_2p_ls_matched_labels_df_2d_ls_seg.pkl"
    cfr2 = "/allen/programs/mindscope/workgroups/learning/pipeline_validation/coregistration/copper_r2_2p_ls_matched_labels_df_2d_ls_seg.pkl"

    with open(cfr1, "rb") as f:
        ls_r1_cz_match_df = pkl.load(f)
        ls_r1_cz_match_df.reset_index(inplace=True)
    with open(cfr2, "rb") as f:
        ls_r2_cz_match_df = pkl.load(f)
        ls_r2_cz_match_df.reset_index(inplace=True)

    # merge df_r1 and  ls_r1_cz_match_df on ls_id and ls_mask_id
    df_r1 = df_r1.merge(ls_r1_cz_match_df[["2p_mask_id", "ls_mask_id"]],
                        left_on="ls_id", right_on="ls_mask_id").drop(columns=["ls_id"])
    df_r2 = df_r2.merge(ls_r2_cz_match_df[["2p_mask_id", "ls_mask_id"]],
                        left_on="ls_id", right_on="ls_mask_id").drop(columns=["ls_id"])

    df_r1 = fix_df(df_r1)
    df_r2 = fix_df(df_r2)

    # merge on 2p_mask_id, keep all rows
    df = df_r1.merge(df_r2, on="2p_mask_id", suffixes=(
        "_r1", "_r2"), how="outer").reset_index(drop=True)

    # add "both_rounds" if ls_mask_id_r1 amd ls_mask_id_r2 are not null
    df["both_rounds"] = df["ls_mask_id_r1"].notnull(
    ) & df["ls_mask_id_r2"].notnull()

    cols = [
        '2p_mask_id',
        'ls_mask_id_r1',
        'ls_mask_id_r2',
        'both_rounds',
        'SST',
        'PV',
        'VIP',
        'Tac1',
        'Npy1',
        'GAD',
        "possible_pv_end_feet",
        'unresolved_type_r1',
        'unresolved_type_r2',
    ]

    # sort df in order of cols
    df = df[cols]

    save = False
    if save:
        df.to_csv("/home/matt.davis/code/mjd_dev/scripts/gene_table/copper_gene_expression_r1_r2_2dsac_segmentation_label1.csv", index=False)

    return df


def annotate_cellxgene_table(df):
    """Copper NRSAC gene table annotation"""

    # set rows where GAD >0 to 1.0
    df.loc[df["GAD"] > 0, "GAD"] = 1.0
    df.loc[df["PV"] > 0, "PV"] = 1.0

    # if possible_pv_end_feet == True, set PV = 1.0
    df.loc[df["possible_pv_end_feet"] == True, "PV"] = 1.0

    # SUBCLASS
    # subclass
    print('lol')
    df["sc_pv"] = (df["PV"] > 0) & (df["SST"] == 0) & (df["VIP"] == 0)
    df["sc_sst"] = (df["SST"] > 0) & (df["PV"] == 0) & (df["VIP"] == 0)
    df["sc_vip"] = (df["VIP"] > 0) & (df["PV"] == 0) & (df["SST"] == 0)
    df["sc_npy"] = (df["Npy1"] > 0) & (df["VIP"] == 0) & (
        df["PV"] == 0) & (df["SST"] == 0)
    df["sc_other"] = (df["sc_pv"] == False) & (df["sc_sst"] == False) & (
        df["sc_vip"] == False) & (df["sc_npy"] == False)

    df["subclass"] = np.nan
    df.loc[(df.sc_pv == True) & (df.sc_sst == False), "subclass"] = "PV"
    df.loc[(df.sc_sst == True) & (df.sc_pv == False), "subclass"] = "SST"
    df.loc[(df.sc_vip == True), "subclass"] = "VIP"
    df.loc[(df.sc_npy == True) & (df.sc_vip == False) & (
        df.sc_pv == False) & (df.sc_sst == False), "subclass"] = "Npy"
    df.loc[(df.sc_other == True), "subclass"] = "Other"

    # set subclass nans to "Other"
    df["subclass"].fillna("Other", inplace=True)
    df["subclass"] = df["subclass"].astype(str)

    import seaborn as sns

    c = sns.color_palette()

    # convert c to list of tuple
    # c = [[tuple(x)] for x in c] # weird list of tuple thing with map
    # c = [(color[0], color[1], color[2], 1.0) for color in c]

    subclass_colors = {'Gad2': c[8], 'PV': c[0], 'SST': c[3], 'VIP': c[2],
                       # 'Npy':c[4], 'Tac1':c[9], 'GCaMP':c[8], 'Ribo':c[7], 'Other':c[8]}
                       'Npy': c[8], 'Tac1': c[9], 'GCaMP': c[8], 'Ribo': c[7], 'Other': c[8]}
    # subclass_colors = {'PV':"r"}
    # set subclass colors, map
    df["subclass_color"] = df["subclass"].map(
        subclass_colors, na_action="ignore")

    # set cat gene order
    df["subclass"] = pd.Categorical(df["subclass"], categories=[
                                    "PV", "SST", "VIP", "Npy", "Other"])

    # SUBTYPE
    # subtype
    df["st_pv_tac1"] = (df["PV"] > 0) & (df["Tac1"] > 0) & (df["Npy1"] == 0)
    df["st_pv_tac1_npy1"] = (df["PV"] > 0) & (
        df["Npy1"] > 0) & (df["Tac1"] > 0)
    df["st_pv"] = (df["PV"] > 0) & (df["st_pv_tac1"] ==
                                    False) & (df["st_pv_tac1_npy1"] == False)
    df["st_sst"] = df["SST"] > 0
    df["st_vip_npy1"] = (df["VIP"] > 0) & (df["Npy1"] > 0)
    df["st_vip_pv"] = (df["VIP"] > 0) & (df["PV"] > 0)
    # st_vip = vip - st_vip_npy1 - st_vip_pv
    df["st_vip"] = (df["VIP"] > 0) & (df["st_vip_npy1"]
                                      == False) & (df["st_vip_pv"] == False)

    df["subtype"] = np.nan
    df.loc[(df.st_pv == True), "subtype"] = "PV"
    df.loc[(df.st_pv_tac1 == True), "subtype"] = "PV-Tac1"
    df.loc[(df.st_pv_tac1_npy1 == True), "subtype"] = "PV-Tac1-Npy1"
    df.loc[(df.st_sst == True), "subtype"] = "SST"
    df.loc[(df.st_vip == True), "subtype"] = "VIP"
    df.loc[(df.st_vip_npy1 == True), "subtype"] = "VIP-Npy1"
    df.loc[(df.st_vip_pv == True), "subtype"] = "VIP-PV"
    import random
    random.seed(0)
    # set cat gene order
    df["subtype"] = pd.Categorical(df["subtype"], categories=[
                                   "PV", "PV-Tac1", "PV-Tac1-Npy1", "SST", "VIP", "VIP-Npy1", "VIP-PV"])

    def get_random_color():
        """Get random color"""
        # set seed

        def r(): return random.randint(0, 255)
        return '#%02X%02X%02X' % (r(), r(), r())

    # cell_type colors
    # get unique colors for all subtypes
    subtype_colors = {}
    for subtype in df["subtype"].unique():
        subtype_colors[subtype] = get_random_color()

    # set subtype colors
    df["subtype_color"] = df["subtype"].map(subtype_colors)

    return df

############################################################################################################
# Plots
############################################################################################################


def imshow_projection(img, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    vmax = np.percentile(img, 99.5)
    ax.imshow(img, cmap='gray', vmax=vmax)
    ax.axis('off')

    return ax


def plot_fov_cz_masks(fov_mask_df,
                      fov_plane_img,
                      cz_stack_masks,
                      cz_stack,
                      ls_cz_match_df,
                      cz_colors="coreg",
                      blank_bg: bool = True,
                      fov_bg: bool = False,
                      cz_bg: bool = False,
                      text_label: bool = False,
                      ax: plt.Axes = None, **kwargs):

    if ax is None:
        fig, ax = plt.subplots(1, figsize=(10, 10))
    # plot contours of explode masks in red and expt2_masks in blue

    # fov_mask_df = expt2_masks # all or deoverlapped

    # cant have both blank_bg and img_bg
    if blank_bg and fov_bg:
        raise ValueError("blank_bg and fov_bg cannot both be True")

    # only one fov should be in fov_mask_df
    assert fov_mask_df["cz_match_plane_index"].nunique() == 1
    match_plane_ind = int(fov_mask_df["cz_match_plane_index"].iloc[0])

    cz_plane_img = cz_stack[match_plane_ind]
    cz_match_masks = cz_stack_masks[match_plane_ind]
    cz_stack_ids, cz_match_masks = explode_masks(cz_match_masks)

    if blank_bg:
        blank_image(ax=ax)
    if fov_bg:
        imshow_projection(fov_plane_img, ax=ax)
    if cz_bg:
        imshow_projection(cz_plane_img, ax=ax)

    ls_id_count = 0
    texts = []
    if cz_colors == "coreg":
        for id, mask in zip(cz_match_masks, cz_stack_ids):
            # get mask num from mask
            # get id in cz_ls_rois_dict, return none if not found
            # if found, get the roi_id

            ls_id = ls_cz_match_df[ls_cz_match_df["2p_mask_id"]
                                   == id].ls_mask_id

            if ls_id.shape[0] > 0:
                ls_id = ls_id.iloc[0]
                ls_id_count += 1

                plt.contour(mask, colors='y', linewidths=1, label=id)

                # get mask centroid
                mask_centroid = np.array(np.where(mask == 1)).mean(axis=1)

                # add the id to the plot at the center of the mask
                id_text = f"{id}-{ls_id}"

                # gather texts
                t = plt.text(
                    mask_centroid[1]-40, mask_centroid[0]+20, id_text, color='w', fontsize=10)
                texts.append(t)
            else:
                plt.contour(mask, colors='r', linewidths=1, label=id)
    elif cz_colors == "subclass":
        # if gene_table is None:
        #     raise ValueError("gene_table must be provided if cz_colors == 'subclass'")
        for id, mask in zip(cz_match_masks, cz_stack_ids):
            # get mask num from mask
            # get id in cz_ls_rois_dict, return none if not found
            # if found, get the roi_id

            ls_id = ls_cz_match_df[ls_cz_match_df["2p_mask_id"]
                                   == id].ls_mask_id

            cz_stack_id = fov_mask_df[fov_mask_df["cz_stack_id"]
                                      == id].cz_stack_id

            # if ls_id.shape[0] > 0:
            if cz_stack_id.shape[0] > 0:
                # ls_id = ls_id.iloc[0]
                # ls_id_count+=1

                # get subclass

                subclass = fov_mask_df[fov_mask_df["cz_stack_id"]
                                       == id].subclass.iloc[0]
                color = fov_mask_df[fov_mask_df["cz_stack_id"]
                                    == id].subclass_color.iloc[0]
                # print(id, subclass, color)

                # if color is not nan
                if color is not np.nan:
                    plt.contour(mask, colors=color, linewidths=1, label=id)

                if text_label:
                    id_text = f"{id}-{ls_id}"
                    mask_centroid = np.array(np.where(mask == 1)).mean(axis=1)
                    t = plt.text(
                        mask_centroid[1]-40, mask_centroid[0]+20, id_text, color='w', fontsize=10)
                    texts.append(t)
            else:
                plt.contour(mask, colors='grey', linewidths=1, label=id)

    for name, row in fov_mask_df.iterrows():
        plt.contour(row.roi_mask_shifted, colors='yellow', linewidths=.5,
                    linestyle='.', alpha=0.5, label=row.cz_stack_id)

    n_nonmatch = len(cz_match_masks) - ls_id_count
    n_match = ls_id_count
    # plt.title('Contours of masks from CZ stack (RED = not LS matched, YELLOW = LS matched) \n and Copper V1 Lower Deoverlapped ROIs (GREEN) \n PCC registration \n IOU matched > 0.1')
    plt.title(
        f"Copper, VISp, Upper plane \n Cortical Z-stack \n(RED = not LS matched n={n_nonmatch}; YELLOW = LS matched n={n_match}")

    # axis off
    plt.axis('off')

    if adjust_text is None and text_label:
        # warning no adjust text
        print("WARNING: text_label is True but adjust_text is None. Plotting without external library (fancy text features).")
        for t in texts:
            t.set_visible(True)
    elif adjust_text is not None and text_label:
        adjust_text(texts)
        # adjust_text(texts, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))  # alt way to add arrows


def plot_cz_ls_masks_with_fov(fov_mask_df,
                              fov_plane_img,
                              cz_stack_masks,
                              cz_stack,
                              ls_cz_match_df,
                              gene_table: pd.DataFrame = None,
                              cz_colors="coreg",
                              plot_fov_rois: bool = False,
                              background: str = "blank",
                              filled_contours: bool = False,
                              text_label: bool = False,
                              ax: plt.Axes = None, **kwargs):

    if ax is None:
        fig, ax = plt.subplots(1, figsize=(10, 10))
    # plot contours of explode masks in red and expt2_masks in blue

    # fov_mask_df = expt2_masks # all or deoverlapped

    # background should be fov, cz, or blank, check
    if background not in ["fov", "cz", "blank"]:
        raise ValueError("background must be 'fov', 'cz', or 'blank'")

    # only one fov should be in fov_mask_df
    assert fov_mask_df["cz_match_plane_index"].nunique() == 1
    match_plane_ind = int(fov_mask_df["cz_match_plane_index"].iloc[0])

    cz_plane_img = cz_stack[match_plane_ind]
    cz_match_masks = cz_stack_masks[match_plane_ind]
    cz_stack_ids, cz_match_masks = explode_masks(cz_match_masks)

    if background == "blank":
        blank_image(ax=ax)
    elif background == "fov":
        imshow_projection(fov_plane_img, ax=ax)
    elif background == "cz":
        imshow_projection(cz_plane_img, ax=ax)

    ls_id_count = 0
    texts = []

    if cz_colors == "subclass":
        # if gene_table is None:
        #     raise ValueError("gene_table must be provided if cz_colors == 'subclass'")
        for i, (id, mask) in enumerate(zip(cz_match_masks, cz_stack_ids)):
            # get mask num from mask
            # get id in cz_ls_rois_dict, return none if not found
            # if found, get the roi_id

            gene_table_roi = gene_table[gene_table["2p_mask_id"] == id]
            assert gene_table_roi.shape[
                0] < 2, f"gene_table_roi.shape[0] == {gene_table_roi.shape[0]}"

            # if ls_id.shape[0] > 0:
            if gene_table_roi.shape[0] == 1:

                cz_stack_id = gene_table_roi["2p_mask_id"].iloc[0]
                round_id = 2
                ls_id = gene_table_roi[f"ls_mask_id_r{round_id}"].iloc[0]

                # check is ls_id is np.nan

                if np.isnan(ls_id):
                    ls_id = "nan"
                else:
                    ls_id = int(ls_id)

                subclass = gene_table_roi.subclass.iloc[0]
                color = gene_table_roi.subclass_color.iloc[0]
                # print(id, subclass, color)

                if color is not np.nan:
                    if filled_contours:
                        ma = np.ma.masked_where(mask == 0, mask)
                        # color in list ?
                        ax.contourf(ma, colors=[color],
                                    alpha=1, label=subclass)
                        ax.contour(mask, colors='k',
                                   linewidths=.05, label=subclass)

                    else:
                        ax.contour(mask, colors=[color],
                                   linewidths=1, label=subclass)

                if text_label:
                    id_text = f"{cz_stack_id}-{ls_id}"
                    mask_centroid = np.array(np.where(mask == 1)).mean(axis=1)
                    # t = plt.text(mask_centroid[1]-40, mask_centroid[0]+20, id_text, color='w', fontsize=10)
                    t = plt.text(
                        mask_centroid[1]-5, mask_centroid[0]+5, id_text, color='w', fontsize=8)
                    texts.append(t)
            else:
                if filled_contours:
                    ma = np.ma.masked_where(mask == 0, mask)
                    ax.contourf(ma, colors='grey', alpha=1,
                                label="Not Coregistered")
                    ax.contour(mask, colors='k', linewidths=.05,
                               label="Not Coregistered")
                else:
                    ax.contour(mask, colors='grey', linewidths=1,
                               label="Not Coregistered")

    if plot_fov_rois:
        for name, row in fov_mask_df.iterrows():
            ax.contour(row.roi_mask_shifted, colors='yellow', linewidths=.5,
                       linestyle='.', alpha=0.5, label=row.cz_stack_id)

    n_nonmatch = len(cz_match_masks) - ls_id_count
    n_match = ls_id_count
    # plt.title('Contours of masks from CZ stack (RED = not LS matched, YELLOW = LS matched) \n and Copper V1 Lower Deoverlapped ROIs (GREEN) \n PCC registration \n IOU matched > 0.1')
    # ax.set_title(f"Copper, VISp, Upper plane \n Cortical Z-stack \n")

    legend = False
    if legend:

        # add legend for contours
        plt.rcParams['legend.fontsize'] = 22
        labels = ["PV", "SST", "VIP", "Npy", "Other", "Not coregistered"]

        c = sns.color_palette()

    # convert c to list of tuple
    # c = [[tuple(x)] for x in c] # weird list of tuple thing with map
    # c = [(color[0], color[1], color[2], 1.0) for color in c]

        subclass_colors = {'Gad2': c[8], 'PV': c[0], 'SST': c[3], 'VIP': c[2],
                           'Npy': c[4], 'Tac1': c[9], 'GCaMP': c[8], 'Ribo': c[7], 'Other': c[8], 'Not coregistered': 'grey'}

        # map colors to labels
        colors = [subclass_colors[label] for label in labels]
        handles = [plt.Line2D([], [], color=c, linewidth=5,
                              linestyle='-') for c in colors]
        # show legend
        by_label = dict(zip(labels, handles))
        # plt.legend(by_label.values(), by_label.keys(), loc='upper right')
        plt.legend(by_label.values(), by_label.keys(),
                   loc='upper right', bbox_to_anchor=(1.5, 1))

    # axis off
    ax.axis('off')

    if text_label:
        adjust_text(texts, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
        # adjust_text(texts)


def plot_cz_ls_masks(match_plane_ind,
                     fov_plane_img,
                     cz_stack_masks,
                     cz_stack,
                     ls_cz_match_df,
                     gene_table: pd.DataFrame = None,
                     cz_colors: str = "coreg",
                     cz_alpha: float = 1,
                     background: str = "blank",
                     filled_contours: bool = False,
                     text_label: bool = False,
                     legend: bool = False,
                     ax: plt.Axes = None, **kwargs):
    """Plot masks from cortical zstack that have been matched to a 2P-FOV plane.

    Useful to show cortical zstack rois that are coregistered light sheet/mfish data with colored 
    by gene expression and lack of coregistration between CZ-LS (grey colored).

    Parameters
    ----------
    match_plane_ind : int
        index of CZ plane to plot. This is usually matched to a particular 2P-FOV.
    fov_plane_img : np.ndarray
        2P-FOV image to plot, potentially as background
    cz_stack_masks : np.ndarray
        CZ masks to plot. Should be a 3D array of masks, with the first dimension being the CZ plane.
        Each mask should be a 2D array of unique integers.
    cz_stack : np.ndarray
        CZ image stack. Should be a 3D array of CZ images, with the first dimension being the CZ plane.
    ls_cz_match_df:
        unused, perhaps duplicate with gene table information.
    gene_table : pd.DataFrame, optional
        Cell x gene table that has expression calls for some mask/cells in the cz_stack_masks.
        TODO: more documentation where to find and the columnns
    cz_colors : str, optional
        How to color the CZ masks.
        "coreg", then color by whether the CZ mask was matched to a light sheet mask. TODO: not implemented
        "subclass", then color by the subclass of the CZ mask (gene_table needed)
        Otherwise, provide a color to use on all masks
    cz_alpha : float, optional
        Alpha value to use for CZ masks, by default 1. 0 usefully hides the CZ masks.
    background : str, optional
        What to plot as background.
        "fov" plots the 2P-FOV image,
        "cz" plots the CZ image,
        "blank" plots nothing (using blank_image(), adjust params there)
    filled_contours : bool, optional
        Whether to fill the CZ masks with color, by default False
    text_label : bool, optional
        Whether to label the CZ masks with their ids (format: CZID-LSID), by default False
        Required: gene_table, where LSIDs are stored. Assumes Round 1 ids.
    legend : bool, optional
        Whether to plot a legend, by default False
    ax : plt.Axes, optional
        Axis to plot on, by default None

    """

    if ax is None:
        fig, ax = plt.subplots(1, figsize=(10, 10))

    if background not in ["fov", "cz", "blank"]:
        raise ValueError("background must be 'fov', 'cz', or 'blank'")

    cz_plane_img = cz_stack[match_plane_ind]
    cz_match_masks = cz_stack_masks[match_plane_ind]
    cz_stack_ids, cz_match_masks = explode_masks(cz_match_masks)

    if background == "blank":
        blank_image(ax=ax)
    elif background == "fov":
        imshow_projection(fov_plane_img, ax=ax)
    elif background == "cz":
        imshow_projection(cz_plane_img, ax=ax)
    colors = None
    # kinda hacky coloring
    if cz_colors == "subclass":
        if gene_table is None:
            raise ValueError(
                "gene_table must be provided if cz_colors == 'subclass'")
    else:
        # make list of colors for each cz_stack_id
        # TODO check valid color string
        colors = [cz_colors] * len(cz_stack_ids)

    # add metric colors
    # qc_key = "intercentroid_distance_um"
    # gene_table = gene_table.merge(ls_cz_match_df, on='2p_mask_id')
    # cmap = plt.get_cmap('viridis')
    # gene_table['color_dist'] = gene_table[qc_key].apply(cmap)

    # qc_key = "iou"
    # cmap = plt.get_cmap('viridis')
    # gene_table['color_iou'] = gene_table[qc_key].apply(cmap)
    # print(gene_table['color_dist'])
    texts = []
    for i, (id, mask) in enumerate(zip(cz_match_masks, cz_stack_ids)):
        gene_table_roi = gene_table[gene_table["2p_mask_id"] == id]
        assert gene_table_roi.shape[
            0] < 2, f"gene_table_roi.shape[0] == {gene_table_roi.shape[0]}"

        if gene_table_roi.shape[0] == 1:

            cz_stack_id = gene_table_roi["2p_mask_id"].iloc[0]
            round_id = 2
            ls_id = gene_table_roi[f"ls_mask_id_r{round_id}"].iloc[0]

            if np.isnan(ls_id):
                ls_id = "nan"
            else:
                ls_id = int(ls_id)

            subclass = gene_table_roi.subclass.iloc[0]
            if cz_colors == "subclass":
                color = gene_table_roi.subclass_color.iloc[0]
            elif cz_colors == "dist":
                color = gene_table_roi.color_dist.iloc[0]
            elif cz_colors == "iou":
                color = gene_table_roi.color_iou.iloc[0]
            else:
                color = colors[i]

            if color is not np.nan:
                if filled_contours:
                    ma = np.ma.masked_where(mask == 0, mask)
                    # color in list ?
                    ax.contourf(ma, colors=[color],
                                alpha=cz_alpha, label=subclass)
                    ax.contour(mask, colors='k',
                               linewidths=.05, label=subclass)

                else:
                    ax.contour(mask, colors=[
                               color], linewidths=1, alpha=cz_alpha, label=subclass)

            if text_label:
                id_text = f"{cz_stack_id}-{ls_id}"
                mask_centroid = np.array(np.where(mask == 1)).mean(axis=1)
                # t = plt.text(mask_centroid[1]-40, mask_centroid[0]+20, id_text, color='w', fontsize=10)
                t = plt.text(
                    mask_centroid[1]-5, mask_centroid[0]+13, id_text, color='w', fontsize=9)
                texts.append(t)
        else:
            color = "grey" if cz_colors == "subclass" else colors[i]
            coreg_alpha = cz_alpha
            coreg_alpha = 1
            color = "#D3D3D3"
            if filled_contours:
                ma = np.ma.masked_where(mask == 0, mask)
                ax.contourf(ma, colors=color, alpha=coreg_alpha,
                            label="Not Coregistered")
                ax.contour(mask, colors='k', linewidths=.05,
                           alpha=coreg_alpha, label="Not Coregistered")
            else:
                ax.contour(mask, colors=color, linewidths=1,
                           alpha=coreg_alpha, label="Not Coregistered")

    if legend:
        if cz_colors == "subclass":
            cz_subclass_legend(ax)

    # axis off
    ax.axis('off')

    # colorbar
    if cz_colors == "dist":
        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        cbar.set_label('Distance (um)', rotation=270, labelpad=20)
    if cz_colors == "iou":
        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        cbar.set_label('Intersection over Union', rotation=270, labelpad=20)

    if text_label:
        # adjust_text(texts, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
        adjust_text(texts, ax=ax)


def plot_cz_ls_masks_n(match_plane_ind,
                       fov_plane_img,
                       cz_stack_masks,
                       cz_stack,
                       ls_cz_match_df,
                       gene_table: pd.DataFrame = None,
                       cz_colors: str = "coreg",
                       cz_alpha: float = 1,
                       background: str = "blank",
                       filled_contours: bool = False,
                       text_label: bool = False,
                       legend: bool = False,
                       ax: plt.Axes = None, **kwargs):
    """Plot masks from cortical zstack that have been matched to a 2P-FOV plane.

    Useful to show cortical zstack rois that are coregistered light sheet/mfish data with colored 
    by gene expression and lack of coregistration between CZ-LS (grey colored).

    Parameters
    ----------
    match_plane_ind : int
        index of CZ plane to plot. This is usually matched to a particular 2P-FOV.
    fov_plane_img : np.ndarray
        2P-FOV image to plot, potentially as background
    cz_stack_masks : np.ndarray
        CZ masks to plot. Should be a 3D array of masks, with the first dimension being the CZ plane.
        Each mask should be a 2D array of unique integers.
    cz_stack : np.ndarray
        CZ image stack. Should be a 3D array of CZ images, with the first dimension being the CZ plane.
    ls_cz_match_df:
        unused, perhaps duplicate with gene table information.
    gene_table : pd.DataFrame, optional
        Cell x gene table that has expression calls for some mask/cells in the cz_stack_masks.
        TODO: more documentation where to find and the columnns
    cz_colors : str, optional
        How to color the CZ masks.
        "coreg", then color by whether the CZ mask was matched to a light sheet mask. TODO: not implemented
        "subclass", then color by the subclass of the CZ mask (gene_table needed)
        Otherwise, provide a color to use on all masks
    cz_alpha : float, optional
        Alpha value to use for CZ masks, by default 1. 0 usefully hides the CZ masks.
    background : str, optional
        What to plot as background.
        "fov" plots the 2P-FOV image,
        "cz" plots the CZ image,
        "blank" plots nothing (using blank_image(), adjust params there)
    filled_contours : bool, optional
        Whether to fill the CZ masks with color, by default False
    text_label : bool, optional
        Whether to label the CZ masks with their ids (format: CZID-LSID), by default False
        Required: gene_table, where LSIDs are stored. Assumes Round 1 ids.
    legend : bool, optional
        Whether to plot a legend, by default False
    ax : plt.Axes, optional
        Axis to plot on, by default None

    """

    if ax is None:
        fig, ax = plt.subplots(1, figsize=(10, 10))

    if background not in ["fov", "cz", "blank"]:
        raise ValueError("background must be 'fov', 'cz', or 'blank'")

    cz_plane_img = cz_stack[match_plane_ind]
    cz_match_masks = cz_stack_masks[match_plane_ind]
    cz_match_masks, cz_stack_ids = explode_masks(cz_match_masks)

    if background == "blank":
        blank_image(ax=ax)
    elif background == "fov":
        imshow_projection(fov_plane_img, ax=ax)
    elif background == "cz":
        imshow_projection(cz_plane_img, ax=ax)

    # kinda hacky coloring
    if cz_colors == "subclass":
        if gene_table is None:
            raise ValueError(
                "gene_table must be provided if cz_colors == 'subclass'")
    else:
        # make list of colors for each cz_stack_id
        # TODO check valid color string
        colors = [cz_colors] * len(cz_stack_ids)

    # add metric colors
    # qc_key = "intercentroid_distance_um"
    # gene_table = gene_table.merge(ls_cz_match_df, on='2p_mask_id')
    # cmap = plt.get_cmap('viridis')
    # gene_table['color_dist'] = gene_table[qc_key].apply(cmap)

    # qc_key = "iou"
    # cmap = plt.get_cmap('viridis')
    # gene_table['color_iou'] = gene_table[qc_key].apply(cmap)
    # print(gene_table['color_dist'])
    texts = []

    stack_df = pd.DataFrame(
        {"cz_stack_id": cz_stack_ids, "cz_mask": cz_match_masks})

    # "cz_stack_id" contains all possible rois in that plane, "2p_mask_id" contains rois considered for coreg

    gene_table = gene_table.merge(
        stack_df, left_on="2p_mask_id", right_on="cz_stack_id", how="outer")

    # # keep only this ids for match_plane, since gene table has other plan potential ids

    # merge with gene_table
    if gene_table is not None:
        gene_table = gene_table.merge(
            stack_df, right_on="2p_mask_id", left_on="cz_stack_id", how="right")
        gene_table = gene_table[gene_table["cz_stack_id"].notnull()]

        # set sublass color for 2p_mask_id = nan to grey
        gene_table.loc[gene_table["2p_mask_id"].isnull(),
                       "subclass"] = "Not coregistered"
        gene_table.loc[gene_table["2p_mask_id"].isnull(), "color"] = "grey"

    # filter gene_table for "2p_mask"
    for i, (mask, id) in enumerate(zip(cz_match_masks, cz_stack_ids)):
        gene_table_roi = gene_table[gene_table["2p_mask_id"] == id]
        assert gene_table_roi.shape[
            0] < 2, f"gene_table_roi.shape[0] == {gene_table_roi.shape[0]}"

        if gene_table_roi.shape[0] == 1:

            cz_stack_id = gene_table_roi["2p_mask_id"].iloc[0]
            round_id = 2
            ls_id = gene_table_roi[f"ls_mask_id_r{round_id}"].iloc[0]

            if np.isnan(ls_id):
                ls_id = "nan"
            else:
                ls_id = int(ls_id)

            subclass = gene_table_roi.subclass.iloc[0]
            if cz_colors == "subclass":
                color = gene_table_roi.subclass_color.iloc[0]
            if cz_colors == "dist":
                color = gene_table_roi.color_dist.iloc[0]
            if cz_colors == "iou":
                color = gene_table_roi.color_iou.iloc[0]
            else:
                color = colors[i]

            if color is not np.nan:
                if filled_contours:
                    ma = np.ma.masked_where(mask == 0, mask)
                    print(color)
                    # color in list ?
                    ax.contourf(ma, colors=[color],
                                alpha=cz_alpha, label=subclass)
                    ax.contour(mask, colors='k',
                               linewidths=.05, label=subclass)

                else:
                    ax.contour(mask, colors=[
                               color], linewidths=1, alpha=cz_alpha, label=subclass)

            if text_label:
                id_text = f"{cz_stack_id}-{ls_id}"
                mask_centroid = np.array(np.where(mask == 1)).mean(axis=1)
                # t = plt.text(mask_centroid[1]-40, mask_centroid[0]+20, id_text, color='w', fontsize=10)
                t = plt.text(
                    mask_centroid[1]-5, mask_centroid[0]+13, id_text, color='w', fontsize=9)
                texts.append(t)
        else:
            color = "grey" if cz_colors == "subclass" else colors[i]
            if filled_contours:
                ma = np.ma.masked_where(mask == 0, mask)
                ax.contourf(ma, colors=color, alpha=cz_alpha,
                            label="Not Coregistered")
                ax.contour(mask, colors='k', linewidths=.05,
                           label="Not Coregistered")
            else:
                ax.contour(mask, colors=color, linewidths=1,
                           alpha=cz_alpha, label="Not Coregistered")

    if legend:
        if cz_colors == "subclass":
            cz_subclass_legend(ax)

    # axis off
    ax.axis('off')

    # colorbar
    if cz_colors == "dist":
        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        cbar.set_label('Distance (um)', rotation=270, labelpad=20)
    if cz_colors == "iou":
        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        cbar.set_label('Intersection over Union', rotation=270, labelpad=20)

    if text_label:
        # adjust_text(texts, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
        adjust_text(texts, ax=ax)


def cz_subclass_legend(ax):
    plt.rcParams['legend.fontsize'] = 22
    labels = ["PV", "SST", "VIP", "Npy", "Other", "Not coregistered"]
    labels = ["PV+", "SST+", "VIP+", "Npy+", "Other", "Not coregistered"]
    labels = ["PV+", "SST+", "VIP+", "Other", "Not coregistered"]

    c = sns.color_palette()

    subclass_colors = {'Gad2': c[8], 'PV+': c[0], 'SST+': c[3], 'VIP+': c[2],
                       # 'Npy': c[4], 'Tac1': c[9], 'GCaMP': c[8], 'Ribo': c[7],
                       'Npy': c[8], 'Tac1': c[9], 'GCaMP': c[8], 'Ribo': c[7],
                       'Other': c[8], 'Not coregistered': 'grey'}

    # map colors to labels
    colors = [subclass_colors[label] for label in labels]
    handles = [plt.Line2D([], [], color=c, linewidth=5,
                          linestyle='-') for c in colors]
    # show legend
    by_label = dict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    # box off
    plt.legend(by_label.values(), by_label.keys(), loc='upper right',
               bbox_to_anchor=(1.5, 1), frameon=False)

# %% MISSING CSIDS


def missing_csids():
    csid_missing_file = "//allen/programs/mindscope/workgroups/learning/analysis_plots/ophys/activity_correlation_lamf/nrsac/roi_match/copper_all_roi_table.pkl"

    # load
    with open(csid_missing_file, "rb") as f:
        csid_missing = pkl.load(f)

    n_nans = csid_missing.cell_specimen_id.isna().sum()
    # # make a list of unique ids,starting with 99999 and 10 digits long
    unique_ids = [9999900000 + i for i in range(n_nans)]

    # get naan index
    nan_idx = csid_missing.cell_specimen_id.isna()

    # replace with unique
    csid_missing.loc[nan_idx, 'cell_specimen_id'] = unique_ids

    save = False
    if save:
        # out file
        out_file = "//allen/programs/mindscope/workgroups/learning/analysis_plots/ophys/activity_correlation_lamf/nrsac/roi_match/copper_missing_osid_roi_table_nan_replaced.pkl"

        # save
        with open(out_file, "wb") as f:
            pkl.dump(csid_missing, f)

    return csid_missing


def add_missing_csid_expt_grp(expt_group):
    for oeid, expt in expt_group.experiments.items():
        csid_missing = missing_csids()
        oeid_missing = csid_missing.ophys_experiment_id.unique()

        # check if oeid is in missing

        # remove ophsy_experiment_id = [1193675218, 1193675220] from missing
        # oeid_missing = oeid_missing[~np.isin(oeid_missing, [1193675218, 1193675220])]
        if oeid in oeid_missing:

            for key in ["cell_specimen_table", "roi_masks", "dff_traces", "events"]:

                # get attribute from expt
                dfc = getattr(expt, key)
                print(dfc)
                dfc.reset_index(inplace=True)

                for idx, row in dfc.iterrows():
                    # get cell_roi_id
                    cell_roi_id = row.cell_roi_id

                    # find matching  csid in csid_missing
                    csid = csid_missing[csid_missing.cell_roi_id ==
                                        cell_roi_id].cell_specimen_id.values[0]
                    # set cell_specimen_id to csid
                    dfc.loc[idx, 'cell_specimen_id'] = csid

                    # drop original index, set cell_specimen_id as index

                dfc.set_index('cell_specimen_id', inplace=True)

                # set attribute
                setattr(expt, key, dfc)


##########################################
# Stim response plots
##########################################

def add_gene_call_expression_plots_to_gene_table(gene_table):
    r1_path = "//allen/programs/mindscope/workgroups/learning/mattd/copper_coreg_dev/gene_expression_plots_copper_r1_2dsac_percVmax"
    r2_path = "//allen/programs/mindscope/workgroups/learning/mattd/copper_coreg_dev/gene_expression_plots_copper_r2_2dsac_t1boost"

    # get all filepaths in r1
    r1_fps = []
    r1_ls_ids = []
    for root, dirs, files in os.walk(r1_path):
        for file in files:
            if file.endswith(".png"):
                r1_fps.append(os.path.join(root, file))

                # extract ls_id (split on _ 4th element)
                r1_ls_ids.append(int(file.split("_")[4]))

    # get all filepaths in r2
    r2_fps = []
    r2_ls_ids = []
    for root, dirs, files in os.walk(r2_path):
        for file in files:
            if file.endswith(".png"):
                r2_fps.append(os.path.join(root, file))

                # extract ls_id (split on _ 4th element)
                r2_ls_ids.append(int(file.split("_")[4]))

    # merge with gene_table on ls_mask_id_r1
    gene_table = gene_table.merge(pd.DataFrame(
        {"ls_mask_id_r1": r1_ls_ids, "r1_fp": r1_fps}), on="ls_mask_id_r1")
    gene_table = gene_table.merge(pd.DataFrame(
        {"ls_mask_id_r2": r2_ls_ids, "r2_fp": r2_fps}), on="ls_mask_id_r2")

    return gene_table


def plot_gene_expression_for_czid(czid, gene_table):
    # get row
    row = gene_table[gene_table["2p_mask_id"] == czid]

    # plot
    fig, ax = plt.subplots(2, 1, figsize=(40, 10))
    ax[0].imshow(plt.imread(row["r1_fp"].values[0]))
    ax[0].set_title("r1")
    ax[1].imshow(plt.imread(row["r2_fp"].values[0]))
    ax[1].set_title("r2")
    plt.show()


def fill_csid_with_rows(expt_group):

    for id, expt in expt_group.experiments.items():
        if expt.dff_traces.index.isnull().all():
            # replaces with row number
            # assume all keys null
            for key in ["dff_traces", "events", "cell_specimen_table", "roi_masks"]:

                # get attrbute and set the index to the row number
                attr = getattr(expt, key)
                # attr.index = attr.index.reset_index(drop=True)
                attr = attr.reset_index(drop=True)

                # reset index name
                attr.index.name = "cell_specimen_id"

                # set the attribute
                setattr(expt, key, attr)


def merge_msrdf_and_gene_table(msr_df):
    copper_coreg_dir = Path(
        "/allen/programs/mindscope/workgroups/learning/mattd/copper_coreg_dev")
    rois_df = pd.read_pickle(
        copper_coreg_dir / 'all_expt_rois_cz_matched_copper_nrsac_3.pkl')

    rois_df["bisect_layer"] = pd.cut(rois_df.cz_match_plane_index, bins=[
                                     0, 40, 80], labels=['upper', 'lower'])

    rois_df2 = annotate_cellxgene_table(rois_df)

    msr_df = msr_df.merge(rois_df2, on="cell_roi_id", suffixes=('', '_y'))

    return msr_df


def filt_by_tstn(df, tstn):
    df["tiny_session_type_num"] = pd.Categorical(
        df["tiny_session_type_num"], categories=tstn, ordered=True)
    df = df[df.tiny_session_type_num.isin(tstn)]
    return df



if __name__ == "__main__":
    # TODO: finish CLI 
    args = parser.parse_args()
    oeid = args.oeid
    cz_stack_path = args.cz_stack_path
    
    expt = get_ophys_expt(oeid) # for now don't need dev object, rois are same (5/11/23)
    # Next:
    # 1. flag to filter rois (by classifier)
    # 2. check cz_stack_path valid/data expected
    # 3. get JK match input or create function to calculate.
    # 4. run coreg
    # 5. save coreg'd rois
    # 6. Bonus: should i accept expt_group as cli?




    
    
