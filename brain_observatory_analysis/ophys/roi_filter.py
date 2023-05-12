# core
from pathlib import Path
from typing import Union

# 3rd party standard
import pandas as pd

# brain observatory
from allensdk.brain_observatory.behavior.behavior_ophys_experiment import \
    BehaviorOphysExperiment
from brain_observatory_qc.data_access.behavior_ophys_experiment_dev import \
    BehaviorOphysExperimentDev


def filter_rois_with_with_nrsac_classifier(expt: Union[BehaviorOphysExperiment, BehaviorOphysExperimentDev]
                                           , pthresh: float = None):
    # TODO: cell_specimen_table, corrected traces

    # show warn, only filters roi_masks, not dff_traces, events, corrected_fluorescence_traces
    print("Roi filtered applied to roi_masks, events, dff_traces")
    roi_filtering_fn = Path(
        '/allen/programs/mindscope/workgroups/learning/pipeline_validation/classify_rois_sac2023/LAMF_NR_SAC2023_iscell_lr_model_version2_04142023.pkl')
    roi_filtering = pd.read_pickle(roi_filtering_fn)
    oeid = expt.ophys_experiment_id

    # n masks
    n_masks = expt.roi_masks.shape[0]

    roi_filtering = roi_filtering[roi_filtering.ophys_experiment_id == oeid]

    if pthresh is not None:
        iscell = roi_filtering.iscell_prob >= pthresh
    else:
        iscell = roi_filtering["iscell_p0.4"] == 1

    filt_cell_roi_ids = roi_filtering[iscell].cell_roi_id.values
    expt.roi_masks = expt.roi_masks[expt.roi_masks.cell_roi_id.isin(
        filt_cell_roi_ids)]
    expt.dff_traces = expt.dff_traces[expt.dff_traces.cell_roi_id.isin(
        filt_cell_roi_ids)]

    # sometimes events doesn't exist
    try:
        expt.events = expt.events[expt.events.cell_roi_id.isin(
            filt_cell_roi_ids)]
    except KeyError:
        print("No events, skipping roi filter")

    n_masks_filt = expt.roi_masks.shape[0]
    expt.is_roi_filtered = True
    print(f"Original masks: {n_masks} \nFiltered masks: {n_masks_filt}")

    return expt
