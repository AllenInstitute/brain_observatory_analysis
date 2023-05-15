from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from allensdk.brain_observatory.behavior.behavior_ophys_experiment import \
    BehaviorOphysExperiment

from brain_observatory_qc.pipeline_dev import calculate_new_dff

from allensdk.brain_observatory.behavior.event_detection import \
    filter_events_array

from brain_observatory_analysis.utilities import experiment_table_utils as etu

from typing import Union

DFF_PATH = Path(
    "//allen/programs/mindscope/workgroups/learning/pipeline_validation/dff")
GH_DFF_PATH = Path(
    "//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/Jinho/data/GH_data/dff")
VB_DFF_PATH = Path(
    "//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/Jinho/data/VB_data/dff")


EVENTS_ROOT_PATH = Path("/allen/programs/mindscope/workgroups/learning/pipeline_validation/events/")
EVENTS_PATH = EVENTS_ROOT_PATH / "oasis_nrsac_v1"

GH_EVENTS_PATH = Path(
    "//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/Jinho/data/GH_data/event_oasis")
VB_EVENTS_PATH = Path(
    "//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/Jinho/data/VB_data/event_oasis")


CELLXGENE_PATH = Path(
    "//allen/programs/mindscope/workgroups/learning/analysis_data_cache/cellXgene/dev")


class BehaviorOphysExperimentDev:
    """Wrapper class for BehaviorOphysExperiment that adds custom
     methods, loads from_lims() only

    Parameters
    ----------
    ophys_experiment_id : int
        The ophys_experiment_id of the experiment to be loaded.
    events_path: str or Path
        Folder where events files are stored. Default is
        "/allen/programs/mindscope/workgroups/learning/pipeline_validation/events/oasis_nrsac_v1".
        will not return events that oeid is not found in this folder.
    filtered_events_params: dict
        Parameters to be passed to filter_events_array
    load_or_calc_new_dff: bool
        If True, will look for new_dff in pipeline_dev folder.
        If False, will use dff_traces from Dev or SDK experiment object.
        Around mid 2023, ophsy_etl_pipeline was updated to use "new_dff", so we no longer need to
        set this to True for experiments after that date, or reprocessed experiments.
    kwargs : dict
        Keyword arguments to be passed to the BehaviorOphysExperiment

    Returns
    -------
    BehaviorOphysExperimentDev
        An instance of the BehaviorOphysExperimentDev class.

    Notes
    -----

    Uses "duck typing" to override the behavior of the
    BehaviorOphysExperiment class. All uknown requests are passed to the
    BehaviorOphysExperiment class. One issue wit this approach:
    that uses isinstance or issubclass will not work as expected.
    see: https://stackoverflow.com/a/60509130

    _get_new_dff looks for new_dffs in pipeline_dev folder, will throw
    error if not found.

    Example usage:
    expt_id = 1191477425
    expt = BehaviorOphysExperimentDev(expt_id, skip_eye_tracking=True)

    """
    def __init__(self,
                 ophys_experiment_id,
                 dev_dff_path: Union[str, Path] = DFF_PATH,
                 dev_events_path: Union[str, Path] = EVENTS_PATH,
                 filtered_events_params: dict = None,
                 calc_new_dff_if_not_exist: bool = False,
                 **kwargs):
        self.inner = BehaviorOphysExperiment.from_lims(ophys_experiment_id,
                                                       **kwargs)
        self.calc_new_dff_if_not_exist = calc_new_dff_if_not_exist
        self.ophys_experiment_id = ophys_experiment_id
        self.metadata = self._update_metadata()
        # self.cell_x_gene = self._get_cell_x_gene() # TODO: implement
        self.is_roi_filtered = False
        self.filtered_events_params = filtered_events_params
        self.dff_path = Path(dev_dff_path)
        self.events_path = Path(dev_events_path)

        self.dff_traces = self._get_new_dff()

        try:
            self.events = self._get_new_events()
        except FileNotFoundError:
            # warn new_events not loaded
            # TODO: should we create one?
            print(f"No new_events file for ophys_experiment_id: "
                  f"{self.ophys_experiment_id}")

    def _get_new_dff(self):
        """Get new dff traces from pipeline_dev folder"""

        dff_path = self.dff_path
        # check if file exits, matching pattern "ophys_experiment_id_dff_*.h5"
        dff_fn = f"{self.ophys_experiment_id}_new_dff.h5"
        dff_file = []
        dff_file = list(dff_path.glob(dff_fn))
        if len(dff_file) == 0:
            # warn and create new dff
            print(f"No dff file for ophys_experiment_id: "
                  f"{self.ophys_experiment_id}, creating new one")

            if self.calc_new_dff_if_not_exist:
                dff_file = self._create_new_dff()
            else:
                raise FileNotFoundError(f"No dff file for ophys_experiment_id: "
                                        f"{self.ophys_experiment_id}")

        elif len(dff_file) > 1:
            raise FileNotFoundError((f">1 dff files for ophys_experiment_id"
                                     f"{self.ophys_experiment_id}"))
        else:
            dff_file = dff_file[0]

        # load dff traces, hacky because of dff trace is list in DataFrame
        # TODO: make into function
        with h5py.File(dff_file, "r") as f:
            # bit of code from dff_file.py in SDK
            traces = np.asarray(f['new_dff'], dtype=np.float64)
            roi_names = np.asarray(f['cell_roi_id'])
            idx = pd.Index(roi_names, name='cell_roi_id').astype('int64')
            new_dff = pd.DataFrame({'dff': [x for x in traces]}, index=idx)

        old_dff = self.inner.dff_traces.copy().reset_index()

        # merge on cell_roi_id
        updated_dff = (pd.merge(new_dff.reset_index(),
                                old_dff.drop(columns=["dff"]),
                                on="cell_roi_id", how="inner")
                       .set_index("cell_specimen_id"))

        return updated_dff

    def _get_new_events(self):
        """Get new events from pipeline_dev folder"""

        # TODO: remimplement versioning?
        # events_folder = f"oasis_nrsac_v{events_version}"  # CHANGE NEW ------>>>>>>>>>>>>>
        # version_folder = EVENTS_PATH / events_folder

        # # check version folder exists
        # if not version_folder.exists():
        #     version_folder = EVENTS_PATH / "oasis_nrsac_v1"
        #     print(f"Events version folder not found: {events_folder}, "
        #           f"defaulting to {version_folder}")

        events_file = self.events_path / f"{self.ophys_experiment_id}.h5"

        if not events_file.exists():
            raise FileNotFoundError(f"Events file not found: {events_file}")

        events_df = self._load_oasis_events_h5_to_df(events_file, self.filtered_events_params)

        return events_df

    def _load_oasis_events_h5_to_df(self,
                                    events_h5: str,
                                    filter_params: dict = None) -> pd.DataFrame:
        """Load h5 file from new_dff module

        Parameters
        ----------
        events_h5 : Path
            Path to h5 file
        filter_params : dict
            Keyword arguments to be passed to filter_events_array, if None
            use default values. See filter_events_array for details.

        Returns
        -------
        pd.DataFrame
            Dataframe with columns "cell_roi_id" and "events" and filtered events
        """
        # default filter params
        filter_scale_seconds = 0.065
        frame_rate_hz, _ = calculate_new_dff.get_correct_frame_rate(self.ophys_experiment_id)
        filter_n_time_steps = 20

        # check filter params for each key, if not present, use default
        if filter_params is not None:
            for key in ["filter_scale_seconds", "frame_rate_hz", "filter_n_time_steps"]:
                if key in filter_params:
                    locals()[key] = filter_params[key]

        with h5py.File(events_h5, 'r') as f:
            h5 = {}
            for key in f.keys():
                h5[key] = f[key][()]

        events = h5['spikes']

        filtered_events = \
            filter_events_array(arr=events,
                                scale=filter_scale_seconds * frame_rate_hz,
                                n_time_steps=filter_n_time_steps)

        dl = [[d] for d in events]  # already numpy arrays
        fe = [np.array(fe) for fe in filtered_events]
        df = pd.DataFrame(dl).rename(columns={0: 'events'})
        df['cell_roi_id'] = h5['cell_roi_id']
        df['filtered_events'] = fe

        # columns order
        df = df[['cell_roi_id', 'events', 'filtered_events']]

        # get dff trace for cell_specimen_id mapping to cell_roi_id
        dff = self.inner.dff_traces.copy().reset_index()
        df = (pd.merge(df, dff[["cell_roi_id", "cell_specimen_id"]],
                       on="cell_roi_id", how="inner").set_index("cell_specimen_id"))

        return df

    def _get_cell_x_gene(self):
        """Get cellXgene dataframe, if available"""

        # check if mouse_name = Copper
        if self.metadata["mouse_name"] == "Copper":

            # fn = "copper_r1_total_experiment_id_table_m15.xlsx"
            # ddf = pd.read_excel(CELLXGENE_PATH / fn, sheet_name=None)
            # gene_df = ddf['Transcriptomic profiles']
            # cell_df = ddf['in']
            # cell_df = cell_df[(cell_df['IOU_between_fov_and_z_stack'] > 0) & (cell_df['IOU_between_2p_and_ls'] > 0)]
            # cellxgene = cell_df.merge(gene_df, left_on='ls_cell_id', right_on='LS GCaMP Cell ID')

            # return all ls_cell_id per cell_specimen_id
            # lscounts = cellxgene.groupby('cell_specimen_id')['ls_cell_id'].apply(list).reset_index()

            fn = "copper_r1_total_experiment_id_table_m23_matt_removed_doublet_rois.xlsx"
            cellxgene = pd.read_excel(CELLXGENE_PATH / fn)

            return cellxgene

        else:
            return None

    def _update_metadata(self):
        """Update metadata, specifically correct ophsy_frame_rate"""
        metadata = self.inner.metadata.copy()
        dt = np.median(np.diff(self.ophys_timestamps))
        metadata["ophys_frame_rate"] = 1 / dt

        # mouse_name
        mouse_id = self.metadata["mouse_id"]
        id_map = etu.MOUSE_NAMES
        if str(mouse_id) in id_map.keys():
            mouse_name = id_map[str(mouse_id)]
        else:
            mouse_name = "Unknown"
        metadata["mouse_name"] = mouse_name
        return metadata

    def _create_new_dff(self):
        """Create new dff traces"""

        # get new dff DataFrame
        new_dff_df, timestamps = calculate_new_dff.get_new_dff_df(
            self.ophys_experiment_id)

        # Save as h5 file, because of the timestamps
        dff_file = calculate_new_dff.save_new_dff_h5(
            self.dev_dff_path, new_dff_df, timestamps, self.ophys_experiment_id)

        print(f"Created new_dff file at: {dff_file}")

        return dff_file

    # Delegate all else to the "inherited" BehaviorOphysExperiment object
    # Need attribute error to pickle/multiprocessing
    # see: https://stackoverflow.com/a/49380669
    def __getattr__(self, attr):
        if 'inner' not in vars(self):
            raise AttributeError
        return getattr(self.inner, attr)

    # # getstate/setstate for multiprocessing
    # # see: https://stackoverflow.com/a/50158865
    # def __getstate__(self):
    #     return self.inner

    # def __setstate__(self, state):
    #     print(state)
    #     self.inner = state
