import pandas as pd
import numpy as np
from typing import Union
from pathlib import Path

from .experiment_loading import start_lamf_analysis, load_ophys_expts

# from mindscope_qc.data_access.behavior_ophys_experiment_dev import \
#     BehaviorOphysExperimentDev
# from allensdk.brain_observatory.behavior.behavior_ophys_experiment \
#     import BehaviorOphysExperiment


class ExperimentGroup():
    """Class to load and store experiments from a group of experiments

    Parameters
    ----------
    expt_table_to_load : pd.DataFrame, optional
        table of experiments to load, by default None
    filters : dict, optional
        key value pairs to filter expt_table, by default {}
    dev : bool, optional
        use BehaviorOphysExperimentDev object
    test_mode : bool, optional
        load only 2 experiments, by default False
    dev_dff_path : Union[str, Path], optional
        path to dev dff files, by default None
    dev_events_path : Union[str, Path], optional
        path to dev events files, by default None

    Attributes
    ----------
    dev : bool

    expt_list_to_load : list
        list of experiments to load
    filters : dict
        key value pairs to filter expt_table
    """

    def __init__(self,
                 expt_table_to_load: pd.DataFrame = None,
                 filters: dict = {},
                 dev: bool = False,
                 skip_eye_tracking: bool = False,
                 test_mode: bool = False,
                 dev_dff_path: Union[str, Path] = None,
                 dev_events_path: Union[str, Path] = None,):
        self.dev = dev
        self.expt_table_to_load = expt_table_to_load
        self.expt_list_to_load = self.expt_table_to_load.index.tolist()
        self.filters = filters
        self.test_mode = test_mode
        self.expt_list = []  # set after loading
        self.expt_table = pd.DataFrame()  # set after loading
        self.experiments = {}  # set after loading
        self.failed_to_load = []  # set after loading
        self.grp_ophys_cells_table = pd.DataFrame()
        self.skip_eye_tracking = skip_eye_tracking
        self.loaded = False

        if dev_dff_path is not None:
            self.dev_dff_path = Path(dev_dff_path)
        else:
            self.dev_dff_path = None

        if dev_events_path is not None:
            self.dev_events_path = Path(dev_events_path)
        else:
            self.dev_events_path = None

        if self.filters:
            # make sure each value in filters is a list
            self.filters = {k: [v] if not isinstance(v, (list, np.ndarray))
                            else v for k, v in self.filters.items()}
            self._filter_expt_table()

        if self.expt_table_to_load is None:
            self.expt_table_to_load = start_lamf_analysis()
            # TODO: change default or allow options

    def load_experiments(self):
        if self.test_mode:
            expt_list = self.expt_list_to_load[:2]
        else:
            expt_list = self.expt_list_to_load

        # if expt_list empty throw error
        if not expt_list:
            raise ValueError("No experiments to load (filter likely failed)")

        self.experiments = \
            load_ophys_expts(expt_list,
                             multi=True,
                             return_failed=False,
                             dev=self.dev,
                             dev_dff_path=self.dev_dff_path,
                             dev_events_path=self.dev_events_path,
                             skip_eye_tracking=self.skip_eye_tracking)
        self._remove_extra_failed()

        # In case where cell_specimen_ids should be corrected later?
        self._get_ophys_cells_table()

        self.expt_table = self._expt_table_loaded()
        self.expt_list = self.expt_table.index.tolist()
        self.loaded = True

    def sample_experiment(self):
        """Just get 1st experiment from dict"""
        return list(self.experiments.values())[0]

    def _filter_expt_table(self):
        """Filter expt_table_to_load by key value pairs in self.filters,
        called when initilized

        TODO: consider making postload filter"""

        # make local so can filter on `ophys_experiment_id` index
        local_expt_table = self.expt_table_to_load.reset_index()
        table_cols = local_expt_table.columns
        assert all([x in table_cols for x in self.filters.keys()])

        # boolean index to filter experiment table by self.filters dict
        filter_index = pd.Series([True] * len(self.expt_table_to_load))
        for key, value in self.filters.items():
            print(key, value)
            filter_index = filter_index & local_expt_table[key].isin(
                value).values

        print(f"Found {sum(filter_index)} experiments matching filters")

        self.expt_table_to_load = (self.expt_table_to_load[filter_index.values]
                                   .sort_values(by="date_of_acquisition"))
        self.expt_list_to_load = self.expt_table_to_load.index.tolist()

    def _check_for_duplicates(self):
        """Check for duplicate experiments in expt_table_full"""
        # check for duplicates
        if self.expt_table_to_load.index.duplicated().any():
            print("Duplicate experiments in expt_table_full")
            print(
                self.expt_table_to_load[self.expt_table_to_load.index.duplicated()])
            raise ValueError

    def _expt_table_loaded(self):
        """Select only experiments that were loaded"""
        return self.expt_table_to_load.loc[self.experiments.keys()]

    def get_all_expts_session_type(self, session_type: list):
        # TODO THINK HOW TO BEST DO THIS
        """Get all experiments of a given list of session types"""
        et = self.expt_table.query("session_type in @session_type")
        # filter self.experiments to only include expts
        expts = {k: v for k, v in self.experiments.items() if k in et.index}

        # return new ExperimentGroup object, with new experiments
        return ExperimentGroup(self.name, self.mouse_ids, self.dir_name,
                               test_mode=self.test_mode,
                               expt_table=et, expt_list=et.index.tolist(),
                               experiments=expts)

        return expts

    def _get_passive_expts(self, expt_table: pd.DataFrame):
        # passive_session = 'OPHYS_2_images_A_passive'
        # qc_states = ["passed", "qc"]  # get clarity on if "qc"

        # query for mouse_ids and get counts of qc states x mouseid
        mouse_id_counts = (expt_table.query("mouse_id in @mouse_ids")
                           .groupby(["mouse_id", "experiment_workflow_state"])
                           .size())
        print(mouse_id_counts, "\n")

        query = ("mouse_id in @mouse_ids"
                 "& session_type == @passive_session"
                 "& experiment_workflow_state in @qc_states")
        expts = expt_table.query(query)

        # d = partial(nu.check_expt_params, verbose = True)
        # expts["new_params"] = nu.apply_func_df(expts, d)

        expts["cohort"] = self.name  # TODO: PUT IN COHORT CLASS

        # TODO: explore duplicates in ai210 especially
        (expts.drop(columns=["driver_line", "ophys_container_id"])
              .drop_duplicates())
        return expts

    def _remove_extra_failed(self):
        """Experiment loading funtion still allows None returned,
        move those to failed list"""
        # add None to failed
        self.failed_to_load.extend([x for x in self.experiments if x is None])

        # remove None values from experiement dict
        self.experiments = {k: v for k, v in self.experiments.items()
                            if v is not None}

    def _get_ophys_cells_table(self):
        """Get ophys cells table for all experiments in group"""

        cell_tables = []
        if self.grp_ophys_cells_table.empty:
            for expt in self.experiments.values():
                cell_tables.append(expt.cell_specimen_table)
        self.grp_ophys_cells_table = pd.concat(cell_tables)

    def stack_traces_from_expt_grp(self, data_type: str = "dff", return_crid: bool = False):
        """ Return all traces from each experiment stacked as numpy array

        Only works if all experiments are from the same ophys_experiment_session_id

        Parameters
        ----------
        data_type : str, optional
            Type of data to return, by default "dff"
            Options: "dff", "events", "filtered_events"
        return_crid : bool, optional
            Return cell_roi_id index, by default False

        Returns
        -------
        np.array
        """

        # if not loaded, load expts
        if not self.loaded:
            self.load_experiments()

        assert len(self.expt_table["ophys_session_id"].unique()) == 1, \
            "expt_table has more than one ophys_session_id, can't stack traces"

        if data_type == "dff":
            attr_key = f"{data_type}_traces"
        elif data_type in ["events", "filtered_events"]:
            attr_key = f"{data_type}"

        col_key = data_type  # "dff", "events", "filtered_events"

        traces = []
        for oeid, expt in self.experiments.items():
            try:
                trace = getattr(expt, attr_key).copy()
            except KeyError:
                # warn
                print(f"WARNING: no {attr_key} for {oeid}, returning blank")
                trace = pd.DataFrame(columns=[col_key])

            traces.append(trace)

        df = pd.concat(traces, axis=0)

        # grab each list from dff and make array
        traces_array = np.array(df[col_key].tolist())
        cell_roi_ids = df.cell_roi_id.values

        if return_crid:
            return traces_array, cell_roi_ids
        else:
            return traces_array
