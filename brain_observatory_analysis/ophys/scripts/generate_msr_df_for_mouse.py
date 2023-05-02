

from brain_observatory_analysis.ophys.experiment_loading import start_lamf_analysis
from brain_observatory_analysis.ophys.experiment_group import ExperimentGroup

import brain_observatory_analysis.ophys.stimulus_response as sr
# (add mjd_dev to path)
import sys
sys.path.append("//home/iryna.yavorska/code/brain_observatory_analysis/brain_observatory_analysis/ophys/scripts")
import coreg_utils as cu

import stim_response_plots as stp

from pathlib import Path
# args parser
import argparse

# event_type arg
parser = argparse.ArgumentParser()
parser.add_argument("--event_type", type=str, default="changes")
parser.add_argument("--data_type", type=str, default="filtered_events")

def fill_csid_with_rows(expt_group):

    for id, expt in expt_group.experiments.items():
        if expt.dff_traces.index.isnull().all():
            # replaces with row number
            # assume all keys null
            for key in ["dff_traces", "events", "cell_specimen_table", "roi_masks"]:
                
                # get attrbute and set the index to the row number
                attr = getattr(expt, key)
                #attr.index = attr.index.reset_index(drop=True)
                attr = attr.reset_index(drop=True)

                # reset index name
                attr.index.name = "cell_specimen_id"
                

                # set the attribute
                setattr(expt, key, attr)


# main
if __name__ == "__main__":
    args = parser.parse_args()
    event_type = args.event_type
    data_type = args.data_type

    expt_table = start_lamf_analysis(verbose=False)

    filters = {"mouse_name": "Copper", "targeted_structure": "VISp"}
    expt_group = ExperimentGroup(expt_table_to_load=expt_table,
                                 filters=filters, dev=True, group_name="copper")
    expt_group.load_experiments()
    # cu.add_missing_csid_expt_grp(expt_group)
    fill_csid_with_rows(expt_group)

    msrdf_path = Path("//allen/programs/mindscope/workgroups/learning/analysis_data_cache/msrdf")
    data_types = ['filtered_events']
    event_types = ["omissions", "changes"]

    for data_type in data_types:
       for event_type in event_types:
            msr_df = sr.get_mean_stimulus_response_expt_group(expt_group,
                                                                    event_type=event_type,
                                                                    data_type=data_type,
                                                                    load_from_file=False,
                                                                    save_to_file=False,
                                                                    multi=True,
                                                                    save_expt_group_msrdf = msrdf_path)




