# argparser
import argparse
parser = argparse.ArgumentParser(description='')
from pathlib import Path


from brain_observatory_analysis.ophys.experiment_loading import get_ophys_expt
import brain_observatory_analysis.ophys.stimulus_response as sr

parser.add_argument('--oeid', type=int, default=None,
                    metavar='oeid')

# add event_type
parser.add_argument('--event_type', type=str, default='changes', required=True,
                    metavar='event_type')

# add data_type
parser.add_argument('--data_type', type=str, default='dff', required=True,
                    metavar='data_type')


MSR_DF_FOLDER = Path("/allen/programs/mindscope/workgroups/learning/analysis_data_cache/msr_df")

def fill_csid_with_dummy(expt):
    """ replaces csid with dummy index"""

    if expt.dff_traces.index.isnull().all():
        for key in ["dff_traces", "cell_specimen_table", "roi_masks", "corrected_fluorescence_traces"]:

            # check each key exists
            if hasattr(expt, key):
                print(key)

                # get attrbute and set the index to the row number
                attr = getattr(expt, key)
                attr = attr.reset_index(drop=True)
                attr.index.name = "cell_specimen_id"

                #expt.__setattr__(key, attr)  # use class setter
                setattr(expt, key, attr)

    return expt

# main
if __name__ == '__main__':
    args = parser.parse_args()

    oeid = args.oeid
    event_type = args.event_type
    data_type = args.data_type

    expt = get_ophys_expt(oeid, dev=True)
    expt = fill_csid_with_dummy(expt)

    if expt is not None:
        msr_df = sr._get_mean_stim_response_df(expt,
                                               event_type=event_type,
                                               data_type=data_type,
                                               load_from_file=False,
                                               save_to_file=False)
    else:
        quit()

    # save as hdf5
    msr_df.to_hdf(MSR_DF_FOLDER / f"{oeid}_{event_type}_{data_type}.h5", key="data", mode="w")
