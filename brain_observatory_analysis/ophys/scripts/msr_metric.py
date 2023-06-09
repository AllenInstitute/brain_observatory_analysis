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

# metric_name
parser.add_argument('--metric_name', type=str, default='mean_response', required=True,
                    metavar='metric_name')

# metrics foldder path

METRICS_FOLDER = Path("/allen/programs/mindscope/workgroups/learning/analysis_data_cache/msr_metrics")
# main
if __name__ == '__main__':
    args = parser.parse_args()

    oeid = args.oeid
    event_type = args.event_type
    data_type = args.data_type
    metric_name = args.metric_name

    expt = get_ophys_expt(oeid, dev=True)

    if expt is not None:
        msr_df = sr._get_mean_stim_response_df(expt,
                                               event_type=event_type,
                                               data_type=data_type,
                                               load_from_file=False,
                                               save_to_file=False)
    else:
        quit()

    assert metric_name in msr_df.columns, f"{metric_name} not in columns"

    metric_df = msr_df[["ophys_experiment_id", "cell_roi_id", metric_name]]
    metric_df.to_csv(METRICS_FOLDER / f"{oeid}_{event_type}_{data_type}_{metric_name}.csv", index=False)
    print(f"saved {METRICS_FOLDER / f'{oeid}_{event_type}_{data_type}_{metric_name}.csv'}")
