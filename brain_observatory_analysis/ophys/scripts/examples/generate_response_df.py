from pathlib import Path
from brain_observatory_analysis.dev import ophys_cache_util
from brain_observatory_qc.data_access import behavior_ophys_experiment_dev as boe_dev
from brain_observatory_qc.data_access.behavior_ophys_experiment_dev import \
     BehaviorOphysExperimentDev

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('oeid', type=int, default=None,
                    metavar='ophys experiment id')
parser.add_argument('dff_path', type=str, default=None,
                    metavar='path to dff files')
parser.add_argument('events_path', type=str, default=None,
                    metavar='path to event files')
parser.add_argument('cache_dir_base', type=str, default=None,
                    metavar='path to cache directory base')

if __name__ == "__main__":
    args = parser.parse_args()
    oeid = args.oeid
    dff_path = args.dff_path
    event_path = args.events_path
    cache_dir_base = Path(args.cache_dir_base)
    cache_dir_dff = cache_dir_base / 'dff'
    cache_dir_events = cache_dir_base / 'events'

    exp = BehaviorOphysExperimentDev(oeid, dev_dff_path=dff_path, dev_events_path=event_path)

    # response_df = ophys_cache_util.get_stim_annotated_response_df(exp, 'dff', 'all', cache_dir=cache_dir_dff)
    response_df = ophys_cache_util.get_stim_annotated_response_df(exp, 'events', 'all', cache_dir=cache_dir_events, output_sampling_rate=None)
