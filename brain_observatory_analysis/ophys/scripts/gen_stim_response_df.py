# argparser
import argparse
parser = argparse.ArgumentParser(description='')


from brain_observatory_analysis.ophys.experiment_loading import get_ophys_expt
import brain_observatory_analysis.ophys.stimulus_response as sr

parser.add_argument('--oeid', type=int, default=None,
                    metavar='oeid')

# add event_type
parser.add_argument('--event_type', type=str, default='changes',
                    metavar='event_type')

# add data_type
parser.add_argument('--data_type', type=str, default='dff',
                    metavar='data_type')

# main
if __name__ == '__main__':
    args = parser.parse_args()

    oeid = args.oeid
    event_type = args.event_type
    data_type = args.data_type


    expt = get_ophys_expt(oeid)


    if expt is not None:
        sr_df = sr._get_stimulus_response_df(expt,
                                                event_type=event_type,
                                                data_type=data_type,
                                                load_from_file=False,
                                                save_to_file=True)
    else:
        print(f'failed to generate stim response df for {oeid}')


    

