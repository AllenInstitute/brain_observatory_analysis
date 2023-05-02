import argparse
import time
from simple_slurm import Slurm
from pathlib import Path

from brain_observatory_analysis.ophys.experiment_loading import start_lamf_analysis

# import ..suite2p_param_search

parser = argparse.ArgumentParser(description='running sbatch for new_dff_for_oeid.py')
parser.add_argument('--env-path', type=str, default='//data/learning/mattd/miniconda3/envs/dev', metavar='path to conda environment to use')
parser.add_argument('--dry_run', action='store_true', default=False, help='dry run')
parser.add_argument('--test_run', action='store_true', default=False, help='test one parameter set')

parser.add_argument('--event_type', type=str, default='changes', metavar="event type to use for dff: omissions, images, changes", required=True)
parser.add_argument('--data_type', type=str, default='dff', metavar="data type to use for dff: dff, events, filtered_events", required=True)

ROOT_DIR = Path("/allen/programs/mindscope/workgroups/learning/analysis_data_cache/stim_response_df_nrsac")

if __name__ == '__main__':
    args = parser.parse_args()
    dry_run = args.dry_run
    test_run = args.test_run
    event_type = args.event_type
    data_type = args.data_type
    python_executable = f"{args.env_path}/bin/python"

    # py file
    python_file = Path('//home/matt.davis/code/brain_observatory_analysis/brain_observatory_analysis/ophys/scripts/gen_stim_response_df.py')

    # job directory
    job_dir = ROOT_DIR
    stdout_location = job_dir / 'job_records'
    stdout_location.mkdir(parents=True, exist_ok=True)

    # oeids
    for event_type in ["changes"]:
        for data_type in ["events", "filtered_events"]:
            expt_table = start_lamf_analysis(verbose=False)

            if event_type == "omissions":
                short_session_types = ["Familiar Images + omissions", "Novel Images + omissions", "Novel Images EXTINCTION"]
                expt_table = expt_table[expt_table['short_session_type'].isin(short_session_types)]

            expt_ids = expt_table.index.values

            if dry_run:
                print('dry run, exiting')
                exit()
            # bash folder size
            
            job_count = 0
            print('number of jobs = {}'.format(len(expt_ids)))

            if test_run:
                expt_ids = expt_ids[0:1]

            print('Total number of jobs = {}'.format(len(expt_ids)))

            for oeid in expt_ids:

                job_count += 1
                print(f'starting cluster job for {oeid}, job count = {job_count}')

                job_title = f'{oeid}_gen_stim_response_df'
                walltime = '00:15:00'
                mem = '2G'
                # tmp = '3G',
                job_id = Slurm.JOB_ARRAY_ID
                job_array_id = Slurm.JOB_ARRAY_MASTER_ID
                output = stdout_location / f'{job_array_id}_{job_id}_{oeid}.out'
                cpus_per_task = 1
                print(output)

                # instantiate a SLURM object
                slurm = Slurm(
                    cpus_per_task=cpus_per_task,
                    job_name=job_title,
                    time=walltime,
                    mem=mem,
                    output=output,
                    # tmp=tmp,
                    partition="braintv"
                )

                args_string = f"--oeid {oeid} --event_type {event_type} --data_type {data_type}"
                print(args_string)

                sbatch_string = f"{python_executable} {python_file} {args_string}"
                print(sbatch_string)
                slurm.sbatch(sbatch_string)
                time.sleep(0.01)
