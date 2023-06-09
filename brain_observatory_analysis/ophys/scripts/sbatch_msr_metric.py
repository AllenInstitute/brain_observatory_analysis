import argparse
import time
from simple_slurm import Slurm
from pathlib import Path

from brain_observatory_analysis.ophys.experiment_loading import start_lamf_analysis, get_recent_expts

# import ..suite2p_param_search

parser = argparse.ArgumentParser(description='running sbatch for new_dff_for_oeid.py')
parser.add_argument('--env-path', type=str, default='//data/learning/mattd/miniconda3/envs/dev', metavar='path to conda environment to use')
parser.add_argument('--dry_run', action='store_true', default=False, help='dry run')
parser.add_argument('--test_run', action='store_true', default=False, help='test one parameter set')
parser.add_argument('--event_type', type=str, default='changes', metavar="event type to use for dff: omissions, images, changes")
parser.add_argument('--data_type', type=str, default='dff', metavar="data type to use for dff: dff, events, filtered_events")
parser.add_argument('--metric_name', type=str, default='mean_response', metavar="metric name to use for stim response df")

ROOT_DIR = Path("/allen/programs/mindscope/workgroups/learning/analysis_data_cache/msr_metrics")


def load_ai210_lamf_expts():
    """"

    Mice: ['662264', '662680']
    # 662264

    """ 

    expt_table = get_recent_expts(date_after="2021-08-01",
                                  projects=["LearningmFISHDevelopment", "LearningmFISHTask1A"], 
                                  pkl_workaround=False)

    expt_table = add_dox_mice_and_columns(expt_table)

    lamf_210_nodox = expt_table[(expt_table["gcamp_name"] == "GCaMP7f") & (~expt_table["dox"])]

    return lamf_210_nodox


def add_dox_mice_and_columns(expt_table):
    lamf_210_dox = ['657850', '659231', '660433', '663051', '666073']
    lamf_195_dox = ['637848', '637851', '631563', '623975', '623972']

    # add dox column = True if mouse_id in dox list
    expt_table["dox"] = expt_table["mouse_id"].isin(lamf_210_dox + lamf_195_dox)

    return expt_table


if __name__ == '__main__':
    args = parser.parse_args()
    dry_run = args.dry_run
    test_run = args.test_run
    event_type = args.event_type
    data_type = args.data_type
    metric_name = args.metric_name
    python_executable = f"{args.env_path}/bin/python"

    # py file
    python_file = Path('//home/matt.davis/code/brain_observatory_analysis/brain_observatory_analysis/ophys/scripts/msr_metric.py')

    # job directory
    job_dir = ROOT_DIR
    stdout_location = job_dir / 'job_records'
    stdout_location.mkdir(parents=True, exist_ok=True)


    #expt_table = start_lamf_analysis(verbose=False)

    expt_table = load_ai210_lamf_expts()
    expt_ids = expt_table.index.values

    # oeids
    for event_type in ["images"]:
        for data_type in ["dff"]:


            if event_type == "omissions":
                short_session_types = ["Familiar Images + omissions", "Novel Images + omissions", "Novel Images EXTINCTION"]
                expt_table = expt_table[expt_table['short_session_type'].isin(short_session_types)]

            

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

                job_title = f'{oeid}_msr_metric'
                walltime = '00:25:00'
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

                args_string = f"--oeid {oeid} --event_type {event_type} --data_type {data_type}  --metric_name {metric_name}"
                print(args_string)

                sbatch_string = f"{python_executable} {python_file} {args_string}"
                print(sbatch_string)
                slurm.sbatch(sbatch_string)
                time.sleep(0.01)
