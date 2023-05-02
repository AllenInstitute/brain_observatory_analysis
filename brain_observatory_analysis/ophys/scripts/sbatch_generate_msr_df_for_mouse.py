import argparse
import time
from simple_slurm import Slurm
from pathlib import Path
import pandas as pd
from brain_observatory_analysis.ophys.experiment_loading import start_lamf_analysis

ROOT_DIR = Path("//allen/programs/mindscope/workgroups/learning/analysis_data_cache/msrdf")

if __name__ == '__main__':
    python_executable = "//home/iryna.yavorska/anaconda3/envs/mfish_glm/bin/python"

    # py file
    python_file = Path('//home/iryna.yavorska/code/brain_observatory_analysis/brain_observatory_analysis/ophys/scripts/generate_msr_df_for_mouse.py')  # update to brain_observatory_qc
    job_dir = ROOT_DIR 
    stdout_location = job_dir / 'job_records'
    stdout_location.mkdir(parents=True, exist_ok=True)

    expts = start_lamf_analysis()

    # Copper
    # mouse_names = ["Copper", "Silicon", "Titanium", "Bronze", "Gold", "Silver", "Mercury", "Aluminum", "Iron", "Cobalt"]
    
    expts.mouse_name = pd.Categorical(expts.mouse_name, categories=mouse_names, ordered=True)
    expts = expts.sort_values(by="mouse_name")
    expt_ids = expts.index.values
    # TODO: called python script just for copper

    job_count = 0
    print('number of jobs = {}'.format(len(expt_ids)))
    
    mouse_names = ["Silver", "Gold", "Silicon"]
    data_types = ['dff', 'events', 'filtered_events']
    event_types = ["changes", "omissions"]

    for mouse_name in mouse_names:
        for data_type in data_types:
            for event_type in event_types:
                job_count += 1
                print(f'CLUSTER JOB: {event_type} {data_type}, {mouse_name}, job count = {job_count}')

                job_title = f'msr_df_{event_type}_{data_type}_{mouse_name}'
                walltime = '01:00:00'
                mem = '5G'
                # tmp = '3G',
                job_id = Slurm.JOB_ARRAY_ID
                job_array_id = Slurm.JOB_ARRAY_MASTER_ID
                output = stdout_location / f'{job_array_id}_{job_id}.out'
                cpus_per_task = 32
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

                # args_string = f"--h5_path {h5} --hpc"
                args_string = f"--event_type {event_type} --data_type {data_type} --mouse_name {mouse_name} "
                print(args_string)

                sbatch_string = f"{python_executable} {python_file} {args_string}"
                print(sbatch_string)
                slurm.sbatch(sbatch_string)
                time.sleep(0.01)
