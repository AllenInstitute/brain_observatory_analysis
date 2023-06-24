import os
import argparse
import time
from simple_slurm import Slurm
from pathlib import Path
# from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache as bpc
from brain_observatory_analysis.dev import ophys_cache_util
from brain_observatory_analysis.ophys.experiment_loading import start_gh_analysis  # start_lamf_analysis, start_vb_analysis
from brain_observatory_qc.data_access import behavior_ophys_experiment_dev as boe_dev 

parser = argparse.ArgumentParser(
    description='running sbatch for generate_response_df.py')
parser.add_argument('--env-path', type=str, default='/home/jinho.kim/anaconda3/envs/allenhpc',
                    metavar='path to conda environment to use')


if __name__ == '__main__':
    args = parser.parse_args()
    python_executable = "{}/bin/python".format(args.env_path)
    print('python executable = {}'.format(python_executable))
    base_dir = Path(
        '/home/jinho.kim/Github/brain_observatory_analysis/brain_observatory_analysis/ophys/scripts/examples/')
    python_file = base_dir / 'generate_response_df.py'
    if os.path.isfile(python_file) is False:
        raise FileNotFoundError(
            'python file {} does not exist'.format(python_file))

    # Set directories
    dff_path = boe_dev.GH_DFF_PATH
    events_path = boe_dev.GH_EVENTS_PATH
    cache_dir_base = ophys_cache_util.GH_RESPONSE_DF_DIR

    # Get ophys experiments to run
    exp_table = start_gh_analysis()
    oeid_list = exp_table.index.values

    stdout_location = cache_dir_base / 'job_records'
    stdout_location.mkdir(parents=True, exist_ok=True)

    job_count = 0

    rerun = True
    for oeid in oeid_list:
        job_count += 1
        print('starting cluster job for {}, job count = {}'.format(
            oeid, job_count))
        job_title = 'ophys_experiment_id_{}'.format(oeid)
        walltime = '0:30:00'
        cpus_per_task = 1
        mem = '50gb'
        job_id = Slurm.JOB_ARRAY_ID
        job_array_id = Slurm.JOB_ARRAY_MASTER_ID
        output = stdout_location / f'{job_array_id}_{job_id}_{oeid}.out'

        # instantiate a SLURM object
        slurm = Slurm(
            cpus_per_task=cpus_per_task,
            job_name=job_title,
            time=walltime,
            mem=mem,
            output=output,
            partition="braintv"
        )

        args_string = f'{oeid} {dff_path} {events_path} {cache_dir_base}'
        slurm.sbatch('{} {} {}'.format(
            python_executable,
            python_file,
            args_string,
        )
        )
        time.sleep(0.01)
