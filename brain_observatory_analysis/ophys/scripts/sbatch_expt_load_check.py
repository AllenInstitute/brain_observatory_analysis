import argparse
import time
from simple_slurm import Slurm
from pathlib import Path
import pandas as pd

from brain_observatory_analysis.ophys.experiment_loading import start_lamf_analysis, get_recent_expts

# import ..suite2p_param_search

parser = argparse.ArgumentParser(description='running sbatch for new_dff_for_oeid.py')
parser.add_argument('--env-path', type=str, default='//data/learning/mattd/miniconda3/envs/dev', metavar='path to conda environment to use')
parser.add_argument('--dry_run', action='store_true', default=False, help='dry run')
parser.add_argument('--test_run', action='store_true', default=False, help='test one parameter set')
parser.add_argument('--event_type', type=str, default='changes', metavar="event type to use for dff: omissions, images, changes")
parser.add_argument('--data_type', type=str, default='dff', metavar="data type to use for dff: dff, events, filtered_events")

ROOT_DIR = Path("/allen/programs/mindscope/workgroups/learning/analysis_data_cache/expt_load_check")


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


def load_lamf_sample():
    expt_table = start_lamf_analysis(verbose=False)

    # filter by gcamp_name = GCaMP7s
    expt_table = expt_table[expt_table["gcamp_name"] == "GCaMP7s"]

    # short session type =['Familiar Images + omissions', 'Novel Images + omissions', 'Novel Images EXTINCTION']
    expt_table = expt_table[expt_table["short_session_type"].isin(['Familiar Images + omissions', 'Novel Images + omissions', 'Novel Images EXTINCTION'])]
    expt_table = expt_table.sample(100, random_state=42)

    return expt_table


def load_dox_expts():
    expt_table = get_recent_expts(date_after="2021-08-01", 
                                  projects=["LearningmFISHDevelopment", "LearningmFISHTask1A"],
                                  pkl_workaround=False)

    expt_table = add_dox_mice_and_columns(expt_table)

    lamf_210_dox = expt_table[(expt_table["gcamp_name"] == "GCaMP7f") & (expt_table["dox"])]
    lamf_195_dox = expt_table[(expt_table["gcamp_name"] == "GCaMP7s") & (expt_table["dox"])]

    return lamf_210_dox, lamf_195_dox

if __name__ == '__main__':
    args = parser.parse_args()
    dry_run = args.dry_run
    test_run = args.test_run
    event_type = args.event_type
    data_type = args.data_type
    python_executable = f"{args.env_path}/bin/python"

    # py file
    python_file = Path('//home/matt.davis/code/brain_observatory_analysis/brain_observatory_analysis/ophys/scripts/expt_load_check.py')

    # job directory
    job_dir = ROOT_DIR
    stdout_location = job_dir / 'job_records'
    stdout_location.mkdir(parents=True, exist_ok=True)

    # expt_table = start_lamf_analysis(verbose=False)

    e1 = load_ai210_lamf_expts()
    e2 = load_lamf_sample()
    e3, e4 = load_dox_expts()
    expt_table = pd.concat([e1, e2, e3, e4])

    expt_ids = expt_table.index.values

    failed_oeids = [1197394868,
    1197394870,
    1197615447,
    1197615440,
    1197615446,
    1197615450,
    1197615443,
    1197615449,
    1197615452,
    1197615442,
    1197830384,
    1197830366,
    1214288626,
    1214288635,
    1214288636,
    1214288631,
    1214288638,
    1214288633,
    1214288628,
    1214471941,
    1214471943,
    1214471935,
    1214471938,
    1214471944,
    1214471940,
    1214471946,
    1214471937,
    1215022338,
    1215707434,
    1215707433,
    1215707431,
    1215707437,
    1216211052,
    1216211050,
    1216211059,
    1216211055,
    1216211062,
    1216211058,
    1216211056,
    1246206304,
    1246206306,
    1246206293,
    1246206298,
    1246206303,
    1246206299,
    1246206296,
    1246484973,
    1246484983,
    1246484971,
    1246484977,
    1246484974,
    1246484980,
    1246484979,
    1247382943,
    1247382949,
    1247382952,
    1247382946,
    1247382951,
    1247382945,
    1247382948,
    1247382954,
    1248768955,
    1248768956,
    1248768953,
    1248768976,
    1248768960,
    1248768963,
    1248124604,
    1248124611,
    1248124614,
    1248124602,
    1248124608,
    1248124607,
    1248124605,
    1248122201,
    1248122203,
    1248122206,
    1248122196,
    1248122200,
    1248122198,
    1248122204,
    1248122194,
    1254931521,
    1254931530,
    1254931527,
    1255888001,
    1255887995,
    1255888004,
    1255888002,
    1255887996,
    1257942954,
    1257942955,
    1257942961,
    1257942952,
    1257942958,
    1257942957,
    1257940603,
    1257940609,
    1257940600,
    1257940611,
    1257940606,
    1258552586,
    1258552588,
    1258552585,
    1258552583,
    1258552589,
    1266500136,
    1267819016,
    1267819017,
    1267819019,
    1267819009,
    1267819022,
    1267819013,
    1267819014,
    1267819020]

    job_count = 0
    print(f'Total number of jobs = {expt_ids}')

    for oeid in expt_ids:

        job_count += 1
        print(f'starting cluster job for {oeid}, job count = {job_count}')

        job_title = f'{oeid}_expt_check'
        walltime = '00:15:00'
        mem = '2G'
        job_id = Slurm.JOB_ARRAY_ID
        job_array_id = Slurm.JOB_ARRAY_MASTER_ID
        output = stdout_location / f'{job_array_id}_{job_id}_{oeid}.out'
        cpus_per_task = 1

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

        args_string = f"--oeid {oeid}"
        print(args_string)

        sbatch_string = f"{python_executable} {python_file} {args_string}"
        print(sbatch_string)
        slurm.sbatch(sbatch_string)
        time.sleep(0.01)
