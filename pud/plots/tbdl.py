import argparse
import pickle
from pathlib import Path
from typing import List

import numpy as np
from dotmap import DotMap
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tqdm.auto import tqdm
from termcolor import cprint

def get_slurm_jobid_from_filename(filename: str):
    """
    my experiment job root has the name format:
    ALGONAME_job_SLURMID_COMMENT
    this function extracts the digit sequence after job
    """
    if 'job' in filename:
        job_index = filename.index('job')
        slurm_digits = ""
        for ch in filename[job_index:]:
            if ch.isdigit():
                slurm_digits += ch
            elif len(slurm_digits) > 0:
                return slurm_digits
    else:
        return None

def find_tb_log_dirs(top_dir:str, tb_file_header:str='events.out.tfevents'):
    """
    get all tensorbord log directories that contain tensorboard event/log files
    """
    if isinstance(top_dir, str):
        top_dir = Path(top_dir)
    tb_log_dirs = []
    for path in top_dir.rglob('{}*'.format(tb_file_header)):
        tb_log_dirs.append(path.parent)
    tb_log_dirs = np.unique(tb_log_dirs).tolist()
    return tb_log_dirs

class TBDataLoader (object):
    def __init__(self, log_dirs:List[Path]):
        super(TBDataLoader, self).__init__()
        self.log_dirs = log_dirs
        
        # init event accumulators for log dirs
        # key: logdir object, val: event accumulator
        self.event_accumulators = {}
        self.init_all(reload=False)
        self.data = DotMap()
        self.keys = DotMap()
    
    def get_scalar_keys(self):
        for ld in self.event_accumulators:
            ea = self.event_accumulators[ld]
            self.keys.scalars[ld] = ea.scalars.Keys()

    def init_all(self, reload=True, verbose=True):
        if verbose:
            print('[INFO] reloading from all tensorboard dirs ...')
        for logdir in tqdm(self.log_dirs, total=len(self.log_dirs), desc="Initialize Event Accumulators ..."):
            self.event_accumulators[logdir.as_posix()]:EventAccumulator  = EventAccumulator(path=logdir.as_posix())
            if reload:
                self.event_accumulators[logdir.as_posix()].Reload()

    def get_all_scalars(self, reload=False):
        for ld in tqdm(self.event_accumulators, total=len(self.event_accumulators.keys()), desc="Getting Scalars..."):
            ea = self.event_accumulators[ld]
            if reload:
                ea.Reload()
            for tag in ea.scalars.Keys():
                cur_event = ea.Scalars(tag)
                self.data.scalars[tag][ld].step = [x.step for x in cur_event]
                self.data.scalars[tag][ld].value = [x.value for x in cur_event]

    def get_scalars_from_tag(self, tag:str):
        for ld in self.event_accumulators:
            ea = self.event_accumulators[ld]
            cur_event = ea.Scalars(tag)
            self.data.scalars[tag][ld].step = [x.step for x in cur_event]
            self.data.scalars[tag][ld].value = [x.value for x in cur_event]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--top-dir', type=str, default="runs/Cheetah_Pareto_BTLS_no_solve", help='top dir to parse tensorborad events')
    parser.add_argument('--out-dir', type=str, default="outputs", help='')
    parser.add_argument('--parse_job_id', action='store_true', help='parse slurm job id from the input dir name or up one parent dir')
    args = parser.parse_args()

    ## get slurm job id
    top_dir = Path(args.top_dir)
    slurm_job_id = ""
    if 'job' in top_dir.name:
        slurm_job_id = get_slurm_jobid_from_filename(top_dir.name)
    elif 'job' in top_dir.parent.name:
        slurm_job_id = get_slurm_jobid_from_filename(top_dir.parent.name)
    
    if len(slurm_job_id) > 0:
        cprint('Found SLURM job id: {}'.format(slurm_job_id), 'green')

    ## get all dirs that contain tensorboard events
    log_dirs = find_tb_log_dirs(args.top_dir)
    tb_l = TBDataLoader(log_dirs)

    #tb_l.reload_all()
    tb_l.get_all_scalars(reload=True)

    top_dir = Path(args.top_dir)
    dump_dir = Path(args.out_dir)

    dump_file = dump_dir.joinpath("{}.pickle".format(top_dir.name))
    if args.parse_job_id and len(slurm_job_id) > 0:
        dump_file = dump_dir.joinpath("{}_job{}.pickle".format(top_dir.name, slurm_job_id))

    with open(dump_file, 'wb') as f:
        pickle.dump(tb_l.data.toDict(), f)
