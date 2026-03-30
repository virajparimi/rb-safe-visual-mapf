"""
Generate CBFS grid policies through multi-processing]
"""

import os
import yaml
import pickle
import argparse
import numpy as np
import networkx as nx
from pathlib import Path
from dotmap import DotMap
from tqdm.auto import tqdm
import multiprocessing as mp

from pud.algos.cbfs.cbfs_eval import CBFS
from pud.envs.safe_pointenv.safe_pointenv import SafePointEnv


def linspace_list(inp_list: list, N: int):
    """
    Divide the inp_list evenly into N shares
    """
    split_inds = np.linspace(0, len(inp_list), N + 1, dtype=int)
    inp_list_split = []
    for i in range(N):
        ind_1 = split_inds[i]
        ind_2 = split_inds[i + 1]
        inp_list_split.append(inp_list[ind_1:ind_2])
    return inp_list_split


def setup(env_kwargs, cost_f_kwargs):
    p_env = SafePointEnv(**env_kwargs, cost_f_args=cost_f_kwargs)

    walls = p_env._walls
    cost_limit = env_kwargs["cost_limit"]
    (height, width) = walls.shape
    g = nx.Graph()
    # Add all the nodes
    for i in range(height):
        for j in range(width):
            if walls[i, j] == 0:
                g.add_node((i, j))

    # Add all the edges
    # Edges over the cost limit is guaranteed not to be feasible
    for i in range(height):
        for j in range(width):
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == dj == 0:
                        continue  # Don't add self loops
                    if i + di < 0 or i + di > height - 1:
                        continue  # No cell here
                    if j + dj < 0 or j + dj > width - 1:
                        continue  # No cell here
                    if walls[i, j] == 1:
                        continue  # Don't add edges to walls
                    if walls[i + di, j + dj] == 1:
                        continue  # Don't add edges to walls
                    # Filtering by cost map
                    if p_env._cost_map[i, j] > cost_limit:
                        continue
                    if p_env._cost_map[i + di, j + dj] > cost_limit:
                        continue
                    g.add_edge((i, j), (i + di, j + dj))

    g_dict = nx.to_dict_of_dicts(G=g)
    return g_dict, p_env


def run_CBFS(
    root, g_dict: dict, cost_map: np.ndarray, cost_limit: float, output_dir: Path
):
    assert len(root) == 2
    out, out_cost = CBFS(g=g_dict, root=root, cost_limit=cost_limit, cost_map=cost_map)
    output_file = output_dir.joinpath(
        "root={},{}_cost_limit={}.pkl".format(root[0], root[1], cost_limit)
    )
    save_data = {
        "trajs": out,  # list
        "costs": out_cost,  # list
    }
    f = open(output_file, "wb")
    pickle.dump(save_data, f)
    f.close()


def mp_runner(kwargs):
    """Multi-processing helper"""
    return run_CBFS(**kwargs)


if __name__ == "__main__":
    """
    python pud/algos/cbfs_mp.py --cfg=configs/config_SafePointEnv.yaml --outdir=pud/envs/precompiles/mptest2 --use_mp
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--load_ratio", type=float, default=0.8, help="ratio to abuse the machine"
    )

    parser.add_argument("--outdir", type=str, help="")

    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/config_SafePointEnv.yaml",
        help="env/training configuration",
    )

    parser.add_argument("--use_mp", action="store_true", help="use multi-processing")
    parser.add_argument(
        "--pbar", action="store_true", help="show progress bar in single-process mode"
    )

    parser.add_argument("--debug", action="store_true", help="disable all file writing")

    args = parser.parse_args()

    cfg = {}
    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = DotMap(cfg)
    cfg.pprint()

    cpu_load_ratio = args.load_ratio
    num_procs = int(os.cpu_count() * cpu_load_ratio)

    mp_out_dir = Path(args.outdir)
    if not args.debug:
        mp_out_dir.mkdir(parents=True, exist_ok=True)

    env_cfgs = cfg.env.toDict()
    cost_f_cfgs = cfg.cost_function.toDict()

    gdict, p_env = setup(env_kwargs=env_cfgs, cost_f_kwargs=cost_f_cfgs)

    p_env: SafePointEnv

    nodes = list(gdict.keys())

    inp_kwargs = []
    for itr in tqdm(range(len(nodes)), total=len(nodes), desc="prepare inputs"):
        inp_kwargs.append(
            {
                "root": nodes[itr],
                "g_dict": gdict,
                "cost_map": p_env._cost_map,
                "cost_limit": cfg.env.cost_limit,
                "output_dir": mp_out_dir,
            }
        )

    if args.use_mp:
        pool = mp.Pool(processes=num_procs)
        results = pool.map(mp_runner, inp_kwargs)
    else:
        print("[INFO] running in single-process mode")
        pbar_single_process = tqdm(total=len(inp_kwargs), disable=not args.pbar)
        for idx_arg in range(len(inp_kwargs)):
            run_CBFS(**inp_kwargs[idx_arg])
            pbar_single_process.update()
        pbar_single_process.close()
