"""
Start from one grid point, construct a tree via breadth-first search expansion, stop expansion when cost-limit
is violated. Construct reference problems for specified ranges of cumulative costs
NOTE: The deviation from the reference distance depends on the maneuver flexibility
(e.g., whether there exists some bottleneck passages, etc.)
Only operate on the discretized maze/graph level, no actual env stepping
"""

import time
import pickle
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from termcolor import cprint
from typing import Optional, List

from pud.algos.data_struct import init_embedded_dict


def CBFS(g: dict, root: tuple, cost_limit: float, cost_map: np.ndarray):
    """
    g: dict of dicts, exported from networkx graph
    root: node of a tuple (i,j)

    #MIT 16.413 Lecture Note Simple Search
    #1. Initialize Q with partial path (S) as only entry; set Visited = ( );
    #2. If Q is empty, fail. Else, pick some partial path N from Q;
    #3. // If head(N) = G, return N; (goal reached!)
    #4. Else
        #a) Remove N from Q;
        #b) Find all children of head(N) (its neighbors in g) not in Visited
        #and create a one-step extension of N to each child;
        #c) Add to Q all the extended paths;
        #d) Add children of head(N) to Visited;
        #e) Go to step 2
    """
    # Setup
    visited = []  # Visited nodes, not partial paths
    partial_paths = []  # Partially expanded paths
    partial_path_costs = []  # Cost of partial paths

    explored_paths = []
    explored_path_costs = []

    root_cost = cost_map[root[0], root[1]]
    if root_cost > cost_limit:
        return partial_paths, partial_path_costs

    visited.append(root)
    partial_paths.append([root])
    partial_path_costs.append(root_cost)

    while len(partial_paths) > 0:
        subpath_n = partial_paths.pop(-1)  # First in first out
        subpath_n_cost = partial_path_costs.pop(-1)
        explored_paths.append(subpath_n)
        explored_path_costs.append(subpath_n_cost)
        subpath_n_head = subpath_n[-1]
        for n_ch in g[subpath_n_head]:
            if n_ch in visited:
                continue
            # Add one-step extension
            visited.append(n_ch)
            n_ch_cost = cost_map[n_ch[0], n_ch[1]]
            new_subpath_cost = subpath_n_cost + n_ch_cost
            if new_subpath_cost > cost_limit:
                continue
            new_subpath = subpath_n.copy()
            new_subpath.append(n_ch)
            partial_paths.insert(0, new_subpath)
            partial_path_costs.insert(0, new_subpath_cost)

    return explored_paths, explored_path_costs


def compile_all_pair_constrained_shortest_trajs(
    gd: dict,
    cost_limit: float,
    cost_map: np.ndarray,
    output_dir: Path,
):
    """
    Store the raw/unnormalized indices of the maze
    cost_limit: speed up the plan generation as it does not have to explore the whole space
    evaluate edge cost

    gd: dict of dicts
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    for root in tqdm(gd):
        out, out_cost = CBFS(gd, root, cost_limit=cost_limit, cost_map=cost_map)
        output_file = output_dir.joinpath(
            "root={}_cost_limit={}.pkl".format(root, cost_limit)
        )
        save_data = {
            "trajs": out,  # list
            "costs": out_cost,  # list
        }
        f = open(output_file, "wb")
        pickle.dump(save_data, f)
        f.close()


def analyze_precompiled_cost_and_lengths(savedir):
    savedir_path = Path(savedir)
    list_fs = list(savedir_path.iterdir())

    # Not needed unless the range of costs need to be redefined
    set_costs = set()
    set_lens = set()
    for _, f in tqdm(
        enumerate(list_fs), total=len(list_fs), desc="calc unique length and costs"
    ):
        tmp_data = None
        with open(f, "rb") as f:
            tmp_data = pickle.load(f)
        trajs = tmp_data["trajs"]
        costs = tmp_data["costs"]
        for i in range(len(trajs)):
            traj_i = trajs[i]
            if len(traj_i) > 1:
                cost_i = costs[i]
                set_costs.add(cost_i)
                len_i = len(traj_i)
                set_lens.add(len_i)
    return set_costs, set_lens


def validate_test_args(
    policies: dict, cost_ranges: List[float], dist_ranges: List[float], terminate=False
):
    """make sure no empty set for the ranges of target dists and costs"""
    trajs = policies["trajs"]
    for d in dist_ranges:
        for c in cost_ranges:
            if len(trajs[d][c]) == 0:
                cprint("invalid targets: dist={}, c={}".format(d, c))
                if terminate:
                    assert False


def catalog_precompiled_paths(savedir, output_path: str):
    """
    Load all prebuilt policies for balanced sampling
    query policies based on traj distance and cost

    savedir: the directory that contains a set of precompiled grid trajectories
    output_path: the FULL target path to store the output catalog file for speedy fetch

    Build a catalog of path

    parent file: root=(0,0) ...
    each parent file contains two lists: trajs, and costs
        indexing should be 0, ..., len(trajs)-1
    collect the unique the cost set
    indexing the parent file in the dir

    each entry = parent file index + index within the parent file
    """
    savedir_path: Path = Path(savedir)
    list_fs = list(savedir_path.iterdir())

    cost_collections = set()
    len_collections = set()

    all_trajs = {}  # Based on costs, path lengths and then file index
    for i_f, f in tqdm(enumerate(list_fs), total=len(list_fs), desc="indexing trajs"):
        tmp_data = None
        with open(f, "rb") as f:
            tmp_data = pickle.load(f)
        trajs = tmp_data["trajs"]
        costs = tmp_data["costs"]
        for i in range(len(trajs)):
            traj_i = trajs[i]
            if len(traj_i) > 1:
                cost_i = costs[i]
                len_i = len(traj_i)

                init_embedded_dict(
                    all_trajs,
                    embeds=[
                        (len_i, dict),
                        (cost_i, dict),
                        (i_f, list),
                    ],
                )
                # Append according to cost, length, and file index
                all_trajs[len_i][cost_i][i_f].append(i)

                cost_collections.add(cost_i)
                len_collections.add(len_i)

    # This pool is huge, save as layered inds
    with open(output_path, "wb") as f:
        data_catalog = {
            "files": list_fs,
            "trajs": all_trajs,
            "parent_dir": savedir.as_posix(),
            "notes": """path_length->costs->file_index->traj_index, file_index starts from 0 based on files""",
            "cost_set": cost_collections,
            "len_set": len_collections,
        }
        pickle.dump(data_catalog, f)

    t0 = time.time()
    with open(output_path, "rb") as f:
        pickle.load(f)
    print("[INFO] loading time of sample policy catalog: {}".format(time.time() - t0))

    return data_catalog


def sample_precompiled_grid_policies(
    policies: dict,
    min_cost: float,
    max_cost: float,
    min_len: float,
    max_len: float,
    ps_costs: Optional[List[float]] = None,
):
    """
    Load sample files on-demand
    """
    # scratch for balanced sampling
    trajs = policies["trajs"]
    files = policies["files"]

    bounded_lens = [x for x in list(trajs.keys()) if (x >= min_len and x <= max_len)]
    if len(bounded_lens) == 0:
        cprint("[ERROR]: length range is empty", "red")
        return
    sample_len = np.random.choice(bounded_lens)

    bounded_costs = [
        x for x in list(trajs[sample_len].keys()) if (x >= min_cost and x <= max_cost)
    ]
    if len(bounded_costs) == 0:
        cprint("[ERROR]: cost range is empty", "red")
        return
    sample_cost = np.random.choice(bounded_costs)

    sample_file_ind = np.random.choice(list(trajs[sample_len][sample_cost].keys()))
    sample_file_path = files[sample_file_ind]
    sample_traj_idx = np.random.choice(trajs[sample_len][sample_cost][sample_file_ind])

    traj_f_data = None
    with open(sample_file_path, "rb") as f:
        traj_f_data = pickle.load(f)

    sample_traj = traj_f_data["trajs"][sample_traj_idx]
    sample_traj_cost = traj_f_data["costs"][sample_traj_idx]
    return sample_traj, sample_traj_cost
