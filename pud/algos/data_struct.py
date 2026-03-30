"""
Helper functions for data structures
"""

import torch
import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Union


def dict_expand(D: dict, keys: list):
    """
    Query multi-level keys from a dict
    keys = ["a", "b"] is the same as
    D["a"]["b"]
    """
    d = D

    if len(keys) == 0:
        return d

    for i, k in enumerate(keys):
        assert k in d, "{} of keys:{} not in dict".format(k, keys)
        if i == len(keys) - 1:
            return d[k]

        d = d[k]


def gather_log(eval_stats: dict, names_n_keys: Dict[str, list]):
    """
    eval_stats has the form of eval_stats[order_id][rest of keys]
    names_n_keys offers the list of keys to read data from eval_stats[id], and
        defines a convenient name, e.g.,
        "name", ["init_info","prediction"]
    """
    logs = {}
    for n in names_n_keys.keys():
        logs[n] = []

    for id in eval_stats.keys():
        for n in names_n_keys.keys():
            logs[n].append(dict_expand(D=eval_stats[id], keys=names_n_keys[n]))
    return logs


def inp_to_torch_device(
    inp: Union[
        NDArray,
        torch.FloatTensor,
        Dict[str, NDArray],
        Dict[str, torch.Tensor],
        Dict[str, torch.FloatTensor],
    ],
    device: torch.device,
):
    """convert dict inps to torch, skip other fields"""
    if isinstance(inp, dict):
        for key in inp:
            if isinstance(inp[key], np.ndarray):
                inp[key] = torch.from_numpy(inp[key]).to(device)  # type: ignore
            elif isinstance(inp[key], torch.Tensor):
                inp[key] = inp[key].to(device)  # type: ignore
        return inp

    if isinstance(inp, np.ndarray):
        inp = torch.from_numpy(inp).to(device)  # type: ignore
    elif isinstance(inp, torch.Tensor):
        inp = inp.to(device)  # type: ignore
    else:
        raise Exception("data type mismatch")
    return inp


def inp_to_numpy(
    inp: Union[torch.Tensor, Dict[str, torch.Tensor]],
):
    """convert dict inps to torch, skip other fields"""
    if isinstance(inp, dict):
        for key in inp:
            if isinstance(inp[key], torch.Tensor):
                inp[key] = inp[key].detach().cpu().numpy()  # type: ignore
        return inp

    if isinstance(inp, torch.Tensor):
        inp = inp.detach().cpu().numpy()  # type: ignore
    else:
        raise Exception("data type mismatch")
    return inp


def init_embedded_dict(D: dict, embeds: List[tuple] = []):
    """
    In-place init of embedded dict
    The init function should be either a list or dict
    embeds = [(key, init_function), ...]

    Example:
    DD = {}
    init_embedded_dict(DD, embeds=[(1, dict), (2, dict), (3, list)])
    init_embedded_dict(DD, embeds=[(1, dict), (2, dict), (3, list)])
    """
    tmp_D = D
    for next_key, init_f in embeds:
        if not (next_key in tmp_D):
            tmp_D[next_key] = init_f()

        assert isinstance(tmp_D[next_key], init_f)
        tmp_D = tmp_D[next_key]
    return


# Group values into groups/partitions
def find_group_ind(val: float, divs: List[float]):
    # Find the group index for the smallest val
    assert len(divs) >= 2, "length of divs needs >= 2"

    group_ind = -1
    for ig in range(0, len(divs) - 1):
        div_start = divs[ig]
        div_end = divs[ig + 1]
        if val >= div_start and val < div_end:
            group_ind = ig
            break
    if group_ind >= 0:
        return group_ind
    return


def arg_group_vals(vals: List[float], divs: List[float]):
    divs = np.sort(divs).tolist()
    assert len(divs) >= 2, "Length of divs needs >= 2"
    inds = np.argsort(vals)
    assert vals[inds[-1]] < divs[-1], "Values out of division upper bound"
    assert vals[inds[0]] >= divs[0], "Values out of division lower bound"

    groups = {}
    for i in range(0, len(divs) - 1):
        groups[i] = {
            "inds": [],
            "vals": [],
            "start": divs[i],
            "end": divs[i + 1],
        }

    # Assign groups and group indices
    group_ind = find_group_ind(val=vals[inds[0]], divs=divs)
    assert group_ind is not None
    div_end = divs[group_ind + 1]
    ind_start = 0
    for i in range(len(inds)):
        i_sort = inds[i]
        if vals[i_sort] >= div_end:
            g_inds = list(range(ind_start, i))
            g_vals = [vals[inds[j]] for j in g_inds]
            groups[group_ind]["inds"].extend(g_inds)
            groups[group_ind]["vals"].extend(g_vals)
            # Setup for next div
            ind_start = i
            group_ind = find_group_ind(val=vals[i_sort], divs=divs)
            assert group_ind is not None
            div_end = divs[group_ind + 1]

    g_inds = list(range(ind_start, i + 1))
    g_vals = [vals[inds[j]] for j in g_inds]
    groups[group_ind]["inds"].extend(g_inds)
    groups[group_ind]["vals"].extend(g_vals)
    return groups


def arg_topk(A: np.ndarray, topK: int):
    """
    Return the inds and vals of top K values
    A: multi-dim array of size (M,N,...), s.t. A[m,n,...] = val
    return:
        inds along each dim of A, e.g., (i0,j0, ...), (i1,j1,...)

    reference: https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
    """
    # Flatten the array
    A_f = A.flatten()
    f_inds = np.argpartition(A_f, kth=-topK, axis=None)
    top_k_inds = f_inds[-topK:]
    top_k_nd_inds = np.unravel_index(top_k_inds, shape=A.shape)
    return top_k_nd_inds


def get_nd_inds_set(inds: tuple[np.ndarray, ...]):
    """
    Inds are output of arg_topk, a tuple of indices along each dim
    return:
    [(ids of all dim of 1st point), (ids of all dim of 2nd point), ...]
    """
    set_inds = []
    for i in range(len(inds[0])):
        k_ind = [None] * len(inds)
        for n in range(len(inds)):
            k_ind[n] = inds[n][i]
        set_inds[i] = tuple(k_ind)
    return set_inds


def test_topk(A, topK):
    inds = arg_topk(-A, topK=topK)  # Find K minimum entries
    inds_set = get_nd_inds_set(inds)
    s0, s1 = A.shape
    for i in range(s0):
        for j in range(s1):
            if (i, j) not in inds_set:
                for ks in inds_set:
                    ki, kj = ks
                    # Check A[ki, kj] is smaller than other entries not included in top K set
                    assert A[ki, kj] <= A[i, j]
