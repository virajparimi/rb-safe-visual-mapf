from copy import deepcopy
from pathlib import Path

import numpy as np


def load_precomputed_risk_context(
    problem_set_file: str,
    num_agents: int,
    risk_bound_percent: float,
    problem_index: int,
):
    if not 0.0 <= risk_bound_percent <= 1.0:
        raise ValueError(
            f"risk_bound_percent must be between 0.0 and 1.0, got {risk_bound_percent}."
        )
    if problem_index < 0:
        raise ValueError(
            f"problem_index must be non-negative, got {problem_index}."
        )

    problem_set_path = Path(problem_set_file).expanduser()
    if problem_set_path.suffix.lower() != ".npz":
        raise FileNotFoundError(
            "risk_bound_percent requires a .npz problem set with precomputed "
            "ICAPS artifacts. Use an *_icaps problem_set_file."
        )

    dataset_dir = problem_set_path.with_suffix("")
    processed_problems_path = dataset_dir / "problems" / f"pbs_{num_agents}.npy"
    lb_path = dataset_dir / "cbs_risk_bounds" / f"lb_{num_agents}.npy"
    ub_path = dataset_dir / "cbs_risk_bounds" / f"ub_{num_agents}.npy"

    missing_paths = [
        str(path)
        for path in (processed_problems_path, lb_path, ub_path)
        if not path.is_file()
    ]
    if missing_paths:
        available_counts = sorted(
            {
                path.stem.split("_")[-1]
                for path in (dataset_dir / "problems").glob("pbs_*.npy")
            }
        )
        available_msg = (
            f" Available precomputed agent counts: {', '.join(available_counts)}."
            if available_counts
            else ""
        )
        raise FileNotFoundError(
            "Missing precomputed ICAPS artifacts for the selected problem set and "
            f"agent count. Expected: {', '.join(missing_paths)}.{available_msg}"
        )

    processed_problems = np.load(processed_problems_path, allow_pickle=True).tolist()
    cbs_lb_data = np.load(lb_path, allow_pickle=True).tolist()
    cbs_ub_data = np.load(ub_path, allow_pickle=True).tolist()

    if not processed_problems:
        raise ValueError(f"No grouped problems found in {processed_problems_path}.")
    if len(processed_problems) != len(cbs_lb_data) or len(processed_problems) != len(cbs_ub_data):
        raise ValueError(
            "Precomputed grouped problems and CBS risk bounds have inconsistent lengths."
        )

    if problem_index >= len(processed_problems):
        raise IndexError(
            "problem_index is out of range for the selected precomputed problem set. "
            f"Got {problem_index}, but only indices 0 to {len(processed_problems) - 1} "
            "are available."
        )

    selected_problem = processed_problems[problem_index]
    for key in ("pbs", "starts", "goals", "graph"):
        if key not in selected_problem:
            raise KeyError(
                f"Precomputed grouped problem is missing required key '{key}'."
            )

    if len(selected_problem["pbs"]) != num_agents:
        raise ValueError(
            "Selected grouped problem does not match the requested num_agents."
        )

    lb = float(cbs_lb_data[problem_index])
    ub = float(cbs_ub_data[problem_index])
    if lb == -1.0 or ub == -1.0:
        raise ValueError(
            "The selected precomputed grouped problem does not have a valid CBS risk bound."
        )

    risk_bound = lb if lb == ub else lb + risk_bound_percent * (ub - lb)

    return {
        "problem_index": problem_index,
        "dataset_dir": dataset_dir,
        "risk_bound_percent": risk_bound_percent,
        "risk_bound": float(risk_bound),
        "problems": deepcopy(selected_problem["pbs"]),
        "starts": deepcopy(selected_problem["starts"]),
        "goals": deepcopy(selected_problem["goals"]),
        "planning_graph": selected_problem["graph"],
    }
