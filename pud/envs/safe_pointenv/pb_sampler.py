import numpy as np
from typing import List, Union, Optional

from pud.algos.data_struct import arg_topk
from pud.algos.lagrange.drl_ddpg_lag import DRLDDPGLag
from pud.algos.vision.vision_agent import LagVisionUVFDDPG
from pud.envs.safe_pointenv.safe_wrappers import SafeGoalConditionedPointWrapper


def calc_pairwise_cost(agent: DRLDDPGLag, rb_vec: np.ndarray, ensemble_agg="max"):
    pcost = agent.get_pairwise_cost(rb_vec, aggregate=None)
    pcost_agg = None
    if ensemble_agg == "max":
        pcost_agg = np.max(pcost, axis=0)
    elif ensemble_agg == "mean":
        pcost_agg = np.mean(pcost, axis=0)
    return pcost_agg


def calc_pairwise_dist(agent: DRLDDPGLag, rb_vec: np.ndarray, ensemble_agg="max"):
    pdist = agent.get_pairwise_dist(rb_vec, aggregate=None)
    pdist_agg = None
    if ensemble_agg == "max":
        pdist_agg = np.max(pdist, axis=0)
    elif ensemble_agg == "mean":
        pdist_agg = np.mean(pdist, axis=0)
    return pdist_agg


def sample_pbs_by_agent(
    env: Union[SafeGoalConditionedPointWrapper],
    agent: DRLDDPGLag,
    num_states: int = 100,
    min_dist: float = 0,
    max_dist: float = 10,
    target_val: Union[float, None] = None,
    ensemble_agg: str = "max",
    use_uncertainty: bool = True,  # boost samples of high uncertainty
    uncertainty_lb: float = 0.0,
    uncertainty_ub: float = 1.0,
    K: int = 5,  # num of samples nearest to the target metric
) -> List[dict]:
    """Sample problems with target metrics according to the predictions of the agent

    Args:
        env (SafeGoalConditionedPointWrapper): env that contains reset_orig, which returns normalized start-goal pairs
        agent (DRLDDPGLag):
        pval_f: (agent, rb_vec) -> pairwise values
        num_states (int, optional): _description_. Defaults to 100.
    """
    # Online generate start and goal states nearest to target cumulative costs
    rb_vec = np.array([env.sample_empty_state() for _ in range(num_states)])
    rb_vec_goal = np.array([env.sample_empty_state() for _ in range(num_states)])

    # Predict the pairwise costs
    pdist = agent.get_pairwise_dist(
        obs_vec=np.stack([env.normalize_obs(s) for s in rb_vec]),
        goal_vec=np.stack([env.normalize_obs(s) for s in rb_vec_goal]),
        aggregate=None,
    )
    pdist_agg = None
    if ensemble_agg == "max":
        pdist_agg = np.max(pdist, axis=0)
    elif ensemble_agg == "mean":
        pdist_agg = np.mean(pdist, axis=0)
    assert pdist_agg is not None

    pdist_std, pdist_std_mean = np.zeros_like(pdist_agg), np.zeros_like(pdist_agg)
    if use_uncertainty:
        pdist_std = np.std(pdist, axis=0)
        pdist_std_mean = np.mean(pdist_std)

    lb_mask = pdist_agg + pdist_std >= min_dist
    ub_mask = pdist_agg - pdist_std <= max_dist
    prod_mask = lb_mask * ub_mask

    gInds = np.where(prod_mask)
    if len(gInds[0]) == 0:
        return []
    else:
        pdist_gInds = pdist_agg[gInds]
        scoring = np.zeros_like(pdist_gInds)  # Smaller is better
        if target_val is not None:
            scoring = scoring + np.abs(pdist_gInds - target_val)
        # Encourage diverse samples
        if use_uncertainty:
            pdist_stds_gInds = pdist_std[gInds]
            scoring = scoring - np.clip(
                pdist_stds_gInds, a_min=uncertainty_lb, a_max=uncertainty_ub
            )

        K = min(K, len(scoring))
        mInds = arg_topk(-scoring, topK=K)  # find K minimum entries
        gmInds = (gInds[0][mInds], gInds[1][mInds])

        nearest_pbs = [{}] * K
        for n in range(K):
            i, j = gmInds[0][n], gmInds[1][n]
            nearest_pbs[n] = {
                "start": rb_vec[i],
                "goal": rb_vec_goal[j],
                "info": {
                    "prediction": pdist_agg[i, j],
                    "proj_dist": pdist_agg[i, j],
                    "ensemble_std_mean": pdist_std_mean,
                },
            }
    return nearest_pbs


def sample_pbs_by_agent_deprecated(
    env: SafeGoalConditionedPointWrapper,
    agent: DRLDDPGLag,
    num_states: int = 100,
    min_dist: float = 0,
    max_dist: float = 10,
    target_val: Union[float, None] = None,
    ensemble_agg: str = "max",
    use_uncertainty: bool = True,  # Boost samples of high uncertainty
    uncertainty_lb: float = 0.0,
    uncertainty_ub: float = 1.0,
    K: int = 5,  # Number of samples nearest to the target metric
) -> List[dict]:
    """Sample problems with target metrics according to the predictions of the agent

    Args:
        env (SafeGoalConditionedPointWrapper): env that contains reset_orig, which returns normalized start-goal pairs
        agent (DRLDDPGLag):
        pval_f: (agent, rb_vec) -> pairwise values
        num_states (int, optional): _description_. Defaults to 100.
    """
    # Online generate start and goal states nearest to target cumulative costs
    rb_vec = []
    for i in range(num_states):
        s0, _ = env.reset_orig()
        rb_vec.append(s0)
    rb_vec = np.array([x["observation"] for x in rb_vec])

    rb_vec_goal = []
    for i in range(num_states):
        s0, info = env.reset_orig()
        rb_vec_goal[i] = s0
    rb_vec_goal = np.array([x["observation"] for x in rb_vec_goal])

    # Predict the pairwise costs
    pdist = agent.get_pairwise_dist(
        obs_vec=rb_vec, goal_vec=rb_vec_goal, aggregate=None
    )
    pdist_agg = None
    if ensemble_agg == "max":
        pdist_agg = np.max(pdist, axis=0)
    elif ensemble_agg == "mean":
        pdist_agg = np.mean(pdist, axis=0)
    assert pdist_agg is not None

    pdist_std, pdist_std_mean = np.zeros_like(pdist_agg), np.zeros_like(pdist_agg)
    if use_uncertainty:
        pdist_std = np.std(pdist, axis=0)
        pdist_std_mean = np.mean(pdist_std)

    lb_mask = pdist_agg + pdist_std >= min_dist
    ub_mask = pdist_agg - pdist_std <= max_dist
    prod_mask = lb_mask * ub_mask

    gInds = np.where(prod_mask)
    if len(gInds[0]) == 0:
        return []
    else:
        pdist_gInds = pdist_agg[gInds]
        scoring = np.zeros_like(pdist_gInds)  # Smaller is better
        if target_val is not None:
            scoring = scoring + np.abs(pdist_gInds - target_val)
        # Encourage diverse samples
        if use_uncertainty:
            pdist_stds_gInds = pdist_std[gInds]
            scoring = scoring - np.clip(
                pdist_stds_gInds, a_min=uncertainty_lb, a_max=uncertainty_ub
            )

        K = min(K, len(scoring))
        mInds = arg_topk(-scoring, topK=K)  # find K minimum entries
        gmInds = (gInds[0][mInds], gInds[1][mInds])

        nearest_pbs = [{}] * K
        for n in range(K):
            i, j = gmInds[0][n], gmInds[1][n]
            nearest_pbs[n] = {
                "start": env.de_normalize_obs(rb_vec[i]),
                "goal": env.de_normalize_obs(rb_vec_goal[j]),
                "info": {
                    "prediction": pdist_agg[i, j],
                    "proj_dist": pdist_agg[i, j],
                    "ensemble_std_mean": pdist_std_mean,
                },
            }
    return nearest_pbs


def sample_cost_pbs_by_agent(
    env: Union[SafeGoalConditionedPointWrapper],
    agent: Union[DRLDDPGLag, LagVisionUVFDDPG],
    num_states: int = 100,
    target_val: Union[float, None] = None,
    min_dist: Optional[float] = None,
    max_dist: Optional[float] = None,
    ensemble_agg: str = "mean",
    K: int = 5,  # Number of samples nearest to the target metric
    use_uncertainty: bool = True,  # Boost samples of high uncertainty
    uncertainty_lb: float = 0.0,
    uncertainty_ub: float = 1.0,
):
    """
    Filter based on distance constraints
    Problems whose start and goals are seperated too far away is meaningless as they are handled by the HRL
    If failed, return an empty list, because the test results would not be informative
    """

    rb_vec = np.array([env.sample_safe_empty_state() for _ in range(num_states)])
    rb_vec_goal = np.array([env.sample_safe_empty_state() for _ in range(num_states)])

    # Ensure that the states are not repeated
    rb_vec += np.random.uniform(size=rb_vec.shape)
    rb_vec_goal += np.random.uniform(size=rb_vec_goal.shape)

    prod_mask = np.ones([num_states, num_states], dtype=bool)
    if min_dist and max_dist:
        # Predict the pairwise costs
        pdist = agent.get_pairwise_dist(
            obs_vec=np.stack([env.normalize_obs(s) for s in rb_vec]),
            goal_vec=np.stack([env.normalize_obs(s) for s in rb_vec_goal]),
            aggregate=None,
        )

        pdist_agg = None
        if ensemble_agg == "max":
            pdist_agg = np.max(pdist, axis=0)
        elif ensemble_agg == "mean":
            pdist_agg = np.mean(pdist, axis=0)
        assert pdist_agg is not None

        pdist_std = 0.0
        if use_uncertainty:
            pdist_std = np.std(pdist, axis=0)

        lb_mask = pdist_agg + pdist_std >= min_dist
        ub_mask = pdist_agg - pdist_std <= max_dist
        prod_mask = lb_mask * ub_mask

    gInds = np.where(prod_mask)
    if len(gInds[0]) == 0:
        return []
    else:
        pcosts = agent.get_pairwise_cost(
            obs_vec=np.stack([env.normalize_obs(s) for s in rb_vec]),
            goal_vec=np.stack([env.normalize_obs(s) for s in rb_vec_goal]),
            aggregate=None,
        )  # num_ens x num_states x num_states
        pcosts_agg = np.mean(pcosts, axis=0)
        pcosts_std = np.std(pcosts, axis=0)
        pcosts_std_mean = np.mean(pcosts_std)
        pcosts_gInds = pcosts_agg[gInds]
        scoring = np.zeros_like(pcosts_gInds)
        if target_val is not None:
            scoring = scoring + np.abs(pcosts_gInds - target_val)
        # Encourage diverse samples
        if use_uncertainty:
            pcosts_std_gInds = pcosts_std[gInds]
            scoring = scoring - np.clip(
                pcosts_std_gInds, a_min=uncertainty_lb, a_max=uncertainty_ub
            )
        K = min(K, len(scoring))
        mInds = arg_topk(-scoring, topK=K)  # Find K minimum entries
        gmInds = (gInds[0][mInds], gInds[1][mInds])

        nearest_pbs = [{}] * K
        for n in range(K):
            i, j = gmInds[0][n], gmInds[1][n]
            nearest_pbs[n] = {
                "start": rb_vec[i],
                "goal": rb_vec_goal[j],
                "info": {
                    "prediction": pcosts_agg[i, j],
                    "proj_dist": None,
                    "ensemble_std_mean": pcosts_std_mean,
                },
            }
        return nearest_pbs


def sample_cost_pbs_by_agent_deprecated(
    env: SafeGoalConditionedPointWrapper,
    agent: DRLDDPGLag,
    num_states: int = 100,
    target_val: Union[float, None] = None,
    min_dist: Optional[float] = 0,
    max_dist: float = 10,
    ensemble_agg: str = "mean",
    K: int = 5,  # Number of samples nearest to the target metric
    use_uncertainty: bool = True,  # Boost samples of high uncertainty
    uncertainty_lb: float = 0.0,
    uncertainty_ub: float = 1.0,
    non_grid: bool = False,
):
    """
    filter based on distance constraints
    problems whose start and goals are seperated too far away is
    meaningless as they are handled by the HRL
    if failed, return an empty list, because the test results would not be informative
    """
    rb_vec = []
    for i in range(num_states):
        if non_grid:
            s0 = env.sample_safe_empty_state(cost_limit=agent.lagrange.cost_limit)
            s0 += np.random.uniform(size=2)
            s0 = {"observation": env.normalize_obs(s0)}
        else:
            s0, info = env.reset_orig()
        rb_vec[i] = s0
    rb_vec = np.array([x["observation"] for x in rb_vec])

    rb_vec_goal = []
    for i in range(num_states):
        if non_grid:
            s0 = env.sample_safe_empty_state(cost_limit=agent.lagrange.cost_limit)
            s0 += np.random.uniform(size=2)
            s0 = {"observation": env.normalize_obs(s0)}
        else:
            s0, info = env.reset_orig()
        rb_vec_goal[i] = s0
    rb_vec_goal = np.array([x["observation"] for x in rb_vec_goal])

    # Predict the pairwise costs
    pdist = agent.get_pairwise_dist(
        obs_vec=rb_vec, goal_vec=rb_vec_goal, aggregate=None
    )
    pdist_agg = None
    if ensemble_agg == "max":
        pdist_agg = np.max(pdist, axis=0)
    elif ensemble_agg == "mean":
        pdist_agg = np.mean(pdist, axis=0)
    assert pdist_agg is not None

    pdist_std = 0.0
    if use_uncertainty:
        pdist_std = np.std(pdist, axis=0)

    lb_mask = pdist_agg + pdist_std >= min_dist
    ub_mask = pdist_agg - pdist_std <= max_dist
    prod_mask = lb_mask * ub_mask

    gInds = np.where(prod_mask)
    if len(gInds[0]) == 0:
        return []
    else:
        pcosts = agent.get_pairwise_cost(
            obs_vec=rb_vec, goal_vec=rb_vec_goal, aggregate=None
        )  # num_ens x num_states x num_states
        pcosts_agg = np.mean(pcosts, axis=0)
        pcosts_std = np.std(pcosts, axis=0)
        pcosts_std_mean = np.mean(pcosts_std)
        pcosts_gInds = pcosts_agg[gInds]
        scoring = np.zeros_like(pcosts_gInds)
        if target_val is not None:
            scoring = scoring + np.abs(pcosts_gInds - target_val)
        # Encourage diverse samples
        if use_uncertainty:
            pcosts_std_gInds = pcosts_std[gInds]
            scoring = scoring - np.clip(
                pcosts_std_gInds, a_min=uncertainty_lb, a_max=uncertainty_ub
            )
        K = min(K, len(scoring))
        mInds = arg_topk(-scoring, topK=K)  # Find K minimum entries
        gmInds = (gInds[0][mInds], gInds[1][mInds])

        nearest_pbs = [{}] * K
        for n in range(K):
            i, j = gmInds[0][n], gmInds[1][n]
            nearest_pbs[n] = {
                "start": env.de_normalize_obs(rb_vec[i]),
                "goal": env.de_normalize_obs(rb_vec_goal[j]),
                "info": {
                    "prediction": pcosts_agg[i, j],
                    "proj_dist": pdist_agg[i, j],
                    "ensemble_std_mean": pcosts_std_mean,
                },
            }
        return nearest_pbs


def load_pb_set(
    file_path: str,
    env: Union[SafeGoalConditionedPointWrapper],
    agent: DRLDDPGLag,
):
    pnts = np.loadtxt(fname=file_path, dtype=float, delimiter=",").astype(
        env.action_space.dtype
    )
    assert len(pnts) % 2 == 0, "the number of points need to be even"
    start_list = []
    goal_list = []

    # The illustration files are assumed to be normalized
    denorm_factor = np.array(
        [env.get_map_height(), env.get_map_width()], dtype=np.float32
    )

    for i in range(len(pnts)):
        if i % 2 == 0:
            start_list.append(pnts[i] * denorm_factor)
        else:
            goal_list.append(pnts[i] * denorm_factor)

    gc_states = {
        "observation": np.stack([env.normalize_obs(x) for x in start_list]),
        "goal": np.stack([env.normalize_obs(x) for x in goal_list]),
    }

    pcosts = agent.get_cost_to_goal(state=gc_states, aggregate=None)
    pcosts_agg = np.mean(pcosts, axis=0)
    pdists = agent.get_dist_to_goal(state=gc_states, aggregate=None)
    pdists_agg = np.mean(pdists, axis=0)
    pcosts_std_mean = np.mean(np.std(pcosts, axis=1))

    pbs = [{}] * len(start_list)
    for i in range(len(start_list)):
        pbs[i] = {
            "start": start_list[i],
            "goal": goal_list[i],
            "info": {
                "prediction": np.mean(pcosts_agg[i]),
                "proj_dist": pdists_agg[i],
                "ensemble_std_mean": pcosts_std_mean,
            },
        }
    return pbs
