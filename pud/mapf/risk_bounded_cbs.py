import time
import heapq
import logging
import numpy as np
from networkx import Graph
from numpy.typing import NDArray
from typing import Dict, List, Tuple, Union

from pud.mapf.cbs import CBSNode, CBSSolver
from pud.mapf.single_agent_planner import RiskBudgetedAStar
from pud.mapf.utils import detect_collisions, get_location, standard_split
from pud.mapf.mapf_exceptions import MAPFError, MAPFErrorCodes


class BudgetAllocater(object):

    def __init__(self, strategy: str, risk_bound: float):
        self.strategy = strategy
        self.risk_bound = risk_bound

    def allocate(self, paths: List[List[int]]):
        raise NotImplementedError


class UniformBudgetAllocater(BudgetAllocater):

    def __init__(self, risk_bound: float):
        super().__init__("uniform", risk_bound)

    def allocate(self, paths, utility=None):
        self.num_agents = len(paths)
        return [self.risk_bound / self.num_agents] * self.num_agents


# Use this one when the utility is based off of the risk of the path
class UtilityBudgetAllocater(BudgetAllocater):
    def __init__(self, risk_bound: float):
        super().__init__("utility", risk_bound)

    def allocate(self, paths, utility):
        risk_allocation = []
        self.num_agents = len(paths)
        utility_sum = sum(utility)
        for agent in range(self.num_agents):
            risk_allocation.append(self.risk_bound * utility[agent] / utility_sum)
        assert np.isclose(sum(risk_allocation), self.risk_bound, atol=1e-5)
        return risk_allocation


# Use this one when the utility is based off of the path length of each path
class InverseUtilityBudgetAllocater(BudgetAllocater):
    def __init__(self, risk_bound: float):
        super().__init__("inverse_utility", risk_bound)

    def allocate(self, paths, utility):
        risk_allocation = []
        self.num_agents = len(paths)
        inverse_utility = [1 / utility for utility in utility]
        inverse_sum = sum(inverse_utility)
        for agent in range(self.num_agents):
            risk_allocation.append(
                self.risk_bound * inverse_utility[agent] / inverse_sum
            )
        assert np.isclose(sum(risk_allocation), self.risk_bound, atol=1e-5)
        return risk_allocation


class RiskBoundedCBSNode(CBSNode):
    def __init__(
        self,
        id: int,
        cost: float,
        paths: List[List[int]],
        collisions: List[Dict],
        constraints: List[Dict],
        risk_allocation: List[float],
        paths_found: List[bool],
    ):
        super().__init__(id, cost, paths, collisions, constraints)
        self.risk_allocation = risk_allocation
        self.paths_found = paths_found

    def copy(self):
        return RiskBoundedCBSNode(
            id=self.id,
            cost=self.cost,
            paths=self.paths.copy(),
            collisions=self.collisions.copy(),
            constraints=self.constraints.copy(),
            risk_allocation=self.risk_allocation.copy(),
            paths_found=self.paths_found.copy(),
        )


class RiskBoundedCBSSolver(CBSSolver):
    def __init__(
        self,
        graph: Graph,
        goals: List[int],
        starts: List[int],
        pdist: NDArray,
        config: Dict,
    ):
        super().__init__(graph, goals, starts, pdist, config)
        self.risk_bound = config["risk_bound"]
        self.budget_allocator = config["budget_allocater"]

        # Variables to keep track of the search tree
        self.open_list = []

        self.risk_reallocation_strategy = config.get(
            "risk_reallocation_strategy", "surplus_deficit"
        )
        market_config = config.get("risk_market_config", {})
        positive_risk_bound = self.risk_bound > 0 and not np.isclose(
            self.risk_bound, 0.0, atol=1e-5
        )
        default_eta0 = (
            0.05 * (self.risk_bound / self.num_agents) if positive_risk_bound else 1.0
        )
        default_eta_min = (
            0.01 * (self.risk_bound / self.num_agents) if positive_risk_bound else 0.1
        )
        self.market_eta0 = max(market_config.get("eta0", default_eta0), 1e-6)
        self.market_eta_min = max(market_config.get("eta_min", default_eta_min), 1e-6)
        default_eps_r = self.market_eta_min * self.num_agents
        self.market_eps_r = market_config.get("eps_R", default_eps_r)
        self.market_price_tol = market_config.get("price_tol", 1e-3)
        self.market_tie_tol = market_config.get("tie_tol", 1e-10)
        self.market_max_sweeps = market_config.get("max_sweeps", 6)
        self.market_max_price_iters = market_config.get("max_price_iters", 20)
        self.market_cache_precision = int(market_config.get("cache_precision", 6))
        self.price_cache: Dict[
            Tuple,
            Dict[Tuple[int, float], Union[None, Dict[str, Union[float, List[int]]]]],
        ] = {}
        self.market_last_meta: Dict = {}

        self.budget_allocator_cls = None
        if self.budget_allocator == "uniform":
            self.budget_allocator_cls = UniformBudgetAllocater
        elif self.budget_allocator == "utility":
            self.budget_allocator_cls = UtilityBudgetAllocater
        elif self.budget_allocator == "inverse_utility":
            self.budget_allocator_cls = InverseUtilityBudgetAllocater
        else:
            raise RuntimeError(
                MAPFError(MAPFErrorCodes.INVALID_BUDGET_ALLOCATER)["message"]
            )
        self.budget_allocator = self.budget_allocator_cls(self.risk_bound)

    def make_planners(self, config: Dict) -> None:
        self.single_agent_planners = {}
        for agent in range(self.num_agents):
            self.single_agent_planners[agent] = RiskBudgetedAStar(
                config=config,
                agent_id=agent,
                graph=self.graph,
                goal=self.goals[agent],
                start=self.starts[agent],
            )

    def min_feasible_cost(
        self,
        agent: int,
        constraints: List[Dict],
        experience: Union[List[int], None] = None,
    ) -> Union[float, MAPFErrorCodes]:
        agent_path = self.single_agent_planners[agent].find_path(
            constraints=constraints,
            experience=experience,
            edge_attribute="cost",
            max_time=self.max_time // self.num_agents,
        )
        if type(agent_path) is MAPFErrorCodes:
            error_code = agent_path
            return error_code
        else:
            risk_value = self.compute_cost(agent_path, risk=True)  # type: ignore
            return risk_value

    def reallocate_risk(
        self,
        failing_agents: List[int],
        cbs_node: RiskBoundedCBSNode,
    ) -> Union[Tuple[List[float], List], MAPFErrorCodes]:

        if self.risk_reallocation_strategy == "price_clearing":
            return self.price_clearing_reallocate(failing_agents, cbs_node)
        return self.surplus_deficit_reallocate(failing_agents, cbs_node)

    def surplus_deficit_reallocate(
        self,
        failing_agents: List[int],
        cbs_node: RiskBoundedCBSNode,
    ) -> Union[Tuple[List[float], List], MAPFErrorCodes]:

        passing_agents = [
            agent for agent in range(self.num_agents) if agent not in failing_agents
        ]

        required_budgets = {}
        for agent in failing_agents:
            logging.debug(
                "Computing minimum feasible cost for agent {} with constraints {}".format(
                    agent, cbs_node.constraints
                )
            )
            minimum_cost = self.min_feasible_cost(
                agent, cbs_node.constraints, cbs_node.paths[agent]
            )
            logging.debug("Minimum cost for agent {} is {}".format(agent, minimum_cost))

            # If the agent is not able to find a risk-based path within the constraints
            if type(minimum_cost) is MAPFErrorCodes:
                error_code = minimum_cost
                return error_code

            required_budgets[agent] = minimum_cost

        additional_budget = sum(required_budgets.values()) - sum(
            cbs_node.risk_allocation[agent] for agent in failing_agents
        )

        budget_available = 0
        passing_agent_minimum_feasible_budgets = {}
        for agent in passing_agents:
            passing_agent_minimum_feasible_budgets[agent] = self.min_feasible_cost(
                agent, cbs_node.constraints, cbs_node.paths[agent]
            )
            logging.debug(
                "Minimum cost for passing agent {} is {}".format(
                    agent, passing_agent_minimum_feasible_budgets[agent]
                )
            )

            # If the agent is not able to find a risk-based path within the constraints
            if type(passing_agent_minimum_feasible_budgets[agent]) is MAPFErrorCodes:
                error_code = passing_agent_minimum_feasible_budgets[agent]
                return error_code

            budget_available += (
                cbs_node.risk_allocation[agent]
                - passing_agent_minimum_feasible_budgets[agent]
            )

        # If the cumulative minimum feasible risk allocations of passing agents are less than the required budget
        # for the failing agents, then we no reallocation of the risk will make the failing agents pass without
        # violating the risk-budgeted paths of the passing agents
        if round(additional_budget, 10) > round(budget_available, 10):
            return MAPFErrorCodes.BUDGET_MISMATCH

        new_allocation = cbs_node.risk_allocation.copy()
        for agent in failing_agents:
            new_allocation[agent] = required_budgets[agent]

        for agent in passing_agents:
            max_reduction = (
                cbs_node.risk_allocation[agent]
                - passing_agent_minimum_feasible_budgets[agent]
            )
            reduction = min(max_reduction, additional_budget)
            new_allocation[agent] -= reduction
            additional_budget -= reduction
            if additional_budget <= 0:
                break

        diff = np.logical_not(np.isclose(cbs_node.risk_allocation, new_allocation))
        return new_allocation, np.where(diff)[0].tolist()

    def price_clearing_reallocate(
        self,
        failing_agents: List[int],
        cbs_node: RiskBoundedCBSNode,
    ) -> Union[Tuple[List[float], List], MAPFErrorCodes]:

        signature = self.constraint_signature(cbs_node.constraints)
        cache = self.price_cache.setdefault(signature, {})

        agents = list(range(self.num_agents))

        # Used to find the candidates around current allocation of an agent
        eta = max(self.market_eta0, self.market_eta_min)

        # Compute per-agent feasibility windows.

        # Minimum risk corresponds to optimizing for risk only
        risk_optimized_risk_map: Dict[int, float] = {}
        risk_optimized_length_map: Dict[int, float] = {}

        # Maximum risk corresponds to optimizing for length only
        length_optimized_risk_map: Dict[int, float] = {}
        length_optimized_length_map: Dict[int, float] = {}

        # price_stars: Dict[int, float] = {}
        for agent in agents:
            risk_optimized_result = self.risk_optimized_path(agent, cbs_node, cache)
            if type(risk_optimized_result) is MAPFErrorCodes:
                error_code = risk_optimized_result
                return error_code
            risk_optimized_risk, risk_optimized_length, _ = risk_optimized_result  # type: ignore
            risk_optimized_risk_map[agent] = max(0.0, risk_optimized_risk)
            risk_optimized_length_map[agent] = risk_optimized_length

            length_optimized_result = self.length_optimized_path(agent, cbs_node, cache)
            if type(length_optimized_result) is MAPFErrorCodes:
                error_code = length_optimized_result
                return error_code
            length_optimized_risk, length_optimized_length, _ = length_optimized_result  # type: ignore
            length_optimized_risk_map[agent] = max(
                risk_optimized_risk_map[agent], length_optimized_risk
            )
            length_optimized_length_map[agent] = length_optimized_length

            # if np.isclose(
            #     length_optimized_risk_map[agent],
            #     risk_optimized_risk_map[agent],
            #     atol=self.market_tie_tol,
            # ):
            #     price_stars[agent] = 0.0
            #     continue
            # denom = length_optimized_risk_map[agent] - risk_optimized_risk_map[agent]
            # numerator = (
            #     length_optimized_length_map[agent] - risk_optimized_length_map[agent]
            # )
            # if np.isclose(denom, 0.0, atol=1e-5):
            #     price_stars[agent] = 0.0
            # else:
            #     price_stars[agent] = numerator / denom

        minimum_risk = sum(risk_optimized_risk_map.values())
        if minimum_risk > self.risk_bound and not np.isclose(
            minimum_risk, self.risk_bound, atol=1e-5
        ):
            logging.debug(
                "Total minimum risk {} exceeds risk bound {}".format(
                    minimum_risk, self.risk_bound
                )
            )
            return MAPFErrorCodes.BUDGET_MISMATCH

        if np.isclose(minimum_risk, self.risk_bound, atol=self.market_eps_r):
            new_allocation = [risk_optimized_risk_map[a] for a in agents]
            self.update_paths_from_cache(
                agents, risk_optimized_risk_map, cbs_node, cache
            )
            changed = np.where(
                np.logical_not(np.isclose(cbs_node.risk_allocation, new_allocation))
            )[0].tolist()
            return new_allocation, changed

        maximum_risk = sum(length_optimized_risk_map.values())
        if maximum_risk < self.risk_bound or np.isclose(
            maximum_risk, self.risk_bound, atol=1e-5
        ):
            # Allocate maximum risk to all agents since it is feasible
            new_allocation = [length_optimized_risk_map[a] for a in agents]
            self.update_paths_from_cache(
                agents, length_optimized_risk_map, cbs_node, cache
            )
            changed = np.where(
                np.logical_not(np.isclose(cbs_node.risk_allocation, new_allocation))
            )[0].tolist()
            return new_allocation, changed

        # Estimate the risk allocation at the lowest price point (p = 0).
        delta_low = {
            agent: self.clamp_cap(
                cbs_node.risk_allocation[agent],
                risk_optimized_risk_map[agent],
                length_optimized_risk_map[agent],
            )
            for agent in agents
        }
        agent_order = self.agent_order(failing_agents, agents)
        delta_low = self.price_best_response(
            eta=eta,
            price=0.0,
            cache=cache,
            cbs_node=cbs_node,
            allocation=delta_low,
            agent_order=agent_order,
            delta_min=risk_optimized_risk_map,
            delta_max=length_optimized_risk_map,
        )
        sum_low = sum(delta_low.values())
        if sum_low < self.risk_bound or np.isclose(
            sum_low, self.risk_bound, atol=1e-5
        ):
            new_allocation = [delta_low[a] for a in agents]
            self.update_paths_from_cache(agents, delta_low, cbs_node, cache)
            changed = np.where(
                np.logical_not(np.isclose(cbs_node.risk_allocation, new_allocation))
            )[0].tolist()
            return new_allocation, changed

        price_low = 0.0
        # max_price_star = max(price_stars.values()) if price_stars else 0.0
        # price_high = max(self.market_price_tol, max_price_star + self.market_price_tol)
        price_high = 1e6

        best_feasible_alloc = None
        best_feasible_sum = -np.inf

        # Bisection on price with warm-started allocations.
        price_iters = 0
        delta_seed = risk_optimized_risk_map.copy()
        while (price_high - price_low) > self.market_price_tol and (
            price_iters < self.market_max_price_iters
        ):
            price_iters += 1
            price_mid = 0.5 * (price_low + price_high)
            delta_mid = self.price_best_response(
                eta=eta,
                cache=cache,
                price=price_mid,
                cbs_node=cbs_node,
                allocation=delta_seed,
                agent_order=agent_order,
                delta_min=risk_optimized_risk_map,
                delta_max=length_optimized_risk_map,
            )
            sum_mid = sum(delta_mid.values())

            # if abs(sum_mid - self.risk_bound) <= eps_r:
            if sum_mid < self.risk_bound or np.isclose(
                sum_mid, self.risk_bound, atol=1e-5
            ):
                if best_feasible_alloc is None or sum_mid > best_feasible_sum:
                    best_feasible_alloc = delta_mid.copy()
                    best_feasible_sum = sum_mid

            if sum_mid > self.risk_bound and not np.isclose(
                sum_mid, self.risk_bound, atol=1e-5
            ):
                price_low = price_mid
            else:
                price_high = price_mid

            delta_seed = delta_mid
            eta = max(eta / 2.0, self.market_eta_min)

        logging.debug("Price iterations {}".format(price_iters))

        if best_feasible_alloc is not None:
            delta_best = best_feasible_alloc
        else:
            logging.debug(
                "No feasible allocation found within price iterations.".format()
            )
            return MAPFErrorCodes.BUDGET_MISMATCH

        new_allocation = [delta_best[a] for a in agents]
        self.update_paths_from_cache(agents, delta_best, cbs_node, cache)
        diff = np.logical_not(np.isclose(cbs_node.risk_allocation, new_allocation))
        changed_agents = np.where(diff)[0].tolist()

        # has_failing_change = any(agent in failing_agents for agent in changed_agents)
        # logging.debug("Changed agents: {}".format(changed_agents))
        # if not has_failing_change:
        #     logging.debug("No failing agents changed their allocation.")
        #     return MAPFErrorCodes.BUDGET_MISMATCH

        return new_allocation, changed_agents

    def agent_order(self, failing_agents: List[int], agents: List[int]) -> List[int]:
        failing_set = set(failing_agents)
        order = failing_agents.copy()
        for agent in agents:
            if agent not in failing_set:
                order.append(agent)
        return order

    def price_best_response(
        self,
        eta: float,
        price: float,
        agent_order: List[int],
        delta_min: Dict[int, float],
        delta_max: Dict[int, float],
        allocation: Dict[int, float],
        cbs_node: RiskBoundedCBSNode,
        cache: Dict[Tuple[int, float], Union[None, Dict[str, Union[float, List[int]]]]],
    ) -> Dict[int, float]:
        sweeps = 0
        step = max(eta, self.market_eta_min)
        current = allocation.copy()

        while True:
            sweeps += 1
            sweep_changed = False
            for agent in agent_order:
                cap = current[agent]
                cap_lo = self.clamp_cap(cap - step, delta_min[agent], delta_max[agent])
                cap_hi = self.clamp_cap(cap + step, delta_min[agent], delta_max[agent])
                candidates = {cap_lo, cap, cap_hi}

                best_score = np.inf
                for candidate_cap in candidates:
                    score = self.price_score(
                        agent,
                        candidate_cap,
                        price,
                        cbs_node,
                        cache,
                        delta_min[agent],
                        delta_max[agent],
                    )
                    if score <= best_score:
                        best_score = score
                        best_cap = candidate_cap
                if not np.isclose(best_cap, cap, atol=self.market_tie_tol):
                    sweep_changed = True
                    current[agent] = best_cap
            if not sweep_changed or sweeps >= self.market_max_sweeps:
                break
        return current

    def price_score(
        self,
        agent: int,
        cap: float,
        price: float,
        cbs_node: RiskBoundedCBSNode,
        cache: Dict[Tuple[int, float], Union[None, Dict[str, Union[float, List[int]]]]],
        cap_min: float,
        cap_max: float,
    ) -> float:
        clamped = self.clamp_cap(cap, cap_min, cap_max)
        path_info = self.get_or_plan_path(agent, clamped, cbs_node, cache)
        if type(path_info) is MAPFErrorCodes or path_info is None:
            return np.inf
        return path_info["length"] + price * clamped  # type: ignore

    def get_or_plan_path(
        self,
        agent: int,
        risk_budget: float,
        cbs_node: RiskBoundedCBSNode,
        cache: Dict[Tuple[int, float], Union[None, Dict[str, Union[float, List[int]]]]],
    ) -> Union[None, Dict[str, Union[float, List[int]]], MAPFErrorCodes]:
        rounded = self.round_cap(risk_budget)
        key = (agent, rounded)
        if key in cache:
            return cache[key]

        experience = cbs_node.paths[agent] if self.use_experience else None
        agent_path = self.single_agent_planners[agent].find_constrained_path(
            constraints=cbs_node.constraints,
            risk_budget=rounded,
            experience=experience,
            max_time=self.max_time // self.num_agents,
        )
        if type(agent_path) is MAPFErrorCodes:
            cache[key] = None
            error_code = agent_path
            return error_code

        length = self.compute_cost(agent_path)
        risk = self.compute_cost(agent_path, risk=True)
        if risk - rounded > 1e-6:
            cache[key] = None
            return None

        cache[key] = {"length": length, "risk": risk, "path": agent_path}
        return cache[key]

    def risk_optimized_path(
        self,
        agent: int,
        cbs_node: RiskBoundedCBSNode,
        cache: Dict[Tuple[int, float], Union[None, Dict[str, Union[float, List[int]]]]],
    ) -> Union[Tuple[float, float, List[int]], MAPFErrorCodes]:
        experience = cbs_node.paths[agent] if self.use_experience else None
        agent_path = self.single_agent_planners[agent].find_path(
            constraints=cbs_node.constraints,
            experience=experience,
            edge_attribute="cost",
            max_time=self.max_time // self.num_agents,
        )
        if type(agent_path) is MAPFErrorCodes:
            error_code = agent_path
            return error_code

        agent_risk = self.compute_cost(agent_path, risk=True)
        agent_length = self.compute_cost(agent_path)
        self.store_path_info(agent, agent_risk, agent_length, agent_path, cache)
        return agent_risk, agent_length, agent_path

    def length_optimized_path(
        self,
        agent: int,
        cbs_node: RiskBoundedCBSNode,
        cache: Dict[Tuple[int, float], Union[None, Dict[str, Union[float, List[int]]]]],
    ) -> Union[Tuple[float, float, List[int]], MAPFErrorCodes]:
        experience = cbs_node.paths[agent] if self.use_experience else None
        path = self.single_agent_planners[agent].find_path(
            constraints=cbs_node.constraints,
            experience=experience,
            max_time=self.max_time // self.num_agents,
        )
        if type(path) is MAPFErrorCodes:
            return path
        risk = self.compute_cost(path, risk=True)
        length = self.compute_cost(path)
        self.store_path_info(agent, risk, length, path, cache)
        return risk, length, path

    def store_path_info(
        self,
        agent: int,
        risk: float,
        length: float,
        path: List[int],
        cache: Dict[Tuple[int, float], Union[None, Dict[str, Union[float, List[int]]]]],
    ) -> None:
        rounded = self.round_cap(risk)
        cache[(agent, rounded)] = {"length": length, "risk": risk, "path": path}

    def update_paths_from_cache(
        self,
        agents: List[int],
        allocation: Dict[int, float],
        cbs_node: RiskBoundedCBSNode,
        cache: Dict[Tuple[int, float], Union[None, Dict[str, Union[float, List[int]]]]],
    ) -> None:
        for agent in agents:
            info = self.get_or_plan_path(agent, allocation[agent], cbs_node, cache)
            if isinstance(info, dict) and "path" in info:
                cbs_node.paths[agent] = info["path"]  # type: ignore
                cbs_node.paths_found[agent] = True

    def constraint_signature(self, constraints: List[Dict]) -> Tuple:
        def make_hashable(value):
            if isinstance(value, dict):
                return tuple(sorted((k, make_hashable(v)) for k, v in value.items()))
            if isinstance(value, list):
                return tuple(make_hashable(v) for v in value)
            if isinstance(value, tuple):
                return tuple(make_hashable(v) for v in value)
            return value

        return tuple(sorted(make_hashable(constraint) for constraint in constraints))

    def round_cap(self, risk: float) -> float:
        return round(max(0.0, risk), self.market_cache_precision)

    def clamp_cap(self, risk_budget: float, min_risk: float, max_risk: float) -> float:
        return min(max_risk, max(min_risk, risk_budget))

    def push_node(
        self, cbs_node: RiskBoundedCBSNode, risk_allocations_changed: int = 0
    ) -> None:
        heapq.heappush(
            self.open_list,
            (
                cbs_node.cost,
                len(cbs_node.collisions),
                risk_allocations_changed,
                cbs_node,
            ),
        )

        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            self.search_tree.add_node(
                cbs_node.id,
                label="{}->{}".format(cbs_node.id, cbs_node.cost),
                cost=cbs_node.cost,
                paths=str(cbs_node.paths),
                collisions=len(cbs_node.collisions),
                risk_allocation=str(cbs_node.risk_allocation),
            )

        self.num_generated += 1
        if self.num_generated % self.tree_save_frequency == 0:
            self.save_search_tree()

    def classify_collision(self, cbs_node: RiskBoundedCBSNode, collision: Dict) -> str:
        cardinality = "non-cardinal"

        constraints = standard_split(collision)
        for constraint in cbs_node.constraints:
            if constraint not in constraints:
                constraints.append(constraint)

        agent_A = collision["agent_A"]
        alternative_path_A = self.single_agent_planners[agent_A].find_constrained_path(
            constraints=constraints,
            risk_budget=cbs_node.risk_allocation[agent_A],
            experience=cbs_node.paths[agent_A] if self.use_experience else None,
            max_time=self.max_time // self.num_agents,
        )

        if type(alternative_path_A) is MAPFErrorCodes:
            error_code = alternative_path_A
            logging.debug(MAPFError(error_code)["message"])
        else:
            if len(alternative_path_A) > len(cbs_node.paths[agent_A]):  # type: ignore
                cardinality = "semi-cardinal"

        agent_B = collision["agent_B"]
        alternative_path_B = self.single_agent_planners[agent_B].find_constrained_path(
            constraints=constraints,
            risk_budget=cbs_node.risk_allocation[agent_B],
            experience=cbs_node.paths[agent_B] if self.use_experience else None,
            max_time=self.max_time // self.num_agents,
        )

        if type(alternative_path_B) is MAPFErrorCodes:
            error_code = alternative_path_B
            logging.debug(MAPFError(error_code)["message"])
        else:
            if len(alternative_path_B) > len(cbs_node.paths[agent_B]):  # type: ignore
                if cardinality == "semi-cardinal":
                    cardinality = "cardinal"
                else:
                    cardinality = "semi-cardinal"

        return cardinality

    def find_paths(self) -> RiskBoundedCBSNode:
        start_time = time.time()
        logging.debug("Finding paths using Risk Bounded CBS solver")

        # Define the root of the risk-bounded CBS tree
        root = RiskBoundedCBSNode(
            id=0,
            cost=0,
            paths=[],
            collisions=[],
            constraints=[],
            risk_allocation=[],
            paths_found=[False] * self.num_agents,
        )

        # Compute the paths for each agent ignoring the risk constraints
        for agent in range(self.num_agents):
            agent_path = self.single_agent_planners[agent].find_path(
                constraints=root.constraints,
                experience=None,
                max_time=self.max_time // self.num_agents,
            )
            if type(agent_path) is MAPFErrorCodes:
                error_code = agent_path
                raise RuntimeError(
                    MAPFError(MAPFErrorCodes.NO_INIT_PATH, error_code)["message"]
                )

            root.paths.append(agent_path)

        root.cost = self.compute_sum_of_costs(root.paths)
        root.collisions = detect_collisions(
            root.paths, self.pdist, self.collision_radius
        )

        # Compute the risk allocation for each agent based on their current utility
        utility = [self.compute_cost(path) for path in root.paths]
        root.risk_allocation = self.budget_allocator.allocate(root.paths, utility)
        logging.debug("Risk allocation {}".format(root.risk_allocation))

        # The open list tracks nodes for the CBS tree based on how many agents' risk allocation was changed
        # The root's risk allocation is the initial risk allocation so no agents' risk allocation was changed
        # The last element in the tuple keep track of the agents that failed to find paths within their risk allocation

        # Heap is ordered based on the cost of the node, the number of collisions, the number of agent's whose
        # risk allocation was changed, the node id and the node itself
        self.push_node(root, 0)

        # Run the loop till timeout or the open list is empty
        while len(self.open_list) > 0 and time.time() - start_time < self.max_time:

            # The last element in the open-list tuple is the node itself
            current_node = heapq.heappop(self.open_list)[-1]

            logging.debug("Current node ID {}".format(current_node.id))
            self.num_expanded += 1

            # Recompute the paths for the agents with updated risk-allocation
            if not all(current_node.paths_found):

                logging.debug("Computing paths for agents with updated risk-budgets")

                for agent in range(self.num_agents):

                    agent_path = self.single_agent_planners[
                        agent
                    ].find_constrained_path(
                        constraints=current_node.constraints,
                        risk_budget=current_node.risk_allocation[agent],
                        experience=(
                            current_node.paths[agent] if self.use_experience else None
                        ),
                        max_time=self.max_time // self.num_agents,
                    )

                    if type(agent_path) is not MAPFErrorCodes:
                        current_node.paths_found[agent] = True
                        current_node.paths[agent] = agent_path
                    else:
                        logging.debug(
                            MAPFError(MAPFErrorCodes.NO_PATH, agent_path)["message"]
                        )

            # If all agents were able to find paths within their risk allocation then we need to proceed
            # with conflict resolution step of CBS
            if all(current_node.paths_found):
                # If we reach here then the a_star was able to find paths for all the agents
                # within their allocated risk-budget. Now we need to focus on de-conflicting the paths
                current_node.collisions = detect_collisions(
                    current_node.paths, self.pdist, self.collision_radius
                )

                # If there are no conflicts then all the agents' paths are risk-bounded and conflict-free!
                if len(current_node.collisions) == 0:
                    logging.debug(
                        "Found paths with no collisions with node {}".format(
                            current_node
                        )
                    )
                    # Check whether the accumulated risk is within the risk bound
                    accumulated_risk = self.compute_sum_of_costs(
                        current_node.paths, risk=True
                    )
                    logging.debug(
                        "Accumulated risk of the current solution is {}, risk bound {}".format(
                            accumulated_risk, self.risk_bound
                        )
                    )
                    if accumulated_risk < self.risk_bound or np.isclose(
                        accumulated_risk, self.risk_bound, atol=1e-5
                    ):
                        # If the accumulated risk is within the risk bound then we have found a solution
                        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                            self.search_tree.add_node(
                                current_node.id,
                                label="{}->{}".format(
                                    current_node.id, current_node.cost
                                ),
                                color="green",
                                cost=current_node.cost,
                                paths=str(current_node.paths),
                                collisions=len(current_node.collisions),
                            )
                            self.save_search_tree()
                        return current_node

                # Choose a random conflict and split the CBS tree based on the conflict
                collision = self.choose_collision(current_node)
                constraints = self.splitter_function(collision)  # type: ignore

                for constraint in constraints:

                    logging.debug("Tackling constraint {}".format(constraint))
                    # One branch of the CBS tree
                    successor = RiskBoundedCBSNode(
                        id=self.num_generated,
                        cost=0,
                        paths=current_node.paths.copy(),
                        collisions=[],
                        constraints=[constraint],
                        paths_found=current_node.paths_found.copy(),
                        risk_allocation=current_node.risk_allocation.copy(),
                    )

                    # Copy the constraints from the parent node
                    for old_constraint in current_node.constraints:
                        if old_constraint not in successor.constraints:
                            successor.constraints.append(old_constraint)

                    # Find a path for the conflicting agent with its assigned risk-budget
                    constraint_agent = constraint["agent_id"]
                    agent_path = self.single_agent_planners[
                        constraint_agent
                    ].find_constrained_path(
                        constraints=successor.constraints,
                        risk_budget=current_node.risk_allocation[constraint_agent],
                        experience=(
                            current_node.paths[constraint_agent]
                            if self.use_experience
                            else None
                        ),
                        max_time=self.max_time // self.num_agents,
                    )

                    # If the path is found then we need to recompute the paths for the other agents
                    # whose path might be affected by this new path
                    if type(agent_path) is not MAPFErrorCodes:
                        # Update the path of the conflicting agent in this branch's CBS node
                        successor.paths[constraint_agent] = agent_path
                        successor.paths_found[constraint_agent] = True

                        if constraint["positive"]:
                            # Extract the agents whose paths are affected by the new path of the conflicting agent
                            violating_agents = self.extract_violating_agents(
                                successor, constraint
                            )
                            additional_constraints = []
                            for agent, conflict_type in violating_agents:
                                if agent == constraint_agent:
                                    continue

                                violating_constraints = successor.constraints.copy()
                                if conflict_type == "vertex":
                                    ind = 0 if len(constraint["location"]) == 1 else 1
                                    new_constraint = {
                                        "agent_id": agent,
                                        "location": [constraint["location"][ind]],
                                        "timestep": constraint["timestep"],
                                        "positive": False,
                                        "final": False,
                                    }
                                elif conflict_type == "edge":
                                    new_constraint = {
                                        "agent_id": agent,
                                        "location": constraint["location"][::-1],
                                        "timestep": constraint["timestep"],
                                        "positive": False,
                                        "final": False,
                                    }
                                else:
                                    current_location = get_location(
                                        successor.paths[agent], constraint["timestep"]
                                    )
                                    previous_location = get_location(
                                        successor.paths[agent],
                                        constraint["timestep"] - 1,
                                    )
                                    new_constraint = {
                                        "agent_id": agent,
                                        "location": [
                                            previous_location,
                                            current_location,
                                        ],
                                        "timestep": constraint["timestep"],
                                        "positive": False,
                                        "final": False,
                                    }

                                violating_constraints.append(new_constraint)
                                additional_constraints.append(new_constraint)

                                # Recompute their paths with their risk allocation
                                agent_path = self.single_agent_planners[
                                    agent
                                ].find_constrained_path(
                                    # constraints=successor.constraints,
                                    constraints=violating_constraints,
                                    risk_budget=current_node.risk_allocation[agent],
                                    experience=(
                                        current_node.paths[agent]
                                        if self.use_experience
                                        else None
                                    ),
                                    max_time=self.max_time // self.num_agents,
                                )

                                # If their path is found then update their path in this branch's CBS node
                                if type(agent_path) is not MAPFErrorCodes:
                                    logging.debug(
                                        "Path found for violating agent {}".format(
                                            agent
                                        )
                                    )
                                    successor.paths[agent] = agent_path
                                    successor.paths_found[agent] = True
                                else:
                                    # If we cannot find a path for this violating agent then keep track of them so
                                    # that we can recompute the risk allocation and re-insert the node into the open
                                    # list
                                    logging.debug(
                                        "Path was not found for violating agent {}".format(
                                            agent
                                        )
                                    )
                                    successor.paths_found[agent] = False

                            successor.constraints.extend(additional_constraints)
                            # If some of the violating agents were not able to find paths within their risk allocation
                            # then we need to recompute the risk allocation and re-insert the node into the open list
                            # with the current state of the CBS tree stored in the node so that later it can be expanded
                            # to compute the paths for the violating agents with the updated risk-allocation
                            if not all(successor.paths_found):
                                failing_violating_agents = [
                                    agent
                                    for agent, _ in violating_agents
                                    if not successor.paths_found[agent]
                                ]
                                new_allocation = self.reallocate_risk(
                                    failing_violating_agents,
                                    successor,
                                )

                                # If the reallocation of the risk is successful then we need to re-insert the node
                                if type(new_allocation) is not MAPFErrorCodes:
                                    # In this case, path of atleast one agent has been modified
                                    new_allocation, agents_allocation_changed = new_allocation  # type: ignore
                                    logging.debug(
                                        "Violating agents paths not found so reallocated risk {}".format(
                                            new_allocation
                                        )
                                    )
                                    successor.cost = self.compute_sum_of_costs(
                                        successor.paths
                                    )
                                    successor.collisions = detect_collisions(
                                        successor.paths,
                                        self.pdist,
                                        self.collision_radius,
                                    )
                                    successor.risk_allocation = new_allocation
                                    for allocation_changed in agents_allocation_changed:
                                        successor.paths_found[allocation_changed] = (
                                            False
                                        )

                                    self.push_node(
                                        successor, len(agents_allocation_changed)
                                    )

                                    if (
                                        logging.getLogger().getEffectiveLevel()
                                        == logging.DEBUG
                                    ):
                                        changes = "V = {} +C = ({}, {}, {})".format(
                                            violating_agents,
                                            constraint["agent_id"],
                                            constraint["location"],
                                            constraint["timestep"],
                                        )
                                        self.search_tree.add_edge(
                                            current_node.id,
                                            successor.id,
                                            label=changes,
                                        )
                                        logging.debug(
                                            "Generated: {}".format(self.num_generated)
                                        )
                                        logging.debug(
                                            "Edge between {} and {}".format(
                                                current_node.id, successor.id
                                            )
                                        )
                                else:
                                    # If the reallocation of the risk is not successful then we cannot find a solution
                                    # from this branch of the CBS tree that satisfies the risk allocation and the
                                    # constraint imposed on it
                                    error_code = new_allocation
                                    logging.debug(MAPFError(error_code)["message"])

                                    if (
                                        logging.getLogger().getEffectiveLevel()
                                        == logging.DEBUG
                                    ):
                                        self.search_tree.add_node(
                                            successor.id,
                                            label=MAPFError(error_code)["message"],
                                            color="red",
                                        )
                                        self.search_tree.add_edge(
                                            current_node.id,
                                            successor.id,
                                            label="Violating risk reallocation failed",
                                        )
                                        self.num_generated += 1
                                continue

                        # If the constraint was positive and all the violating agents were able to find paths
                        # within their risk allocation and the updated constraint or if the constraint was negative
                        # and the constrained agent was able to find its path with the risk allocation and its own
                        # constraint then we need to update the cost and collisions of the
                        # successor node and insert it into the open list so that later it can be expanded
                        # to resolve other potential conflicts
                        successor.cost = self.compute_sum_of_costs(successor.paths)
                        successor.collisions = detect_collisions(
                            successor.paths,
                            self.pdist,
                            self.collision_radius,
                        )

                        self.push_node(successor)

                        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                            changes = "+" if constraint["positive"] else "-"
                            changes += "C = ({}, {}, {}) satisfied".format(
                                constraint["agent_id"],
                                constraint["location"],
                                constraint["timestep"],
                            )
                            self.search_tree.add_edge(
                                current_node.id, successor.id, label=changes
                            )
                            logging.debug("Generated: {}".format(self.num_generated))
                            logging.debug(
                                "Edge between {} and {}".format(
                                    current_node.id, successor.id
                                )
                            )
                    else:
                        # If the path is not found for the conflicting agent within its risk allocation
                        # then we need to recompute the risk allocation for the failing agent and check if
                        # the failure of the agent is due to the risk allocation. If the failure is due to the
                        # risk allocation then we need to recompute the risk allocation and re-insert the node
                        # into the open list with the current state of the CBS tree stored in the node
                        current_node.paths_found[constraint_agent] = False
                        new_allocation = self.reallocate_risk(
                            [constraint_agent],
                            successor,
                        )

                        # If the reallocation of the risk is successful then we need to re-insert the node
                        if type(new_allocation) is not MAPFErrorCodes:
                            new_allocation, agents_allocation_changed = new_allocation  # type: ignore
                            logging.debug(
                                "Constraint agent's path not found so reallocated risk {}".format(
                                    new_allocation
                                )
                            )
                            successor.cost = current_node.cost
                            successor.risk_allocation = new_allocation
                            successor.collisions = current_node.collisions.copy()
                            for allocation_changed in agents_allocation_changed:
                                successor.paths_found[allocation_changed] = False

                            # Difference between the current node and the successor node is that
                            # the successor node has the updated risk allocation and the paths_found
                            # variable is updated for the agents whose risk allocation was changed
                            # along with an additional constraint that the constrained agent was asked to satisfy
                            self.push_node(successor, len(agents_allocation_changed))

                            if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                                changes = "+" if constraint["positive"] else "-"
                                changes += "C = ({}, {}, {}) not satisfied".format(
                                    constraint["agent_id"],
                                    constraint["location"],
                                    constraint["timestep"],
                                )
                                self.search_tree.add_edge(
                                    current_node.id, successor.id, label=changes
                                )
                                logging.debug(
                                    "Generated: {}".format(self.num_generated)
                                )
                                logging.debug(
                                    "Edge between {} and {}".format(
                                        current_node.id, successor.id
                                    )
                                )
                        else:
                            # If the reallocation of the risk is not successful then we cannot find a solution from
                            # this branch of the CBS tree
                            error_code = new_allocation
                            logging.debug(MAPFError(error_code)["message"])

                            if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                                self.search_tree.add_node(
                                    successor.id,
                                    label=MAPFError(error_code)["message"],
                                    color="red",
                                )
                                self.search_tree.add_edge(
                                    current_node.id,
                                    successor.id,
                                    label="Constraint agent's risk reallocation failed",
                                )
                                self.num_generated += 1
            else:

                # A star call to some agents failed. We need to recompute the risk allocation
                # and re-insert the node into the open list with the current state of the CBS tree
                # stored in the node so that later it can be expanded to compute the paths for the
                # failing agents with the updated risk-allocation

                failing_agents = [
                    agent
                    for agent in range(self.num_agents)
                    if not current_node.paths_found[agent]
                ]
                logging.info("A-Star failed for agents: {}".format(failing_agents))

                # Recompute the risk allocation for the failing agents
                new_allocation = self.reallocate_risk(
                    failing_agents,
                    current_node,
                )

                # If the reallocation of the risk is successful then we need to re-insert the node
                if type(new_allocation) is not MAPFErrorCodes:
                    new_allocation, agents_allocation_changed = new_allocation  # type: ignore
                    successor = current_node.copy()
                    successor.id = self.num_generated
                    successor.risk_allocation = new_allocation
                    for allocation_changed in agents_allocation_changed:
                        successor.paths_found[allocation_changed] = False

                    self.push_node(successor, len(agents_allocation_changed))

                    if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                        changes = "RA failure"
                        self.search_tree.add_edge(
                            current_node.id, successor.id, label=changes
                        )
                        logging.debug("Generated: {}".format(self.num_generated))
                        logging.debug(
                            "Edge between {} and {}".format(
                                current_node.id, successor.id
                            )
                        )
                    logging.debug(
                        "Risk allocation {}".format(successor.risk_allocation)
                    )
                else:
                    # If the reallocation of the risk is not successful then we cannot find a solution from
                    # this branch of the CBS tree that satisfies the risk allocation and the constraint imposed
                    # on it
                    error_code = new_allocation
                    successor = current_node.copy()
                    successor.id = self.num_generated
                    logging.debug(MAPFError(error_code)["message"])

                    if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                        self.search_tree.add_node(
                            successor.id,
                            label=MAPFError(error_code)["message"],
                            color="red",
                        )
                        self.search_tree.add_edge(
                            current_node.id,
                            successor.id,
                            label="Risk reallocation failed",
                        )
                        self.num_generated += 1

        # If we terminate the search due to timeout the explicitly return the timeout error code
        # otherwise return the no path error code
        if time.time() - start_time >= self.max_time:
            raise RuntimeError(MAPFError(MAPFErrorCodes.TIMELIMIT_REACHED)["message"])
        else:
            raise RuntimeError(MAPFError(MAPFErrorCodes.NO_PATH)["message"])
