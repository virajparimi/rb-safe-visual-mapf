import heapq
import logging
import time
from networkx import Graph
from typing import Dict, List
from numpy.typing import NDArray

from pud.mapf.cbs import CBSNode, CBSSolver
from pud.mapf.mapf_exceptions import MAPFError, MAPFErrorCodes
from pud.mapf.single_agent_planner import RiskBudgetedAStar
from pud.mapf.utils import detect_collisions, get_location


class PathConstrainedCBSSolver(CBSSolver):

    def __init__(
        self,
        graph: Graph,
        goals: List[int],
        starts: List[int],
        pdist: NDArray,
        config: Dict,
    ):

        self.risk_budget = config["risk_budget"]
        super().__init__(graph, goals, starts, pdist, config)

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

    def find_paths(self) -> CBSNode:
        self.start_time = time.time()
        logging.debug("Finding paths using Path-Constrained CBS Solver")

        root = CBSNode(
            id=0,
            cost=0,
            paths=[],
            collisions=[],
            constraints=[],
        )

        for agent_id in range(self.num_agents):
            agent_path = self.single_agent_planners[agent_id].find_constrained_path(
                constraints=root.constraints,
                experience=None,
                risk_budget=self.risk_budget / self.num_agents,
                max_time=self.max_time // self.num_agents
            )

            if type(agent_path) is MAPFErrorCodes:
                error_code = agent_path
                raise RuntimeError(MAPFError(MAPFErrorCodes.NO_INIT_PATH, error_code)["message"])

            root.paths.append(agent_path)

        root.cost = self.compute_sum_of_costs(root.paths)
        root.collisions = detect_collisions(
            root.paths, self.pdist, self.collision_radius
        )
        self.push_node(root)

        while len(self.open_list) > 0 and time.time() - self.start_time < self.max_time:

            current_node = heapq.heappop(self.open_list)[-1]

            logging.debug("Current node ID {}".format(current_node.id))
            self.num_expanded += 1

            if len(current_node.collisions) == 0:
                if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                    self.search_tree.add_node(
                        current_node.id,
                        label="{}->{}".format(current_node.id, current_node.cost),
                        color="green",
                        cost=current_node.cost,
                        paths=str(current_node.paths),
                        collisions=len(current_node.collisions),
                    )
                    self.save_search_tree()
                return current_node

            collision = self.choose_collision(current_node)
            constraints = self.splitter_function(collision)  # type: ignore

            for constraint in constraints:

                logging.debug("Tackling constraint {}".format(constraint))
                successor = CBSNode(
                    id=self.num_generated,
                    cost=0,
                    paths=current_node.paths.copy(),
                    collisions=[],
                    constraints=[constraint],
                )

                for old_constraint in current_node.constraints:
                    if old_constraint not in successor.constraints:
                        successor.constraints.append(old_constraint)

                constraint_agent = constraint["agent_id"]
                agent_path = self.single_agent_planners[constraint_agent].find_constrained_path(
                    constraints=successor.constraints,
                    experience=current_node.paths[constraint_agent] if self.use_experience else None,
                    risk_budget=self.risk_budget / self.num_agents,
                    max_time=self.max_time // self.num_agents
                )

                skip = False
                if type(agent_path) is MAPFErrorCodes:
                    error_code = agent_path
                    logging.debug("Constraint agent failed to find a path. So skipping it")

                    if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                        changes = "+" if constraint["positive"] else "-"
                        changes += "C = ({}, {}, {}) not satisfied".format(
                            constraint["agent_id"],
                            constraint["location"],
                            constraint["timestep"],
                        )
                        changes += "\n|C| = {}".format(len(successor.collisions))
                        self.search_tree.add_node(
                            successor.id,
                            label="Constraint agent path not found",
                            color="red"
                        )
                        self.search_tree.add_edge(
                            current_node.id, successor.id, label=changes
                        )
                        self.num_generated += 1
                else:
                    successor.paths[constraint_agent] = agent_path

                    if constraint["positive"]:

                        violating_agents = self.extract_violating_agents(
                            successor, constraint
                        )

                        for agent, conflict_type in violating_agents:
                            if agent == constraint_agent:
                                continue

                            violating_constraints = successor.constraints.copy()
                            if conflict_type == "vertex":
                                ind = 0 if len(constraint["location"]) == 1 else 1
                                violating_constraints.append({
                                    "agent_id": agent,
                                    "location": [constraint["location"][ind]],
                                    "timestep": constraint["timestep"],
                                    "positive": False,
                                    "final": False
                                })
                            elif conflict_type == "edge":
                                violating_constraints.append({
                                    "agent_id": agent,
                                    "location": constraint["location"][::-1],
                                    "timestep": constraint["timestep"],
                                    "positive": False,
                                    "final": False
                                })
                            else:
                                current_location = get_location(
                                    successor.paths[agent], constraint["timestep"]
                                )
                                previous_location = get_location(
                                    successor.paths[agent],
                                    constraint["timestep"] - 1,
                                )
                                violating_constraints.append({
                                    "agent_id": agent,
                                    "location": [previous_location, current_location],
                                    "timestep": constraint["timestep"],
                                    "positive": False,
                                    "final": False
                                })

                            agent_path = self.single_agent_planners[agent].find_constrained_path(
                                # constraints=successor.constraints,
                                constraints=violating_constraints,
                                experience=current_node.paths[agent] if self.use_experience else None,
                                risk_budget=self.risk_budget / self.num_agents,
                                max_time=self.max_time // self.num_agents
                            )

                            if type(agent_path) is MAPFErrorCodes:
                                error_code = agent_path
                                logging.debug("Violating agent {} failed to find a path. So skipping it".format(agent))
                                if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                                    changes = "+" if constraint["positive"] else "-"
                                    changes += "C = ({}, {}, {}) not satisfied".format(
                                        constraint["agent_id"],
                                        constraint["location"],
                                        constraint["timestep"],
                                    )
                                    changes += "\n|C| = {}".format(len(successor.collisions))
                                    self.search_tree.add_node(
                                        successor.id,
                                        label="Violating agent path not found",
                                        color="red"
                                    )
                                    self.search_tree.add_edge(
                                        current_node.id, successor.id, label=changes
                                    )
                                    self.num_generated += 1
                                skip = True
                                break
                            else:
                                successor.paths[agent] = agent_path
                                successor.constraints = violating_constraints

                    if not skip:
                        successor.collisions = detect_collisions(
                            successor.paths, self.pdist, self.collision_radius
                        )
                        successor.cost = self.compute_sum_of_costs(successor.paths)
                        self.push_node(successor)

                        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                            changes = "+" if constraint["positive"] else "-"
                            changes += "C = ({}, {}, {}) satisfied".format(
                                constraint["agent_id"],
                                constraint["location"],
                                constraint["timestep"],
                            )
                            changes += "\n|C| = {}".format(len(successor.collisions))
                            self.search_tree.add_edge(
                                current_node.id, successor.id, label=changes
                            )
                        logging.debug("Generated: {}".format(self.num_generated))

        if time.time() - self.start_time > self.max_time:
            raise RuntimeError(MAPFError(MAPFErrorCodes.TIMELIMIT_REACHED)["message"])
        else:
            raise RuntimeError(MAPFError(MAPFErrorCodes.NO_PATH)["message"])
