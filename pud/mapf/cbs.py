import os
import time
import heapq
import random
import logging
import numpy as np
import networkx as nx
from pathlib import Path
from networkx import Graph
from numpy.typing import NDArray
from typing import List, Dict, Tuple

from pud.mapf.single_agent_planner import AStar
from pud.mapf.mapf_exceptions import MAPFError, MAPFErrorCodes
from pud.mapf.utils import detect_collisions, get_location, intersection_check, standard_split, disjoint_split


class CBSNode(object):
    def __init__(
        self,
        id: int,
        cost: float,
        paths: List[List[int]],
        collisions: List[Dict],
        constraints: List[Dict],
    ):
        self.id = id
        self.cost = cost
        self.paths = paths
        self.collisions = collisions
        self.constraints = constraints

    def __lt__(self, other):
        if self.cost == other.cost:
            if len(self.collisions) == len(other.collisions):
                return self.id < other.id
            return len(self.collisions) < len(other.collisions)
        return self.cost < other.cost

    def copy(self):
        return CBSNode(
            id=self.id,
            cost=self.cost,
            paths=self.paths.copy(),
            collisions=self.collisions.copy(),
            constraints=self.constraints.copy(),
        )


class CBSSolver(object):

    def __init__(
        self,
        graph: Graph,
        goals: List[int],
        starts: List[int],
        pdist: NDArray,
        config: Dict,
    ):

        self.problem_config = config

        if config["seed"] is not None:
            random.seed(config["seed"])
        else:
            random.seed(0)

        self.pdist = pdist
        self.graph = graph

        self.goals = goals
        self.starts = starts
        self.num_agents = len(starts)

        self.num_expanded = 0
        self.num_generated = 0

        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            self.logdir = config["logdir"]
            if not os.path.exists(self.logdir):
                os.makedirs(self.logdir)
            self.tree_save_frequency = config["tree_save_frequency"]
            # Use for debugging purposes
            self.search_tree = nx.DiGraph()
        else:
            self.tree_save_frequency = np.inf

        self.max_time = config["max_time"]
        self.splitter = config["split_strategy"]
        self.use_experience = config["use_experience"]
        self.use_cardinality = config["use_cardinality"]
        self.edge_attributes = config["edge_attributes"]
        self.collision_radius = config["collision_radius"]

        self.risk_attribute = config["risk_attribute"]
        assert (
            len(nx.get_edge_attributes(self.graph, self.risk_attribute)) > 0
        ), "Risk attribute {} not found in the graph".format(self.risk_attribute)

        self.open_list = []

        self.splitter_function = None
        if self.splitter == "standard":
            self.splitter_function = standard_split
        elif self.splitter == "disjoint":
            self.splitter_function = disjoint_split
        else:
            raise RuntimeError(MAPFError(MAPFErrorCodes.INVALID_SPLITTER)["message"])

        # Add self-loops
        for node, neighbors in self.graph.adjacency():
            # ASSUMPTION 1: The cost of waiting at a node is the minimum risk of reaching this node from its neighbors
            min_cost = min((data[self.risk_attribute] for data in neighbors.values()), default=np.inf)
            self.graph.add_edge(node, node, weight=0, step=1, cost=min_cost)

        self.make_planners(config)

    def make_planners(self, config: Dict) -> None:
        self.single_agent_planners = {}
        for agent in range(self.num_agents):
            self.single_agent_planners[agent] = AStar(
                config=config,
                agent_id=agent,
                graph=self.graph,
                goal=self.goals[agent],
                start=self.starts[agent],
            )

    def compute_cost(self, path: List[int], risk: bool = False) -> float:
        cost = 0.0
        for i in range(len(path) - 1):
            cost += self.graph[path[i]][path[i + 1]][self.risk_attribute] if risk else 1
        return cost

    def compute_sum_of_costs(self, paths: List[List[int]], risk: bool = False) -> float:
        sum_of_costs = 0
        for path in paths:
            sum_of_costs += self.compute_cost(path, risk)
        return sum_of_costs

    def push_node(self, cbs_node: CBSNode) -> None:
        heapq.heappush(
            self.open_list,
            (cbs_node.cost, len(cbs_node.collisions), cbs_node),
        )

        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            self.search_tree.add_node(
                cbs_node.id,
                label="{}->{}".format(cbs_node.id, cbs_node.cost),
                cost=cbs_node.cost,
                paths=str(cbs_node.paths),
                collisions=len(cbs_node.collisions),
            )

        self.num_generated += 1
        if self.num_generated % self.tree_save_frequency == 0:
            self.save_search_tree()

    def extract_violating_agents(
        self, successor: CBSNode, constraint: Dict
    ) -> List[Tuple]:

        violating_agents = set()
        for agent in range(self.num_agents):

            if agent == constraint["agent_id"]:
                continue

            current_location = get_location(
                successor.paths[agent], constraint["timestep"]
            )
            previous_location = get_location(
                successor.paths[agent],
                constraint["timestep"] - 1,
            )

            if len(constraint["location"]) == 1:  # Vertex constraint
                if constraint["location"][0] == current_location:
                    violating_agents.add((agent, "vertex"))
            else:
                successor_path_location = [
                    previous_location,
                    current_location,
                ]
                if (constraint["location"] == successor_path_location[::-1]):
                    violating_agents.add((agent, "edge"))
                elif (constraint["location"][0] == successor_path_location[0]
                      or constraint["location"][1] == successor_path_location[1]):
                    violating_agents.add((agent, "vertex"))
                elif intersection_check(constraint["location"][0], constraint["location"][1],
                                        previous_location, current_location,
                                        self.pdist, agent_radius=self.collision_radius):
                    violating_agents.add((agent, "intersection"))

        logging.debug("Violating agents are {}".format(violating_agents))
        return list(violating_agents)

    def save_search_tree(self) -> None:
        filepath = Path(self.logdir) / f"search_tree_step_{self.num_generated}.dot"
        nx.drawing.nx_pydot.write_dot(self.search_tree, filepath)
        logging.info(f"Search tree saved to {filepath}")

    def choose_collision(self, cbs_node: CBSNode) -> Dict:
        if self.use_cardinality:
            collision_types = self.classify_collisions(cbs_node)
            if "cardinal" in collision_types:
                return cbs_node.collisions[collision_types.index("cardinal")]
            elif "semi-cardinal" in collision_types:
                return cbs_node.collisions[collision_types.index("semi-cardinal")]
        collision = random.choice(cbs_node.collisions)
        return collision

    def classify_collisions(self, cbs_node: CBSNode) -> List[str]:
        collision_types = []
        for collision in cbs_node.collisions:
            collision_type = self.classify_collision(cbs_node, collision)
            collision_types.append(collision_type)
        return collision_types

    def classify_collision(self, cbs_node: CBSNode, collision: Dict) -> str:
        cardinality = "non-cardinal"

        constraints = standard_split(collision)
        for constraint in cbs_node.constraints:
            if constraint not in constraints:
                constraints.append(constraint)

        agent_A = collision["agent_A"]
        alternative_path_A = self.single_agent_planners[agent_A].find_path(
            constraints=constraints,
            edge_attribute=self.edge_attributes[0],
            max_time=self.max_time // self.num_agents,
            experience=cbs_node.paths[agent_A] if self.use_experience else None,
        )

        if type(alternative_path_A) is MAPFErrorCodes:
            error_code = alternative_path_A
            logging.debug(MAPFError(error_code)["message"])
        else:
            if len(alternative_path_A) > len(cbs_node.paths[agent_A]):  # type: ignore
                cardinality = "semi-cardinal"

        agent_B = collision["agent_B"]
        alternative_path_B = self.single_agent_planners[agent_B].find_path(
            constraints=constraints,
            edge_attribute=self.edge_attributes[0],
            max_time=self.max_time // self.num_agents,
            experience=cbs_node.paths[agent_B] if self.use_experience else None,
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

    def find_paths(self) -> CBSNode:
        self.start_time = time.time()
        logging.debug("Finding paths using CBS Solver")

        root = CBSNode(
            id=0,
            cost=0,
            paths=[],
            collisions=[],
            constraints=[],
        )

        for agent_id in range(self.num_agents):
            agent_path = self.single_agent_planners[agent_id].find_path(
                experience=None,
                constraints=root.constraints,
                edge_attribute=self.edge_attributes[0],
                max_time=self.max_time // self.num_agents,
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
            logging.debug("Collision: {}".format(collision))
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

                seen = {}
                conflict = False
                for c in successor.constraints:
                    key = (c["agent_id"], tuple(c["location"]), c["timestep"])
                    if key in seen and seen[key] != c["positive"]:
                        conflict = True
                        break
                    seen[key] = c["positive"]
                if conflict:
                    logging.debug(f"Skipping successor {successor.id} due to contradictory constraints")
                    continue

                constraint_agent = constraint["agent_id"]
                agent_path = self.single_agent_planners[constraint_agent].find_path(
                    constraints=successor.constraints,
                    edge_attribute=self.edge_attributes[0],
                    max_time=self.max_time // self.num_agents,
                    experience=current_node.paths[constraint_agent] if self.use_experience else None,
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

                            agent_path = self.single_agent_planners[agent].find_path(
                                # constraints=successor.constraints,
                                constraints=violating_constraints,
                                edge_attribute=self.edge_attributes[0],
                                max_time=self.max_time // self.num_agents,
                                experience=current_node.paths[agent] if self.use_experience else None,
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
                        if len(successor.collisions) > len(current_node.collisions):
                            continue
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
