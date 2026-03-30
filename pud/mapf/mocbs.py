# Code adapted from https://github.com/wonderren/public_pymomapf/blob/master/libmomapf/mocbs.py
# Choosing to perform the tree-by-tree expansion strategy
from __future__ import annotations
import os
import time
import copy
import random
import logging
import itertools
import numpy as np
import networkx as nx
from pathlib import Path
from networkx import Graph
from numpy.typing import NDArray
from typing import List, Dict, Tuple

from pud.mapf.cbs import CBSNode
from pud.mapf.utils import (
    PrioritySet,
    get_location,
    less_dominant,
    standard_split,
    disjoint_split,
    detect_collisions,
    dominate_or_equal,
)
from pud.mapf.single_agent_planner import MultiObjectiveAStar


class MultiObjectiveCBSNode(CBSNode):

    def __init__(
        self,
        id: int,
        root: int,
        cost: float,
        cost_vector: NDArray,
        paths: List[List[int]],
        collisions: List[Dict],
        constraints: List[Dict],
    ):
        super().__init__(id, cost, paths, collisions, constraints)
        self.time = None
        self.root = root
        self.cost_vector = cost_vector

    def __lt__(self, other):
        if self.cost == other.cost:
            if len(self.collisions) == len(other.collisions):
                return self.id < other.id
            return len(self.collisions) < len(other.collisions)
        return self.cost < other.cost

    def copy(self):
        return MultiObjectiveCBSNode(
            id=self.id,
            cost=self.cost,
            root=self.root,
            paths=self.paths.copy(),
            collisions=self.collisions.copy(),
            cost_vector=self.cost_vector.copy(),
            constraints=self.constraints.copy(),
        )


class MultiObjectiveCBSSolver(object):
    def __init__(
        self,
        graph: Graph,
        starts: List[int],
        goals: List[int],
        pdist: NDArray,
        config: Dict,
    ):
        if config["seed"] is not None:
            random.seed(config["seed"])
        else:
            random.seed(0)

        self.graph = graph

        self.goals = goals
        self.starts = starts
        self.num_agents = len(starts)

        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            self.logdir = config["logdir"]
            if not os.path.exists(self.logdir):
                os.makedirs(self.logdir)
            self.tree_save_frequency = config["tree_save_frequency"]
            # Use for debugging purposes
            self.search_tree = nx.DiGraph()
        else:
            self.tree_save_frequency = np.inf

        self.edge_attributes = config["edge_attributes"]
        self.cost_dimension = len(self.edge_attributes)

        self.open_list = PrioritySet()
        self.open_by_tree = {}

        self.nodes = {}
        self.nondominant_goal_nodes = set()

        self.num_expanded = 0
        self.num_generated = 0
        self.roots_generated = 0

        self.pdist = pdist

        self.max_time = config["max_time"]
        self.splitter = config["split_strategy"]
        self.use_cardinality = config["use_cardinality"]
        self.collision_radius = config["collision_radius"]

        self.splitter_function = None
        if self.splitter == "standard":
            self.splitter_function = standard_split
        elif self.splitter == "disjoint":
            self.splitter_function = disjoint_split
        else:
            raise RuntimeError("Invalid splitter strategy")

        self.risk_attribute = config["risk_attribute"]
        assert (
            len(nx.get_edge_attributes(self.graph, self.risk_attribute)) > 0
        ), "Risk attribute {} not found in the graph".format(self.risk_attribute)

        # Add self-loops
        for node, neighbors in self.graph.adjacency():
            # ASSUMPTION 1: The cost of waiting at a node is the minimum risk of reaching this node from its neighbors
            min_cost = min((data[self.risk_attribute] for data in neighbors.values()), default=np.inf)
            self.graph.add_edge(node, node, weight=0, step=1, cost=min_cost)

        self.single_agent_planner_times = []
        self.single_agent_planner_expansions = []

        self.make_planners(config)

    def make_planners(self, config: Dict) -> None:
        self.single_agent_planners = {}
        for agent in range(self.num_agents):
            self.single_agent_planners[agent] = MultiObjectiveAStar(
                config=config,
                agent_id=agent,
                graph=self.graph,
                goal=self.goals[agent],
                start=self.starts[agent],
            )

    def init_search_on_demand(self) -> bool:
        self.pareto_individual_paths = {}

        for agent in range(self.num_agents):
            self.pareto_individual_paths[agent] = []

            plan_start_time = time.time()
            pareto_paths, _ = self.single_agent_planners[agent].find_path(
                constraints=[], max_time=self.max_time // self.num_agents
            )
            plan_end_time = time.time()
            self.single_agent_planner_times.append(plan_end_time - plan_start_time)
            self.single_agent_planner_expansions.append(
                self.single_agent_planners[agent].num_expanded
            )

            for key in pareto_paths:
                self.pareto_individual_paths[agent].append(pareto_paths[key])

        self.num_roots = 1
        for agent in range(self.num_agents):
            self.num_roots *= len(self.pareto_individual_paths[agent])

        self.init_tree_indices = {agent: 0 for agent in range(self.num_agents)}

        if not self.gen_root_on_demand()[0]:
            return False

        self.update_indices_on_demand()
        return True

    def compute_cost_vector(self, paths: List[List[int]]) -> NDArray:
        cost_vector = np.zeros(self.cost_dimension, dtype=np.float32)
        for path in paths:
            for i in range(len(path) - 1):
                edge = (path[i], path[i + 1])
                for idx, edge_attribute in enumerate(self.edge_attributes):
                    edge_cost = np.array(self.graph.edges[edge][edge_attribute], dtype=np.float32)
                    if edge_attribute == "step" and edge_cost == 0:
                        edge_cost = 1
                    cost_vector[idx] += edge_cost
        return cost_vector

    def gen_root_on_demand(self) -> Tuple[bool, int]:

        if self.init_tree_indices[self.num_agents - 1] >= len(
            self.pareto_individual_paths[self.num_agents - 1]
        ):
            return False, -1

        root_node = MultiObjectiveCBSNode(
            id=self.num_generated,
            root=self.num_generated,
            cost=0,
            cost_vector=np.zeros(self.cost_dimension, dtype=np.float32),
            paths=[],
            collisions=[],
            constraints=[],
        )
        for agent in range(self.num_agents):
            agent_path = self.pareto_individual_paths[agent][
                self.init_tree_indices[agent]
            ]
            root_node.paths.append(agent_path)
        root_node.cost_vector = self.compute_cost_vector(root_node.paths)
        root_node.cost = np.sum(root_node.cost_vector)

        root_node.collisions = detect_collisions(
            root_node.paths, self.pdist, self.collision_radius
        )

        self.open_by_tree[root_node.id] = PrioritySet()
        self.open_list.add(
            (root_node.cost, len(root_node.collisions), root_node), root_node.id
        )
        self.open_by_tree[root_node.id].add(
            (root_node.cost, len(root_node.collisions), root_node), root_node.id
        )

        if root_node.root == 0:

            if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                self.search_tree.add_node(
                    root_node.id,
                    label="{}->{}".format(root_node.id, root_node.cost),
                    cost=root_node.cost,
                    cost_vector=str(root_node.cost_vector),
                    paths=str(root_node.paths),
                    collisions=len(root_node.collisions),
                )

        self.num_generated += 1
        self.roots_generated += 1
        self.nodes[root_node.id] = root_node

        return True, root_node.id

    def update_indices_on_demand(self) -> None:
        self.init_tree_indices[0] = self.init_tree_indices[0] + 1
        index = 0
        while index < self.num_agents - 1 and self.init_tree_indices[index] >= len(
            self.pareto_individual_paths[index]
        ):
            self.init_tree_indices[index] = 0
            self.init_tree_indices[index + 1] = self.init_tree_indices[index + 1] + 1
            index += 1

    def select_node_on_demand(self) -> Tuple[bool, int]:
        if self.open_by_tree[self.current_root].size() == 0:
            # Move to the next tree

            if not self.gen_root_on_demand()[0]:
                return False, -1

            self.update_indices_on_demand()

            self.open_by_tree.pop(self.current_root)
            for tree in self.open_by_tree:
                self.current_root = tree
                break

        _, popped_node_id = self.open_by_tree[self.current_root].pop()
        self.open_list.remove(popped_node_id)
        return True, popped_node_id

    def filter_goal_node(self, node: MultiObjectiveCBSNode) -> bool:
        for goal_node_id in self.nondominant_goal_nodes:
            if dominate_or_equal(
                self.nodes[goal_node_id].cost_vector, node.cost_vector
            ):
                return True
        return False

    def sort_paths_lexicographically(self, paths: Dict, cost_vector: Dict) -> List:
        output = []
        pq = PrioritySet()
        for id in cost_vector:
            pq.add(tuple(cost_vector[id]), id)
        while pq.size() > 0:
            _, id = pq.pop()
            output.append(paths[id])
        return output

    def refine_nondominant_goal_nodes(self, node_id: int) -> None:
        temporary_set = copy.deepcopy(self.nondominant_goal_nodes)
        for goal_node_id in temporary_set:
            if dominate_or_equal(
                self.nodes[node_id].cost_vector, self.nodes[goal_node_id].cost_vector
            ):
                self.nondominant_goal_nodes.remove(goal_node_id)

    def extract_violating_agents(
        self, successor: MultiObjectiveCBSNode, constraint: Dict
    ) -> List[int]:

        violating_agents = []
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

            if len(constraint["location"]) == 1:
                if constraint["location"][0] == current_location:
                    violating_agents.append(agent)
            else:
                successor_path_location = [
                    previous_location,
                    current_location,
                ]
                if (
                    constraint["location"] == successor_path_location[::-1]
                    or constraint["location"][0] == successor_path_location[0]
                    or constraint["location"][1] == successor_path_location[1]
                ):
                    violating_agents.append(agent)

        logging.debug("Violating agents are {}".format(violating_agents))
        return violating_agents

    def save_search_tree(self) -> None:
        filepath = Path(self.logdir) / f"search_tree_step_{self.num_generated}.dot"
        nx.drawing.nx_pydot.write_dot(self.search_tree, filepath)
        logging.info(f"Search tree saved to {filepath}")

    def choose_collision(self, node: MultiObjectiveCBSNode) -> Dict:
        if self.use_cardinality:
            collision_types = self.classify_collisions(node)
            if "cardinal" in collision_types:
                return node.collisions[collision_types.index("cardinal")]
            elif "semi-cardinal" in collision_types:
                return node.collisions[collision_types.index("semi-cardinal")]
        collision = random.choice(node.collisions)
        return collision

    def classify_collisions(self, node: MultiObjectiveCBSNode) -> List[str]:
        collision_types = []
        for collision in node.collisions:
            collision_type = self.classify_collision(node, collision)
            collision_types.append(collision_type)
        return collision_types

    def classify_collision(self, node: MultiObjectiveCBSNode, collision: Dict) -> str:
        cardinality = "non-cardinal"

        constraints = standard_split(collision)
        for constraint in node.constraints:
            if constraint not in constraints:
                constraints.append(constraint)

        agent_A = collision["agent_A"]
        _, path_A_cost_vectors = self.single_agent_planners[agent_A].find_path(
            constraints=constraints,
            max_time=self.max_time // self.num_agents,
        )

        node_path_A_cost_vector = self.compute_cost_vector([node.paths[agent_A]])
        for cost_vector in path_A_cost_vectors:
            if less_dominant(path_A_cost_vectors[cost_vector], node_path_A_cost_vector):
                cardinality = "cardinal"
                break

        agent_B = collision["agent_B"]
        _, path_B_cost_vectors = self.single_agent_planners[agent_B].find_path(
            constraints=constraints,
            max_time=self.max_time // self.num_agents,
        )

        node_path_B_cost_vector = self.compute_cost_vector([node.paths[agent_B]])
        for cost_vector in path_B_cost_vectors:
            if less_dominant(path_B_cost_vectors[cost_vector], node_path_B_cost_vector):
                if cardinality == "semi-cardinal":
                    cardinality = "cardinal"
                else:
                    cardinality = "semi-cardinal"

        return cardinality

    def find_paths(self) -> Tuple[Dict, Dict, bool]:
        start_time = time.time()

        init_success = self.init_search_on_demand()

        if not init_success:
            return {}, {}, False

        for tree in self.open_by_tree:
            self.current_root = tree
            break

        while time.time() - start_time < self.max_time:
            success, current_node_id = self.select_node_on_demand()
            if not success:
                break
            self.num_expanded += 1

            current_node = self.nodes[current_node_id]
            logging.debug("Expanding node {}".format(current_node_id))
            logging.debug("Tree root id: {}".format(current_node.root))
            if len(current_node.collisions) == 0:
                if len(self.nondominant_goal_nodes) == 0:
                    print("First solution time (s): ", time.time() - start_time)
                self.refine_nondominant_goal_nodes(current_node.id)
                current_node.time = time.time() - start_time
                self.nondominant_goal_nodes.add(current_node.id)

                if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                    self.search_tree.add_node(
                        current_node.id,
                        label="{}->{}".format(current_node.id, current_node.cost),
                        color="green",
                        cost=current_node.cost,
                        cost_vector=str(current_node.cost_vector),
                        paths=str(current_node.paths),
                        collisions=len(current_node.collisions),
                    )
                    self.save_search_tree()
                continue

            if self.filter_goal_node(current_node):
                continue

            collision = self.choose_collision(current_node)
            constraints = self.splitter_function(collision)  # type: ignore

            for constraint in constraints:
                logging.debug("Tackling constraint {}".format(constraint))
                constraint_agent = constraint["agent_id"]

                successor = MultiObjectiveCBSNode(
                    id=self.num_generated,
                    root=current_node.root,
                    cost=0,
                    cost_vector=np.zeros(self.cost_dimension, dtype=np.float32),
                    paths=current_node.paths.copy(),
                    collisions=[],
                    constraints=[constraint],
                )

                for old_constraint in current_node.constraints:
                    if old_constraint not in successor.constraints:
                        successor.constraints.append(old_constraint)

                constraint_agent = constraint["agent_id"]

                plan_start_time = time.time()
                agent_paths, cost_vectors = self.single_agent_planners[
                    constraint_agent
                ].find_path(
                    constraints=successor.constraints,
                    max_time=self.max_time // self.num_agents,
                )
                plan_end_time = time.time()
                self.single_agent_planner_times.append(plan_end_time - plan_start_time)
                self.single_agent_planner_expansions.append(
                    self.single_agent_planners[constraint_agent].num_expanded
                )

                agent_paths = self.sort_paths_lexicographically(
                    agent_paths, cost_vectors
                )
                if len(agent_paths) == 0:
                    logging.debug("No path found for agent {}".format(constraint_agent))

                num_gen_before = self.num_generated
                for agent_path in agent_paths:
                    new_successor = successor.copy()
                    new_successor.id = self.num_generated
                    new_successor.paths[constraint_agent] = agent_path

                    skip = False
                    if constraint["positive"]:
                        violating_agents = self.extract_violating_agents(
                            new_successor, constraint
                        )
                        violating_agent_paths = {}
                        for v_agent in violating_agents:
                            if v_agent == constraint_agent:
                                continue

                            plan_start_time = time.time()
                            (
                                v_agent_paths,
                                v_cost_vectors,
                            ) = self.single_agent_planners[v_agent].find_path(
                                constraints=new_successor.constraints,
                                max_time=self.max_time // self.num_agents,
                            )
                            plan_end_time = time.time()
                            self.single_agent_planner_times.append(
                                plan_end_time - plan_start_time
                            )
                            self.single_agent_planner_expansions.append(
                                self.single_agent_planners[v_agent].num_expanded
                            )

                            v_agent_paths = self.sort_paths_lexicographically(
                                v_agent_paths, v_cost_vectors
                            )

                            if len(v_agent_paths) == 0:
                                skip = True
                                break
                            else:
                                violating_agent_paths[v_agent] = v_agent_paths

                    if not skip:

                        if constraint["positive"]:
                            cartesian_product = list(
                                itertools.product(*violating_agent_paths.values())
                            )
                            combination = [
                                dict(zip(violating_agent_paths.keys(), x))
                                for x in cartesian_product
                            ]

                            for violating_agent_combination in combination:
                                updated_successor = new_successor.copy()
                                updated_successor.id = self.num_generated
                                for v_agent in violating_agent_combination:
                                    updated_successor.paths[v_agent] = (
                                        violating_agent_combination[v_agent]
                                    )

                                updated_successor.cost_vector = (
                                    self.compute_cost_vector(updated_successor.paths)
                                )
                                updated_successor.cost = np.sum(
                                    updated_successor.cost_vector, dtype=np.float32
                                )
                                if self.filter_goal_node(updated_successor):
                                    continue

                                updated_successor.collisions = detect_collisions(
                                    updated_successor.paths,
                                    self.pdist,
                                    self.collision_radius,
                                )

                                self.nodes[self.num_generated] = updated_successor
                                self.num_generated += 1

                                if self.num_generated % self.tree_save_frequency == 0:
                                    self.save_search_tree()

                                self.open_list.add(
                                    (
                                        updated_successor.cost,
                                        len(updated_successor.collisions),
                                        updated_successor,
                                    ),
                                    updated_successor.id,
                                )
                                self.open_by_tree[updated_successor.root].add(
                                    (
                                        updated_successor.cost,
                                        len(updated_successor.collisions),
                                        updated_successor,
                                    ),
                                    updated_successor.id,
                                )
                                logging.debug(
                                    "Generated node {}".format(updated_successor.id)
                                )

                                if (
                                    updated_successor.root == 0
                                    and logging.getLogger().getEffectiveLevel()
                                    == logging.DEBUG
                                ):
                                    self.search_tree.add_node(
                                        updated_successor.id,
                                        label="{}->{}".format(
                                            updated_successor.id, updated_successor.cost
                                        ),
                                        cost=updated_successor.cost,
                                        cost_vector=str(updated_successor.cost_vector),
                                        paths=str(updated_successor.paths),
                                        collisions=len(updated_successor.collisions),
                                    )
                                    changes = "+C = ({}, {}, {}) satisfied".format(
                                        constraint["agent_id"],
                                        constraint["location"],
                                        constraint["timestep"],
                                    )
                                    changes += "\n|C| = {}".format(
                                        len(updated_successor.collisions)
                                    )
                                    self.search_tree.add_edge(
                                        current_node.id,
                                        updated_successor.id,
                                        label=changes,
                                    )
                        else:
                            new_successor.cost_vector = self.compute_cost_vector(
                                new_successor.paths
                            )
                            new_successor.cost = np.sum(new_successor.cost_vector)

                            new_successor.collisions = detect_collisions(
                                new_successor.paths,
                                self.pdist,
                                self.collision_radius,
                            )

                            if self.filter_goal_node(new_successor):
                                continue

                            self.nodes[self.num_generated] = new_successor
                            self.num_generated += 1

                            if self.num_generated % self.tree_save_frequency == 0:
                                self.save_search_tree()

                            self.open_list.add(
                                (
                                    new_successor.cost,
                                    len(new_successor.collisions),
                                    new_successor,
                                ),
                                new_successor.id,
                            )
                            self.open_by_tree[new_successor.root].add(
                                (
                                    new_successor.cost,
                                    len(new_successor.collisions),
                                    new_successor,
                                ),
                                new_successor.id,
                            )
                            logging.debug("Generated node {}".format(new_successor.id))

                            if (
                                new_successor.root == 0
                                and logging.getLogger().getEffectiveLevel()
                                == logging.DEBUG
                            ):
                                self.search_tree.add_node(
                                    new_successor.id,
                                    label="{}->{}".format(
                                        new_successor.id, new_successor.cost
                                    ),
                                    cost=new_successor.cost,
                                    cost_vector=str(new_successor.cost_vector),
                                    paths=str(new_successor.paths),
                                    collisions=len(new_successor.collisions),
                                )
                                changes = "-C = ({}, {}, {}) satisfied".format(
                                    constraint["agent_id"],
                                    constraint["location"],
                                    constraint["timestep"],
                                )
                                changes += "\n|C| = {}".format(
                                    len(new_successor.collisions)
                                )
                                self.search_tree.add_edge(
                                    current_node.id,
                                    new_successor.id,
                                    label=changes,
                                )

                logging.debug(
                    "Generated {} nodes".format(self.num_generated - num_gen_before)
                )

        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            self.save_search_tree()
        all_paths = {}
        all_cost_vectors = {}
        for goal_node_id in self.nondominant_goal_nodes:
            high_level_node = self.nodes[goal_node_id]
            if high_level_node.time is not None and high_level_node.time <= self.max_time:
                all_paths[goal_node_id] = high_level_node.paths
                all_cost_vectors[goal_node_id] = high_level_node.cost_vector

        if len(self.nondominant_goal_nodes) == 0:
            return all_paths, all_cost_vectors, False
        return all_paths, all_cost_vectors, True
