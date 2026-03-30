from __future__ import annotations
import copy
import time
import heapq
import numpy as np
import networkx as nx
from numpy.typing import NDArray
from networkx import Graph, shortest_path
from typing import Dict, List, Tuple, Union
from pud.mapf.mapf_exceptions import MAPFErrorCodes
from pud.mapf.utils import PrioritySet, dominate_or_equal


def compute_cost(path: List[int], graph: Graph, edge_attribute: str = ""):
    """
    Compute the cost of the path
    """
    cost = 0
    for i in range(len(path) - 1):
        cost += (
            float(graph[path[i]][path[i + 1]][edge_attribute])
            if len(edge_attribute) > 0
            else 1
        )
    return cost


def compute_sum_of_costs(
    paths: List[List[int]], graph: Graph, edge_attribute: str = ""
) -> float:
    """
    Compute the sum of costs of the paths
    """
    sum_of_costs = 0
    for path in paths:
        sum_of_costs += compute_cost(path, graph, edge_attribute)
    return sum_of_costs


def compute_heuristics(graph: Graph, goal: int, edge_attribute: str = "step"):
    """
    Compute the heuristic for each node in the graph
    """
    return nx.single_source_dijkstra_path_length(graph, goal, weight=edge_attribute)


def build_constraint_table(
    constraints: List[Dict], agent_id: int
) -> Dict[int, List[Dict]]:

    constraint_table = {}
    for constraint in constraints:
        if constraint["agent_id"] == agent_id:
            timestep = constraint["timestep"]
            if timestep not in constraint_table:
                constraint_table[timestep] = [constraint]
            else:
                constraint_table[timestep].append(constraint)
    return constraint_table


def build_constraint_table_with_ds(
    constraints: List[Dict], agent_id: int
) -> Dict[int, List[Dict]]:

    constraint_table = {}
    if not constraints:
        return constraint_table

    for constraint in constraints:
        timestep = constraint["timestep"]
        timestep_constraint = []
        if timestep in constraint_table:
            timestep_constraint = constraint_table[timestep]

        if constraint["positive"] and constraint["agent_id"] == agent_id:
            timestep_constraint.append(constraint)
            constraint_table[timestep] = timestep_constraint
        elif not constraint["positive"] and constraint["agent_id"] == agent_id:
            timestep_constraint.append(constraint)
            constraint_table[timestep] = timestep_constraint
        elif constraint["positive"]:
            negative_constraint = constraint.copy()
            negative_constraint["agent_id"] = agent_id
            if len(negative_constraint["location"]) == 2:
                negative_constraint["location"] = negative_constraint["location"][::-1]
            negative_constraint["positive"] = False
            timestep_constraint.append(negative_constraint)
            constraint_table[timestep] = timestep_constraint

    return constraint_table


def is_constrained(
    current_location: int | str,
    next_location: int | str,
    timestep: int,
    constraint_table: Dict[int, List[Dict]],
    goal: bool = False,
):

    if not goal:
        if timestep in constraint_table:
            for constraint in constraint_table[timestep]:
                if [next_location] == constraint["location"] or [
                    current_location,
                    next_location,
                ] == constraint["location"]:
                    return True
        else:
            flattened_constraints = []
            constraints = [
                constraint
                for timestep_idx, constraint in constraint_table.items()
                if timestep_idx < timestep
            ]
            for constraint in constraints:
                for c in constraint:
                    flattened_constraints.append(c)
            for constraint in flattened_constraints:
                if [next_location] == constraint["location"] and constraint["final"]:
                    return True
    else:
        flattened_constraints = []
        constraints = [
            constraint
            for timestep_idx, constraint in constraint_table.items()
            if timestep_idx > timestep
        ]
        for constraint in constraints:
            for c in constraint:
                flattened_constraints.append(c)
        for constraint in flattened_constraints:
            if [next_location] == constraint["location"]:
                return True

    return False


def is_constrained_with_ds(
    current_location: int | str,
    next_location: int | str,
    timestep: int,
    max_timestep: int,
    constraint_table: Dict[int, List[Dict]],
    constraint_agent: int,
    goal: bool = False,
):
    if not goal:
        if timestep not in constraint_table:
            return False

        for constraint in constraint_table[timestep]:
            if constraint_agent == constraint["agent_id"]:
                if len(constraint["location"]) == 1:
                    if (
                        constraint["positive"]
                        and next_location != constraint["location"][0]
                    ):
                        return True
                    elif (
                        not constraint["positive"]
                        and next_location == constraint["location"][0]
                    ):
                        return True
                else:
                    if constraint["positive"] and constraint["location"] != [
                        current_location,
                        next_location,
                    ]:
                        return True
                    if not constraint["positive"] and constraint["location"] == [
                        current_location,
                        next_location,
                    ]:
                        return True
    else:
        for t in range(timestep + 1, max_timestep + 1):
            if t not in constraint_table:
                continue
            for constraint in constraint_table[t]:

                if constraint_agent == constraint["agent_id"]:
                    if len(constraint["location"]) == 1:
                        if (
                            constraint["positive"]
                            and current_location != constraint["location"][0]
                        ):
                            return True
                        elif (
                            not constraint["positive"]
                            and current_location == constraint["location"][0]
                        ):
                            return True
    return False


def extract_path(goal_node: Node) -> List[int]:
    path = []
    current_node = goal_node
    while current_node is not None:
        path.append(current_node.location)
        current_node = current_node.parent
    path.reverse()
    return path


class Node:
    def __init__(self, id, location, g_value, h_value, parent, timestep):
        self.id = id
        self.parent = parent
        self.g_value = g_value
        self.h_value = h_value
        self.location = location
        self.timestep = timestep

    def __lt__(self, other):
        if self.g_value + self.h_value == other.g_value + other.h_value:
            if self.h_value == other.h_value:
                if self.timestep == other.timestep:
                    if self.location == other.location:
                        return self.id < other.id
                    return self.location < other.location
                return self.timestep < other.timestep
            return self.h_value < other.h_value
        return self.g_value + self.h_value < other.g_value + other.h_value


class RiskNode(Node):
    def __init__(self, id, location, g_value, h_value, parent, timestep, risk):
        self.risk = risk
        super().__init__(id, location, g_value, h_value, parent, timestep)

    def __lt__(self, other):
        if self.g_value + self.h_value == other.g_value + other.h_value:
            if self.risk == other.risk:
                if self.h_value == other.h_value:
                    if self.timestep == other.timestep:
                        if self.location == other.location:
                            return self.id < other.id
                        return self.location < other.location
                    return self.timestep < other.timestep
                return self.h_value < other.h_value
            return self.risk < other.risk
        return self.g_value + self.h_value < other.g_value + other.h_value


class AStar(object):
    def __init__(
        self,
        graph: Graph,
        agent_id: int,
        start: int,
        goal: int,
        config: Dict,
    ):
        self.goal = goal
        self.start = start
        self.graph = graph
        self.agent_id = agent_id

        self.num_expanded = 0
        self.num_generated = 0

        self.splitter = config["split_strategy"]
        self.max_distance = config["max_distance"]
        self.use_experience = config["use_experience"]
        self.edge_attributes = config["edge_attributes"]

        self.open_list = []
        self.closed_list = {}

        self.heuristic = {}
        for edge_attribute in self.edge_attributes:
            self.heuristic[edge_attribute] = self.compute_heuristics(edge_attribute)

    def compute_heuristics(self, edge_attribute="step"):
        return nx.single_source_dijkstra_path_length(
            self.graph, self.goal, weight=edge_attribute
        )

    def reset(self) -> None:
        self.num_expanded = 0
        self.num_generated = 0
        self.open_list = []
        self.closed_list = {}

    def push_node(self, node: Node) -> None:
        heapq.heappush(
            self.open_list,
            (
                node.g_value + node.h_value,
                node.h_value,
                node,
            ),
        )
        self.num_generated += 1

    def successor_generator(
        self, current_node: Node, neighbor_location: int, edge_attribute: str = "step"
    ) -> Node:
        # if neighbor_location == current_node.location:
        #     # We are waiting here
        #     successor = Node(
        #         parent=current_node,
        #         id=self.num_generated,
        #         location=neighbor_location,
        #         h_value=current_node.h_value,
        #         g_value=current_node.g_value + 1,
        #         timestep=current_node.timestep + 1,
        #     )
        # else:
        successor_gadd = self.graph[current_node.location][neighbor_location][
            edge_attribute
        ]

        # if edge_attribute == "cost":
        #     # # This ensures that the agent gives more priority to the cost of the edge
        #     # successor_gadd *= 3 * self.max_distance
        #     # This ensures that the agent makes progress even when the edge attribute is zero
        #     successor_gadd += 1

        successor = Node(
            parent=current_node,
            id=self.num_generated,
            location=neighbor_location,
            g_value=current_node.g_value + successor_gadd,
            timestep=current_node.timestep + 1,
            h_value=self.heuristic[edge_attribute][neighbor_location],
        )

        return successor

    def add_child(
        self,
        current_node: Node,
        neighbor_location: int,
        constraint_table: Dict[int, List[Dict]],
        max_constraint: int,
        edge_attribute: str = "step",
    ) -> Union[Node, None]:
        successor = self.successor_generator(
            current_node, neighbor_location, edge_attribute
        )

        if is_constrained_with_ds(
            current_node.location,
            successor.location,
            successor.timestep,
            max_constraint,
            constraint_table,
            self.agent_id,
        ):
            return

        if (successor.location, successor.timestep) in self.closed_list:
            existing_node = self.closed_list[(successor.location, successor.timestep)]
            if (
                successor.g_value + successor.h_value
                <= existing_node.g_value + existing_node.h_value
                and successor.g_value < existing_node.g_value
            ):
                self.closed_list[(successor.location, successor.timestep)] = successor
                self.push_node(successor)
        else:
            self.closed_list[(successor.location, successor.timestep)] = successor
            self.push_node(successor)

        return successor

    def find_path(
        self,
        constraints: List[Dict],
        experience: Union[List[int], None] = None,
        max_time: int = 300,
        edge_attribute: str = "step",
    ) -> Union[List[int], MAPFErrorCodes]:
        start_time = time.time()
        self.reset()

        if len(constraints) == 0:
            return shortest_path(self.graph, self.start, self.goal, edge_attribute)  # type: ignore

        if self.start not in self.heuristic[edge_attribute]:
            return MAPFErrorCodes.START_GOAL_DISCONNECT

        if self.splitter == "standard":
            constraint_table = build_constraint_table(constraints, self.agent_id)
        elif self.splitter == "disjoint":
            constraint_table = build_constraint_table_with_ds(
                constraints, self.agent_id
            )

        max_constraint = 0
        if constraint_table.keys():
            max_constraint = max(constraint_table.keys())

        root = Node(
            g_value=0,
            timestep=0,
            parent=None,
            location=self.start,
            id=self.num_generated,
            h_value=self.heuristic[edge_attribute][self.start],
        )

        if root.location == self.goal:
            if root.timestep <= max_constraint:
                if not is_constrained_with_ds(
                    self.goal,
                    self.goal,
                    root.timestep,
                    max_constraint,
                    constraint_table,
                    self.agent_id,
                    goal=True,
                ):
                    max_constraint = 0

        self.push_node(root)

        if (
            self.use_experience
            and experience is not None
            and root.location in experience
        ):
            self.push_partial_experience(
                state=root,
                experience=experience,
                max_constraint=max_constraint,
                constraint_table=constraint_table,
                edge_attribute=edge_attribute,
            )
        self.closed_list[(root.location, root.timestep)] = root

        while len(self.open_list) != 0 and time.time() - start_time < max_time:

            current_node = heapq.heappop(self.open_list)[-1]
            self.num_expanded += 1

            if current_node.location == self.goal and not is_constrained_with_ds(
                self.goal,
                self.goal,
                current_node.timestep,
                max_constraint,
                constraint_table,
                self.agent_id,
                goal=True,
            ):
                return extract_path(current_node)

            if (
                self.use_experience
                and experience is not None
                and current_node.location in experience
            ):
                self.push_partial_experience(
                    state=current_node,
                    experience=experience,
                    max_constraint=max_constraint,
                    constraint_table=constraint_table,
                    edge_attribute=edge_attribute,
                )

            for neighbor in self.graph.neighbors(current_node.location):
                _ = self.add_child(
                    current_node,
                    neighbor,
                    constraint_table,
                    max_constraint,
                    edge_attribute,
                )

        if time.time() - start_time > max_time:
            return MAPFErrorCodes.TIMELIMIT_REACHED
        else:
            return MAPFErrorCodes.NO_PATH

    def push_partial_experience(
        self,
        state: Node,
        experience: List[int],
        max_constraint: int,
        constraint_table: Dict[int, List[Dict]],
        edge_attribute: str = "step",
    ) -> None:
        for idx, location in enumerate(experience):
            if location == state.location:
                break
        if idx == len(experience) - 1:
            return

        prev_successor = state
        experience_suffix = experience[idx + 1:]  # noqa

        for successor_location in experience_suffix:
            successor = self.add_child(
                prev_successor,
                successor_location,
                constraint_table,
                max_constraint,
                edge_attribute,
            )
            if successor is None:
                break
            prev_successor = successor


class RiskBudgetedAStar(AStar):
    def __init__(
        self,
        graph: Graph,
        agent_id: int,
        start: int,
        goal: int,
        config: Dict,
    ):
        super().__init__(graph, agent_id, start, goal, config)

    def push_constrained_node(self, node: RiskNode) -> None:
        heapq.heappush(
            self.open_list,
            (
                node.g_value + node.h_value,
                node.h_value,
                node.risk + self.heuristic["cost"][node.location],
                self.heuristic["cost"][node.location],
                node,
            ),
        )
        self.num_generated += 1

    def constrained_successor_generator(
        self,
        current_node: RiskNode,
        neighbor_location: int,
        edge_attribute: str = "step",
    ) -> RiskNode:
        # if neighbor_location == current_node.location:
        #     successor_risk = current_node.risk
        #     successor = RiskNode(
        #         parent=current_node,
        #         risk=successor_risk,
        #         id=self.num_generated,
        #         location=neighbor_location,
        #         h_value=current_node.h_value,
        #         g_value=current_node.g_value + 1,
        #         timestep=current_node.timestep + 1,
        #     )
        # else:
        successor_gadd = self.graph[current_node.location][neighbor_location][
            edge_attribute
        ]
        successor_risk = (
            current_node.risk
            + self.graph[current_node.location][neighbor_location]["cost"]
        )
        successor = RiskNode(
            parent=current_node,
            risk=successor_risk,
            id=self.num_generated,
            location=neighbor_location,
            timestep=current_node.timestep + 1,
            g_value=current_node.g_value + successor_gadd,
            h_value=self.heuristic[edge_attribute][neighbor_location],
        )

        return successor

    def add_constrained_child(
        self,
        current_node: RiskNode,
        neighbor_location: int,
        risk_budget: float,
        constraint_table: Dict[int, List[Dict]],
        max_constraint: int,
        edge_attribute: str = "step",
    ) -> Union[RiskNode, None]:
        successor = self.constrained_successor_generator(
            current_node, neighbor_location, edge_attribute
        )

        potential_risk = successor.risk + self.heuristic["cost"][successor.location]
        if (
            is_constrained_with_ds(
                current_node.location,
                successor.location,
                successor.timestep,
                max_constraint,
                constraint_table,
                self.agent_id,
            )
            or round(potential_risk, 10) > round(risk_budget, 10)
        ):
            return

        if (successor.location, successor.timestep) in self.closed_list:
            existing_node = self.closed_list[(successor.location, successor.timestep)]
            if (
                successor.g_value + successor.h_value
                <= existing_node.g_value + existing_node.h_value
                and successor.g_value <= existing_node.g_value
                and successor.risk < existing_node.risk
            ):
                self.closed_list[(successor.location, successor.timestep)] = successor
                self.push_constrained_node(successor)
        else:
            self.closed_list[(successor.location, successor.timestep)] = successor
            self.push_constrained_node(successor)

        return successor

    def find_constrained_path(
        self,
        constraints: List[Dict],
        risk_budget: float,
        experience: Union[List[int], None] = None,
        max_time: int = 300,
        edge_attribute: str = "step",
    ) -> Union[List[int], MAPFErrorCodes]:
        start_time = time.time()
        self.reset()

        if self.start not in self.heuristic[edge_attribute]:
            return MAPFErrorCodes.START_GOAL_DISCONNECT

        max_constraint = 0
        if self.splitter == "standard":
            constraint_table = build_constraint_table(constraints, self.agent_id)
        elif self.splitter == "disjoint":
            constraint_table = build_constraint_table_with_ds(
                constraints, self.agent_id
            )
        if constraint_table.keys():
            max_constraint = max(constraint_table.keys())

        root = RiskNode(
            risk=0,
            g_value=0,
            timestep=0,
            parent=None,
            location=self.start,
            id=self.num_generated,
            h_value=self.heuristic[edge_attribute][self.start],
        )

        if root.location == self.goal:
            if root.timestep <= max_constraint:
                if not is_constrained_with_ds(
                    self.goal,
                    self.goal,
                    root.timestep,
                    max_constraint,
                    constraint_table,
                    self.agent_id,
                    goal=True,
                ):
                    max_constraint = 0

        self.push_constrained_node(root)

        if (
            self.use_experience
            and experience is not None
            and root.location in experience
        ):
            self.push_constrained_partial_experience(
                state=root,
                experience=experience,
                risk_budget=risk_budget,
                constraint_table=constraint_table,
                max_constraint=max_constraint,
                edge_attribute=edge_attribute,
            )
        self.closed_list[(root.location, root.timestep)] = root

        while len(self.open_list) != 0 and time.time() - start_time < max_time:

            current_node = heapq.heappop(self.open_list)[-1]
            self.num_expanded += 1

            # If we have reached the goal and the goal is not constrained and the risk is within the budget
            if (
                current_node.location == self.goal
                and not is_constrained_with_ds(
                    self.goal,
                    self.goal,
                    current_node.timestep,
                    max_constraint,
                    constraint_table,
                    self.agent_id,
                    goal=True,
                )
                and round(current_node.risk, 10) <= round(risk_budget, 10)
            ):
                return extract_path(current_node)

            if (
                self.use_experience
                and experience is not None
                and current_node.location in experience
            ):
                self.push_constrained_partial_experience(
                    state=current_node,
                    experience=experience,
                    risk_budget=risk_budget,
                    constraint_table=constraint_table,
                    max_constraint=max_constraint,
                    edge_attribute=edge_attribute,
                )

            for neighbor in self.graph.neighbors(current_node.location):
                self.add_constrained_child(
                    current_node,
                    neighbor,
                    risk_budget,
                    constraint_table,
                    max_constraint,
                    edge_attribute,
                )

        if time.time() - start_time > max_time:
            return MAPFErrorCodes.TIMELIMIT_REACHED
        else:
            return MAPFErrorCodes.NO_PATH

    def push_constrained_partial_experience(
        self,
        state: RiskNode,
        risk_budget: float,
        experience: List[int],
        max_constraint: int,
        constraint_table: Dict[int, List[Dict]],
        edge_attribute: str = "step",
    ) -> None:
        for idx, location in enumerate(experience):
            if location == state.location:
                break

        if idx == len(experience) - 1:
            return

        prev_successor = state
        experience_suffix = experience[idx + 1:]  # noqa

        for successor_location in experience_suffix:
            successor = self.add_constrained_child(
                current_node=prev_successor,
                neighbor_location=successor_location,
                risk_budget=risk_budget,
                constraint_table=constraint_table,
                max_constraint=max_constraint,
                edge_attribute=edge_attribute,
            )
            if successor is None:
                break
            prev_successor = successor


class LagrangianAStar(AStar):
    def __init__(
        self,
        graph: Graph,
        agent_id: int,
        start: int,
        goal: int,
        lagrangian: float,
        config: Dict,
    ):
        self.lagrangian = lagrangian
        super().__init__(graph, agent_id, start, goal, config)
        assert (
            len(self.edge_attributes) == 2
        ), "Lagrangian A* requires two edge attributes"
        assert (
            "step" in self.edge_attributes
        ), "Lagrangian A* requires step edge attribute"
        assert (
            "cost" in self.edge_attributes
        ), "Lagrangian A* requires cost edge attribute"

    def custom_edge_weight(self, u, v, data):
        step = data.get("step", 0)
        cost = data.get("cost", 0)
        return step + self.lagrangian * cost

    def compute_heuristics(self, edge_attribute="step"):
        return nx.single_source_dijkstra_path_length(
            self.graph, self.goal, weight=self.custom_edge_weight  # type: ignore
        )

    def successor_generator(
        self, current_node: Node, neighbor_location: int, edge_attribute: str = "step"
    ) -> Node:
        successor_gadd = (
            self.graph[current_node.location][neighbor_location][edge_attribute]
            + self.lagrangian
            * self.graph[current_node.location][neighbor_location]["cost"]
        )
        successor_hvalue = (
            self.heuristic[edge_attribute][neighbor_location]
            if neighbor_location != current_node.location
            else current_node.h_value
        )
        successor = Node(
            parent=current_node,
            id=self.num_generated,
            location=neighbor_location,
            h_value=successor_hvalue,
            g_value=current_node.g_value + successor_gadd,
            timestep=current_node.timestep + 1,
        )
        return successor


class FrontierBiObjective(object):
    def __init__(self):
        self.node_ids = set()
        self.g2min = np.inf

    def check(self, cost_vector: NDArray) -> bool:
        if self.g2min <= cost_vector[-1]:
            return True
        return False

    def update(self, node: MultiObjectiveNode) -> None:
        self.g2min = node.g_vector[-1]
        self.node_ids.add(node.id)


class BiObjectiveAStar(object):
    def __init__(
        self,
        graph: Graph,
        agent_id: int,
        start: int,
        goal: int,
        config: Dict,
    ):
        self.goal = goal
        self.start = start
        self.agent_id = agent_id

        self.graph = graph

        self.num_expanded = 0
        self.num_generated = 0

        self.splitter = config["split_strategy"]
        self.cost_dim = len(config["edge_attributes"])
        self.edge_attributes = config["edge_attributes"]

        self.open_list = PrioritySet()
        self.closed_list = {}

        self.solutions = FrontierBiObjective()
        self.id_to_frontier_map = {}

        self.heuristic = {}
        for edge_attribute in self.edge_attributes:
            self.heuristic[edge_attribute] = self.compute_heuristics(edge_attribute)

    def compute_heuristics(self, edge_attribute: str = "step"):
        return nx.single_source_dijkstra_path_length(
            self.graph, self.goal, weight=edge_attribute
        )

    def get_heuristic(self, location: int) -> NDArray:
        heuristic = np.zeros(self.cost_dim)
        for idx, edge_attribute in enumerate(self.edge_attributes):
            heuristic[idx] = self.heuristic[edge_attribute][location]
        return heuristic

    def get_frontier_key(self, state: MultiObjectiveNode) -> Tuple[int, int]:
        return state.location, state.timestep

    def reset(self):
        self.num_expanded = 0
        self.num_generated = 0
        self.open_list = PrioritySet()
        self.closed_list = {}

        self.solutions = FrontierBiObjective()
        self.id_to_frontier_map = {}

        self.start_state = MultiObjectiveNode(
            id=self.num_generated,
            location=self.start,
            g_vector=np.zeros(self.cost_dim),
            h_vector=self.get_heuristic(self.start),
            parent=None,
            timestep=0,
        )
        self.num_generated += 1

        self.goal_state = MultiObjectiveNode(
            id=self.num_generated,
            location=self.goal,
            g_vector=np.ones(self.cost_dim) * np.inf,
            h_vector=self.get_heuristic(self.goal),
            parent=None,
            timestep=0,
        )
        self.num_generated += 1

    def frontier_check(self, current_node: MultiObjectiveNode) -> bool:
        current_node_key = self.get_frontier_key(current_node)
        if current_node_key in self.id_to_frontier_map:
            return self.id_to_frontier_map[current_node_key].check(
                current_node.g_vector
            )
        return False

    def solution_check(self, current_node: MultiObjectiveNode) -> bool:
        if len(self.solutions.node_ids) == 0:
            return False
        f_value = current_node.g_vector + current_node.h_vector
        return self.solutions.check(f_value)

    def update_frontier(self, node: MultiObjectiveNode) -> None:
        node_key = self.get_frontier_key(node)
        if node_key not in self.id_to_frontier_map:
            self.id_to_frontier_map[node_key] = FrontierBiObjective()
        self.id_to_frontier_map[node_key].update(node)

    def find_path(
        self, constraints: List[Dict], max_time: int = 300
    ) -> Tuple[Dict, Dict]:
        start_time = time.time()
        self.reset()

        if self.splitter == "standard":
            constraint_table = build_constraint_table(constraints, self.agent_id)
        elif self.splitter == "disjoint":
            constraint_table = build_constraint_table_with_ds(
                constraints, self.agent_id
            )

        max_constraint = 0
        if constraint_table.keys():
            max_constraint = max(constraint_table.keys())

        f_value = self.start_state.g_vector + self.start_state.h_vector
        priority_key = tuple(f_value), self.start_state
        self.open_list.add(priority_key, self.start_state.id)
        self.closed_list[self.start_state.id] = self.start_state

        while self.open_list.size() != 0 and time.time() - start_time < max_time:
            priority_key, current_node_id = self.open_list.pop()

            current_node = priority_key[-1]
            self.num_expanded += 1

            if self.frontier_check(current_node) or self.solution_check(current_node):
                continue

            if current_node.location == self.goal and not is_constrained_with_ds(
                self.goal,
                self.goal,
                current_node.timestep,
                max_constraint,
                constraint_table,
                self.agent_id,
                goal=True,
            ):
                self.solutions.update(current_node)

            self.update_frontier(current_node)

            for neighbor in self.graph.neighbors(current_node.location):
                successor_gadd_vector = []
                for edge_attribute in self.edge_attributes:
                    successor_gadd_vector.append(
                        self.graph[current_node.location][neighbor][edge_attribute]
                    )
                if current_node.location == neighbor:
                    successor_gadd_vector[self.edge_attributes.index("step")] += 1

                successor = MultiObjectiveNode(
                    id=self.num_generated,
                    location=neighbor,
                    g_vector=current_node.g_vector + np.array(successor_gadd_vector),
                    h_vector=self.get_heuristic(neighbor),
                    parent=current_node,
                    timestep=current_node.timestep + 1,
                )

                if is_constrained_with_ds(
                    current_node.location,
                    successor.location,
                    successor.timestep,
                    max_constraint,
                    constraint_table,
                    self.agent_id,
                ):
                    continue

                if self.frontier_check(successor):
                    continue

                successor_priority_key = (
                    tuple(successor.g_vector + successor.h_vector),
                    successor,
                )
                self.open_list.add(successor_priority_key, successor.id)
                self.closed_list[successor.id] = successor
                self.num_generated += 1

        paths = {}
        cost_vectors = {}
        for node_id in self.solutions.node_ids:
            path = []
            current_node = self.closed_list[node_id]
            cost_vectors[node_id] = current_node.g_vector
            while current_node is not None:
                path.append(current_node.location)
                current_node = current_node.parent
            path.reverse()
            paths[node_id] = path

        return paths, cost_vectors


class MultiObjectiveNode(object):
    def __init__(
        self,
        id: int,
        location: int,
        g_vector: NDArray,
        h_vector: NDArray,
        parent: MultiObjectiveNode | None,
        timestep: int,
    ):
        self.id = id
        self.g_vector = g_vector
        self.h_vector = h_vector
        self.location = location
        self.parent = parent
        self.timestep = timestep

    def __lt__(self, other: MultiObjectiveNode) -> bool:
        if np.all(self.g_vector + self.h_vector == other.g_vector + other.h_vector):
            if np.all(self.h_vector == other.h_vector):
                if self.timestep == other.timestep:
                    if self.location == other.location:
                        return self.id < other.id
                    return self.location < other.location
                return self.timestep < other.timestep
            return bool(np.all(self.h_vector < other.h_vector))
        return bool(
            np.all(self.g_vector + self.h_vector < other.g_vector + other.h_vector)
        )


class MultiObjectiveAStar(object):
    def __init__(
        self,
        graph: Graph,
        agent_id: int,
        start: int,
        goal: int,
        config: Dict,
    ):
        self.goal = goal
        self.start = start
        self.agent_id = agent_id

        self.graph = graph

        self.num_expanded = 0
        self.num_generated = 0

        self.splitter = config["split_strategy"]
        self.cost_dim = len(config["edge_attributes"])
        self.edge_attributes = config["edge_attributes"]

        self.open_list = PrioritySet()
        self.closed_list = {}
        self.frontier_map = {}

        self.goal_states = set()

        self.heuristic = {}
        for edge_attribute in self.edge_attributes:
            self.heuristic[edge_attribute] = self.compute_heuristics(edge_attribute)

    def compute_heuristics(self, edge_attribute: str = "step"):
        return nx.single_source_dijkstra_path_length(
            self.graph, self.goal, weight=edge_attribute
        )

    def get_heuristic(self, location: int) -> NDArray:
        h_val = np.zeros(self.cost_dim)
        for idx, edge_attribute in enumerate(self.edge_attributes):
            h_val[idx] = self.heuristic[edge_attribute][location]
        return h_val

    def get_frontier_key(self, state: MultiObjectiveNode) -> Tuple[int, int]:
        return (state.location, state.timestep)

    def reset(self):
        self.num_expanded = 0
        self.num_generated = 0
        self.open_list = PrioritySet()
        self.closed_list = {}
        self.frontier_map = {}

        self.goal_states = set()

        self.start_state = MultiObjectiveNode(
            id=self.num_generated,
            location=self.start,
            g_vector=np.zeros(self.cost_dim),
            h_vector=self.get_heuristic(self.start),
            parent=None,
            timestep=0,
        )
        self.num_generated += 1
        self.goal_state = MultiObjectiveNode(
            id=1,
            location=self.goal,
            g_vector=np.ones(self.cost_dim) * np.inf,
            h_vector=self.get_heuristic(self.goal),
            parent=None,
            timestep=0,
        )
        self.num_generated += 1

    def filter_state(self, state: MultiObjectiveNode) -> bool:
        if self.filter_frontier_state(state):
            return True
        if self.filter_goal_state(state):
            return True
        return False

    def filter_frontier_state(self, state: MultiObjectiveNode) -> bool:
        state_key = self.get_frontier_key(state)
        if state_key not in self.frontier_map:
            return False
        for existing_state_id in self.frontier_map[state_key]:
            if existing_state_id == state.id:
                continue
            existing_state = self.closed_list[existing_state_id]
            if dominate_or_equal(
                existing_state.g_vector + existing_state.h_vector,
                state.g_vector + state.h_vector,
            ):
                return True
            if dominate_or_equal(existing_state.g_vector, state.g_vector):
                return True
        return False

    def filter_goal_state(self, state: MultiObjectiveNode) -> bool:
        for goal_state_id in self.goal_states:
            goal_state = self.closed_list[goal_state_id]
            if dominate_or_equal(
                goal_state.g_vector + goal_state.h_vector,
                state.g_vector + state.h_vector,
            ):
                return True
            if dominate_or_equal(goal_state.g_vector, state.g_vector):
                return True
        return False

    def push_node(self, node: MultiObjectiveNode) -> None:
        f_value = np.sum(node.g_vector + node.h_vector)
        self.open_list.add((f_value, np.sum(node.h_vector), node), node.id)
        self.add_to_frontier(node)
        self.num_generated += 1

    def add_to_frontier(self, state: MultiObjectiveNode) -> None:
        self.closed_list[state.id] = state
        state_key = self.get_frontier_key(state)
        if state_key not in self.frontier_map:
            self.frontier_map[state_key] = set()
            self.frontier_map[state_key].add(state.id)
        else:
            self.refine_frontier(state)
            self.frontier_map[state_key].add(state.id)

    def refine_frontier(self, state: MultiObjectiveNode) -> None:
        state_key = self.get_frontier_key(state)
        if state_key not in self.frontier_map:
            return
        temporary_frontier = copy.deepcopy(self.frontier_map[state_key])
        for existing_state_id in temporary_frontier:
            if existing_state_id == state.id:
                continue
            existing_state = self.closed_list[existing_state_id]
            if dominate_or_equal(state.g_vector, existing_state.g_vector):
                self.frontier_map[state_key].remove(existing_state_id)
                self.open_list.remove(existing_state_id)

    def refine_reached_goals(self, state: MultiObjectiveNode) -> None:
        temporary_set = copy.deepcopy(self.goal_states)
        for goal_state_id in temporary_set:
            if goal_state_id == state.id:
                continue
            goal_state = self.closed_list[goal_state_id]
            if dominate_or_equal(state.g_vector, goal_state.g_vector):
                self.goal_states.remove(goal_state_id)

    def pruning(self, state: MultiObjectiveNode) -> bool:
        state_key = self.get_frontier_key(state)
        if state_key not in self.frontier_map:
            return False
        for existing_state_id in self.frontier_map[state_key]:
            if existing_state_id == state.id:
                continue
            existing_state = self.closed_list[existing_state_id]
            if dominate_or_equal(existing_state.g_vector, state.g_vector):
                return True
        return False

    def reconstruct_paths(self) -> Dict:
        paths = {}
        for goal_state_id in self.goal_states:
            goal_state = self.closed_list[goal_state_id]
            path = []
            current_state = goal_state
            while current_state is not None:
                path.append(current_state.location)
                current_state = current_state.parent
            path.reverse()
            paths[goal_state_id] = path
        return paths

    def find_path(
        self, constraints: List[Dict], max_time: int = 300
    ) -> Tuple[Dict, Dict]:
        start_time = time.time()
        self.reset()

        if self.splitter == "standard":
            constraint_table = build_constraint_table(constraints, self.agent_id)
        elif self.splitter == "disjoint":
            constraint_table = build_constraint_table_with_ds(
                constraints, self.agent_id
            )

        max_constraint = 0
        if constraint_table.keys():
            max_constraint = max(constraint_table.keys())
        self.push_node(self.start_state)

        while self.open_list.size() != 0 and time.time() - start_time < max_time:
            _, current_node_id = self.open_list.pop()
            current_node = self.closed_list[current_node_id]
            self.num_expanded += 1

            if self.filter_state(current_node):
                continue

            if current_node.location == self.goal and not is_constrained_with_ds(
                self.goal,
                self.goal,
                current_node.timestep,
                max_constraint,
                constraint_table,
                self.agent_id,
                goal=True,
            ):
                self.goal_states.add(current_node.id)
                self.refine_reached_goals(current_node)

            for neighbor in self.graph.neighbors(current_node.location):
                successor_gadd_vector = []
                for edge_attribute in self.edge_attributes:
                    successor_gadd_vector.append(
                        self.graph[current_node.location][neighbor][edge_attribute]
                    )
                if current_node.location == neighbor:
                    successor_gadd_vector[self.edge_attributes.index("step")] += 1
                successor = MultiObjectiveNode(
                    id=self.num_generated,
                    location=neighbor,
                    g_vector=current_node.g_vector + np.array(successor_gadd_vector),
                    h_vector=self.get_heuristic(neighbor),
                    parent=current_node,
                    timestep=current_node.timestep + 1,
                )

                if is_constrained_with_ds(
                    current_node.location,
                    successor.location,
                    successor.timestep,
                    max_constraint,
                    constraint_table,
                    self.agent_id,
                ):
                    continue

                if self.filter_state(successor):
                    continue

                if not self.pruning(successor):
                    self.push_node(successor)

        paths = self.reconstruct_paths()
        cost_vectors = {}
        for id, _ in paths.items():
            cost_vectors[id] = self.closed_list[id].g_vector
        return paths, cost_vectors


class FrontierLinear(object):
    def __init__(self):
        self.node_gs = {}  # Node ID -> g_vector

    def check(self, cost_vector: NDArray) -> bool:
        for _, node_gs in self.node_gs.items():
            if dominate_or_equal(node_gs, cost_vector):
                return True
        return False

    def dr_check(self, cost_vector: NDArray) -> bool:
        for _, node_gs in self.node_gs.items():
            if dominate_or_equal(node_gs[1:], cost_vector[1:]):
                return True
        return False

    def remove(self, cost_vector: NDArray) -> bool:
        for node_id, node_gs in self.node_gs.items():
            if np.all(node_gs == cost_vector):
                self.node_gs.pop(node_id)
                return True
        return False

    def add(self, node_id: int, cost_vector: NDArray) -> None:
        self.node_gs[node_id] = cost_vector

    def filter(self, cost_vector: NDArray) -> set:
        deleted_ids = set()
        node_gs_copy = copy.deepcopy(self.node_gs)
        for node_id, node_gs in self.node_gs.items():
            if dominate_or_equal(cost_vector, node_gs):
                node_gs_copy.pop(node_id)
            deleted_ids.add(node_id)
        self.node_gs = node_gs_copy
        return deleted_ids

    def update(self, node: MultiObjectiveNode) -> set:
        deleted_ids = self.filter(node.g_vector)
        self.add(node.id, node.g_vector)
        return deleted_ids


class NAMultiObjectiveAStar(object):
    def __init__(
        self, graph: Graph, agent_id: int, start: int, goal: int, config: Dict
    ):
        self.goal = goal
        self.start = start
        self.agent_id = agent_id

        self.graph = graph

        self.num_expanded = 0
        self.num_generated = 0

        self.splitter = config["split_strategy"]
        self.cost_dim = len(config["edge_attributes"])
        self.edge_attributes = config["edge_attributes"]

        self.open_list = PrioritySet()
        self.closed_list = {}

        self.open_gs = {}
        self.closed_gs = {}

        self.goal_states = set()

        self.heuristic = {}
        for edge_attribute in self.edge_attributes:
            self.heuristic[edge_attribute] = compute_heuristics(
                self.graph, self.goal, edge_attribute
            )

    def reset(self):
        self.num_expanded = 0
        self.num_generated = 0

        self.open_list = PrioritySet()
        self.closed_list = {}

        self.open_gs = {}
        self.closed_gs = {}

        self.goal_states = set()

        self.start_state = MultiObjectiveNode(
            id=self.num_generated,
            location=self.start,
            g_vector=np.zeros(self.cost_dim),
            h_vector=self.get_heuristic(self.start),
            parent=None,
            timestep=0,
        )
        self.num_generated += 1

        self.goal_state = MultiObjectiveNode(
            id=self.num_generated,
            location=self.goal,
            g_vector=np.ones(self.cost_dim) * np.inf,
            h_vector=self.get_heuristic(self.goal),
            parent=None,
            timestep=0,
        )
        self.num_generated += 1

    def get_heuristic(self, location: int) -> NDArray:
        h_val = np.zeros(self.cost_dim)
        for idx, edge_attribute in enumerate(self.edge_attributes):
            h_val[idx] = self.heuristic[edge_attribute][location]
        return h_val

    def get_frontier_key(self, state: MultiObjectiveNode) -> Tuple[int, int]:
        return state.location, state.timestep

    def open_gs_to_closed_gs(self, node: MultiObjectiveNode) -> None:
        node_key = self.get_frontier_key(node)
        self.open_gs[node_key].remove(node.g_vector)

        if node_key not in self.closed_gs:
            self.closed_gs[node_key] = FrontierLinear()
        self.closed_gs[node_key].add(node.id, node.g_vector)

    def filter_open_list(self, node: MultiObjectiveNode) -> None:
        temporary_set = copy.deepcopy(self.open_list.set)
        for open_node_id in self.open_list.set:
            open_node = self.closed_list[open_node_id]
            open_node_fval = open_node.g_vector + open_node.h_vector
            if dominate_or_equal(node.g_vector, open_node_fval):
                temporary_set.remove(open_node_id)
        self.open_list.set = temporary_set

    def gs_dominance_check(self, node: MultiObjectiveNode) -> bool:
        node_key = self.get_frontier_key(node)
        if node_key in self.open_gs:
            if self.open_gs[node_key].check(node.g_vector):
                return True
        if node_key in self.closed_gs:
            if self.closed_gs[node_key].dr_check(node.g_vector):
                return True
        return False

    def solution_dominance_check(self, cost_vector: NDArray) -> bool:
        for goal_state_id in self.goal_states:
            goal_state = self.closed_list[goal_state_id]
            if dominate_or_equal(goal_state.g_vector[1:], cost_vector[1:]):
                return True
        return False

    def filter_open_gs(self, node: MultiObjectiveNode) -> None:
        node_key = self.get_frontier_key(node)

        deleted_ids = set()
        if node_key in self.open_gs:
            deleted_ids = self.open_gs[node_key].update(node)
        else:
            self.open_gs[node_key] = FrontierLinear()
            self.open_gs[node_key].add(node.id, node.g_vector)

        if len(deleted_ids) > 0:
            counter = 0
            for open_node_id in self.open_list.set:
                if open_node_id in deleted_ids:
                    self.open_list.remove(open_node_id)
                    counter += 1
                if counter == len(deleted_ids):
                    break

        f_value = node.g_vector + node.h_vector
        priority_key = tuple(f_value), node
        self.open_list.add(priority_key, node.id)
        self.closed_list[node.id] = node
        self.num_generated += 1

    def filter_closed_gs(self, node: MultiObjectiveNode) -> None:
        node_key = self.get_frontier_key(node)
        if node_key in self.closed_gs:
            _ = self.closed_gs[node_key].filter(node.g_vector)

    def find_path(self, constraints: List[Dict], max_time: int = 300):
        start_time = time.time()
        self.reset()

        if self.splitter == "standard":
            constraint_table = build_constraint_table(constraints, self.agent_id)
        elif self.splitter == "disjoint":
            constraint_table = build_constraint_table_with_ds(
                constraints, self.agent_id
            )

        max_constraint = 0
        if constraint_table.keys():
            max_constraint = max(constraint_table.keys())

        f_value = self.start_state.g_vector + self.start_state.h_vector
        priority_key = tuple(f_value), self.start_state
        self.open_list.add(priority_key, self.start_state.id)
        self.closed_list[self.start_state.id] = self.start_state
        self.open_gs[self.get_frontier_key(self.start_state)] = FrontierLinear()
        self.open_gs[self.get_frontier_key(self.start_state)].add(
            self.start_state.id, self.start_state.g_vector
        )

        while self.open_list.size() != 0 and time.time() - start_time < max_time:

            priority_key, current_node_id = self.open_list.pop()
            current_node = self.closed_list[current_node_id]
            self.num_expanded += 1

            self.open_gs_to_closed_gs(current_node)

            if current_node.location == self.goal and not is_constrained_with_ds(
                self.goal,
                self.goal,
                current_node.timestep,
                max_constraint,
                constraint_table,
                self.agent_id,
                goal=True,
            ):
                self.goal_states.add(current_node.id)
                self.filter_open_list(current_node)
                continue

            for neighbor in self.graph.neighbors(current_node.location):
                successor_gadd_vector = []
                for edge_attribute in self.edge_attributes:
                    successor_gadd_vector.append(
                        self.graph[current_node.location][neighbor][edge_attribute]
                    )
                if current_node.location == neighbor:
                    successor_gadd_vector[self.edge_attributes.index("step")] += 1

                successor = MultiObjectiveNode(
                    id=self.num_generated,
                    location=neighbor,
                    g_vector=current_node.g_vector + np.array(successor_gadd_vector),
                    h_vector=self.get_heuristic(neighbor),
                    parent=current_node,
                    timestep=current_node.timestep + 1,
                )

                if is_constrained_with_ds(
                    current_node.location,
                    successor.location,
                    successor.timestep,
                    max_constraint,
                    constraint_table,
                    self.agent_id,
                ):
                    continue

                if self.gs_dominance_check(successor):
                    continue

                successor_fval = successor.g_vector + successor.h_vector
                if self.solution_dominance_check(successor_fval):
                    continue

                self.filter_open_gs(successor)
                self.filter_closed_gs(successor)

        paths = {}
        cost_vectors = {}
        for goal_state_id in self.goal_states:
            path = []
            current_node = self.closed_list[goal_state_id]
            while current_node is not None:
                path.append(current_node.location)
                current_node = current_node.parent
            path.reverse()
            paths[goal_state_id] = path
            cost_vectors[goal_state_id] = self.closed_list[goal_state_id].g_vector

        return paths, cost_vectors
