import unittest
import numpy as np
import networkx as nx
from typing import List

from pud.mapf.bocbs import BiObjectiveCBSSolver
from pud.mapf.mocbs import MultiObjectiveCBSSolver
from pud.mapf.cbs import CBSSolver, detect_collisions
from pud.mapf.lagrangian_cbs import LagrangianCBSSolver
from pud.mapf.namocbs import NAMultiOjectiveCBSSolver
from pud.mapf.risk_bounded_cbs import RiskBoundedCBSSolver

"""
python pud/mapf/unit_tests/test_cbs_solver.py TestCBSSolver.test_cbs_paths
python pud/mapf/unit_tests/test_cbs_solver.py TestCBSSolver.test_cbsds_paths
python pud/mapf/unit_tests/test_cbs_solver.py TestCBSSolver.test_mocbs_paths
python pud/mapf/unit_tests/test_cbs_solver.py TestCBSSolver.test_bocbs_paths
python pud/mapf/unit_tests/test_cbs_solver.py TestCBSSolver.test_namocbs_paths
python pud/mapf/unit_tests/test_cbs_solver.py TestCBSSolver.test_lagrangian_cbs_paths
python pud/mapf/unit_tests/test_cbs_solver.py TestCBSSolver.test_risk_bounded_cbs_paths
"""


class TestCBSSolver(unittest.TestCase):
    def setUp(self):
        self.filename = "pud/mapf/unit_tests/test_cbs_input.txt"

    def compute_cost(self, path: List[int]) -> float:
        cost = 0.0
        for i in range(len(path) - 1):
            cost += self.G[path[i]][path[i + 1]]["cost"]
        return cost

    def load_problem(self, filename):
        f = open(
            self.filename,
            "r",
        )
        line = f.readline()
        rows, columns = [int(x) for x in line.split(" ")]
        rows = int(rows)
        columns = int(columns)

        self.graph_waypoints = []
        self.G = nx.empty_graph(0, create_using=nx.DiGraph)

        boolean_map = []
        for _ in range(rows):
            line = f.readline()
            boolean_map.append([])
            for cell in line:
                if cell == "@":
                    boolean_map[-1].append(True)
                elif cell == ".":
                    boolean_map[-1].append(False)

        boolean_map = np.array(boolean_map)

        line = f.readline()
        num_agents = int(line)

        starts = []
        goals = []
        for _ in range(num_agents):
            line = f.readline()
            start_x, start_y, goal_x, goal_y = [int(x) for x in line.split(" ")]
            starts.append((start_x, start_y))
            goals.append((goal_x, goal_y))

            self.graph_waypoints.append([start_x, start_y])
            self.graph_waypoints.append([goal_x, goal_y])

        f.close()

        risky_nodes = []
        for node in range(boolean_map.shape[0] * boolean_map.shape[1]):
            node_x, node_y = node // boolean_map.shape[1], node % boolean_map.shape[1]
            if boolean_map[node_x, node_y]:
                potential_neighbors = [
                    (node_x - 1, node_y),
                    (node_x + 1, node_y),
                    (node_x, node_y - 1),
                    (node_x, node_y + 1),
                ]
                for unsafe_neighbor in potential_neighbors:
                    if (
                        unsafe_neighbor[0] >= 0
                        and unsafe_neighbor[0] < boolean_map.shape[0]
                        and unsafe_neighbor[1] >= 0
                        and unsafe_neighbor[1] < boolean_map.shape[1]
                        and not boolean_map[unsafe_neighbor[0], unsafe_neighbor[1]]
                    ):
                        risky_nodes.append(
                            unsafe_neighbor[0] * boolean_map.shape[1]
                            + unsafe_neighbor[1]
                        )

        for node in range(boolean_map.shape[0] * boolean_map.shape[1]):
            node_x, node_y = node // boolean_map.shape[1], node % boolean_map.shape[1]
            if boolean_map[node_x, node_y]:
                continue
            self.graph_waypoints.append([node_x, node_y])
            potential_neighbors = [
                (node_x - 1, node_y),
                (node_x + 1, node_y),
                (node_x, node_y - 1),
                (node_x, node_y + 1),
            ]

            for neighbor in potential_neighbors:
                if (
                    neighbor[0] >= 0
                    and neighbor[0] < boolean_map.shape[0]
                    and neighbor[1] >= 0
                    and neighbor[1] < boolean_map.shape[1]
                    and not boolean_map[neighbor[0], neighbor[1]]
                ):
                    if (
                        node in risky_nodes
                        or neighbor[0] * boolean_map.shape[1] + neighbor[1]
                        in risky_nodes
                    ):
                        self.G.add_edge(
                            node,
                            neighbor[0] * boolean_map.shape[1] + neighbor[1],
                            weight=1,
                            step=1,
                            cost=1,
                        )
                    else:
                        self.G.add_edge(
                            node,
                            neighbor[0] * boolean_map.shape[1] + neighbor[1],
                            weight=1,
                            step=1,
                            cost=0,
                        )

        self.start_ids, self.goal_ids = [], []
        for start_node in starts:
            start_node = start_node[0] * boolean_map.shape[1] + start_node[1]
            self.start_ids.append(start_node)
        for goal_node in goals:
            goal_node = goal_node[0] * boolean_map.shape[1] + goal_node[1]
            self.goal_ids.append(goal_node)

        self.graph_waypoints = np.array(self.graph_waypoints, dtype=float)

        # Pre-compute identical Manhattan-distance ensembles (the learned version would differ per slice).
        diff = self.graph_waypoints[:, None, :] - self.graph_waypoints[None, :, :]
        manhattan = np.abs(diff).sum(axis=-1)
        self.pdist = np.repeat(manhattan[None, :, :], repeats=3, axis=0)

    def test_cbs_paths(self):
        self.load_problem(self.filename)
        self.graph_waypoints = np.array(self.graph_waypoints)

        config = {
            "seed": 0,
            "max_time": 100,
            "max_distance": 1,
            "use_experience": True,
            "collision_radius": 0.0,
            "use_cardinality": True,
            "risk_attribute": "cost",
            "tree_save_frequency": 100,
            "edge_attributes": ["step"],
            "split_strategy": "standard",
            "logdir": "pud/mapf/unit_tests/logs/cbs",
        }

        solver = CBSSolver(
            graph=self.G,
            goals=self.goal_ids,
            starts=self.start_ids,
            pdist=self.pdist,
            config=config,
        )
        solution = solver.find_paths()
        paths = solution.paths
        accumulated_risk = 0
        print("<" + "-" * 50 + "CBS" + "-" * 50 + ">")
        for idx, path in enumerate(paths):
            agent_risk = self.compute_cost(path)
            accumulated_risk += agent_risk
            print("Cost of path for agent {}: {}".format(idx, agent_risk))
        print("Cost of solution: {}".format(solution.cost))
        print("Accumulated Risk: {}".format(accumulated_risk))

        self.assertTrue(len(paths) == 5)
        self.assertTrue(detect_collisions(paths, self.pdist, 0.0) == [])

        print("Number of expanded nodes: {}".format(solver.num_expanded))
        print("Number of generated nodes: {}".format(solver.num_generated))

    def test_cbsds_paths(self):
        self.load_problem(self.filename)
        self.graph_waypoints = np.array(self.graph_waypoints)

        config = {
            "seed": 0,
            "max_time": 100,
            "max_distance": 1,
            "use_experience": True,
            "collision_radius": 0.0,
            "use_cardinality": True,
            "risk_attribute": "cost",
            "tree_save_frequency": 100,
            "edge_attributes": ["step"],
            "split_strategy": "disjoint",
            "logdir": "pud/mapf/unit_tests/logs/cbsds",
        }

        solver = CBSSolver(
            graph=self.G,
            goals=self.goal_ids,
            starts=self.start_ids,
            pdist=self.pdist,
            config=config,
        )
        solution = solver.find_paths()
        paths = solution.paths
        print("<" + "-" * 50 + "CBS-DS" + "-" * 50 + ">")
        print("Solution Cost: {}".format(solution.cost))
        accumulated_risk = 0
        for idx, path in enumerate(paths):
            agent_risk = self.compute_cost(path)
            accumulated_risk += agent_risk
            print("Cost of path for agent {}: {}".format(idx, agent_risk))
        print("Cost of solution: {}".format(solution.cost))
        print("Accumulated Risk: {}".format(accumulated_risk))

        self.assertTrue(len(paths) == 5)
        self.assertTrue(detect_collisions(paths, self.pdist, 0.0) == [])

        print("Number of expanded nodes: {}".format(solver.num_expanded))
        print("Number of generated nodes: {}".format(solver.num_generated))

    def test_risk_bounded_cbs_paths(self):
        self.load_problem(self.filename)
        self.graph_waypoints = np.array(self.graph_waypoints)
        config = {
            "seed": 0,
            "max_time": 100,
            "max_distance": 1,
            "risk_bound": 4.0,
            "use_experience": True,
            "collision_radius": 0.0,
            "use_cardinality": True,
            "risk_attribute": "cost",
            "tree_save_frequency": 100,
            "split_strategy": "disjoint",
            "budget_allocater": "uniform",
            "edge_attributes": ["step", "cost"],
            "logdir": "pud/mapf/unit_tests/logs/rbcbs",
            "risk_reallocation_strategy": "price_clearing",
            # "risk_reallocation_strategy": "surplus_deficit",
        }

        solver = RiskBoundedCBSSolver(
            graph=self.G,
            goals=self.goal_ids,
            starts=self.start_ids,
            pdist=self.pdist,
            config=config,
        )
        solution = solver.find_paths()
        paths = solution.paths
        print("<" + "-" * 50 + "RB-CBS" + "-" * 50 + ">")

        accumulated_risk = 0
        for idx, path in enumerate(paths):
            agent_risk = self.compute_cost(path)
            accumulated_risk += agent_risk
            print("Cost of path for agent {}: {}".format(idx, agent_risk))
        print("Cost of solution: {}".format(solution.cost))
        print("Accumulated Risk: {}".format(accumulated_risk))

        print("Number of expanded nodes: {}".format(solver.num_expanded))
        print("Number of generated nodes: {}".format(solver.num_generated))

        self.assertTrue(len(paths) == 5)
        self.assertTrue(accumulated_risk <= config["risk_bound"])
        self.assertTrue(detect_collisions(paths, self.pdist, 0.0) == [])

    def test_lagrangian_cbs_paths(self):
        self.load_problem(self.filename)
        self.graph_waypoints = np.array(self.graph_waypoints)
        config = {
            "seed": 0,
            "max_time": 100,
            "max_distance": 1,
            "lagrangian": 1.0,
            "use_experience": True,
            "collision_radius": 0.0,
            "use_cardinality": True,
            "risk_attribute": "cost",
            "tree_save_frequency": 100,
            "split_strategy": "disjoint",
            "edge_attributes": ["step", "cost"],
            "logdir": "pud/mapf/unit_tests/logs/lcbs",
        }

        solver = LagrangianCBSSolver(
            graph=self.G,
            goals=self.goal_ids,
            starts=self.start_ids,
            pdist=self.pdist,
            config=config,
        )
        solution = solver.find_paths()
        paths = solution.paths
        print("<" + "-" * 50 + "L-CBS" + "-" * 50 + ">")

        accumulated_risk = 0
        for idx, path in enumerate(paths):
            agent_risk = self.compute_cost(path)
            accumulated_risk += agent_risk
            print("Cost of path for agent {}: {}".format(idx, agent_risk))
        print("Cost of solution: {}".format(solution.cost))
        print("Accumulated Risk: {}".format(accumulated_risk))

        print("Number of expanded nodes: {}".format(solver.num_expanded))
        print("Number of generated nodes: {}".format(solver.num_generated))

        self.assertTrue(len(paths) == 5)
        self.assertTrue(detect_collisions(paths, self.pdist, 0.0) == [])

    def test_mocbs_paths(self):
        self.load_problem(self.filename)
        self.graph_waypoints = np.array(self.graph_waypoints)
        config = {
            "seed": 0,
            "max_time": 100,
            "max_distance": 1,
            "use_experience": False,
            "collision_radius": 0.0,
            "use_cardinality": False,
            "risk_attribute": "cost",
            "tree_save_frequency": 1000,
            "split_strategy": "disjoint",
            "edge_attributes": ["step", "cost"],
            "logdir": "pud/mapf/unit_tests/logs/mocbs",
        }

        solver = MultiObjectiveCBSSolver(
            graph=self.G,
            goals=self.goal_ids,
            starts=self.start_ids,
            pdist=self.pdist,
            config=config,
        )
        all_paths, all_cost_vectors, success = solver.find_paths()
        print("<" + "-" * 50 + "MO-CBS" + "-" * 50 + ">")
        if success:
            keys = all_paths.keys()
            path_risks = np.zeros(len(all_paths))
            for idx, id in enumerate(all_paths):
                for path in all_paths[id]:
                    path_risks[idx] += self.compute_cost(path)
            best_path_idx = np.argmin(path_risks)
            best_path_key = list(keys)[best_path_idx]
            paths = all_paths[best_path_key]
            for idx, path in enumerate(paths):
                agent_risk = self.compute_cost(path)
                print("Cost of path for agent {}: {}".format(idx, agent_risk))
            print(
                "Cost of solution: {}".format(
                    all_cost_vectors[best_path_key][
                        config["edge_attributes"].index("step")
                    ]
                )
            )
            print("Accumulated risk: {}".format(np.min(path_risks)))
            # Save the all_paths and all_cost_vectors
            all_sols = []
            all_cvecs = []
            for key in all_paths:
                all_sols.append(all_paths[key])
                all_cvecs.append(all_cost_vectors[key])
            all_sols = np.asarray(all_sols, dtype=object)
            np.savez(
                "pud/mapf/unit_tests/logs/mocbs/mocbs.npz",
                all_sols=all_sols,
                all_cvecs=all_cvecs,
                allow_pickle=True,
            )
        else:
            print("No solution found")

        print("Number of expanded nodes: {}".format(solver.num_expanded))
        print("Number of generated nodes: {}".format(solver.num_generated))

        print("Min / Median / Max / Mean / Std of Low-Level RTs: {:.5f} / {:.5f} / {:.5f} / {:.5f} / {:.5f}".format(
            np.min(solver.single_agent_planner_times),
            np.median(solver.single_agent_planner_times),
            np.max(solver.single_agent_planner_times),
            np.mean(solver.single_agent_planner_times),
            np.std(solver.single_agent_planner_times)
        ))
        print(
            "Min / Median / Max / Mean / Std of Low-Level Exp: {:.2f} / {:.2f} / {:.2f} / {:.2f} / {:.2f}".format(
                np.min(solver.single_agent_planner_expansions),
                np.median(solver.single_agent_planner_expansions),
                np.max(solver.single_agent_planner_expansions),
                np.mean(solver.single_agent_planner_expansions),
                np.std(solver.single_agent_planner_expansions)
            )
        )

        self.assertTrue(len(paths) == 5)
        self.assertTrue(detect_collisions(paths, self.pdist, 0.0) == [])

    def test_bocbs_paths(self):
        self.load_problem(self.filename)
        self.graph_waypoints = np.array(self.graph_waypoints)
        config = {
            "seed": 0,
            "max_time": 100,
            "max_distance": 1,
            "use_experience": False,
            "collision_radius": 0.0,
            "use_cardinality": False,
            "risk_attribute": "cost",
            "tree_save_frequency": 1000,
            "split_strategy": "disjoint",
            "edge_attributes": ["step", "cost"],
            "logdir": "pud/mapf/unit_tests/logs/bocbs",
        }

        solver = BiObjectiveCBSSolver(
            graph=self.G,
            goals=self.goal_ids,
            starts=self.start_ids,
            pdist=self.pdist,
            config=config,
        )
        all_paths, all_cost_vectors, success = solver.find_paths()
        print("<" + "-" * 50 + "BO-CBS" + "-" * 50 + ">")
        if success:
            keys = all_paths.keys()
            path_risks = np.zeros(len(all_paths))
            for idx, id in enumerate(all_paths):
                for path in all_paths[id]:
                    path_risks[idx] += self.compute_cost(path)
            best_path_idx = np.argmin(path_risks)
            best_path_key = list(keys)[best_path_idx]
            paths = all_paths[best_path_key]
            for idx, path in enumerate(paths):
                agent_risk = self.compute_cost(path)
                print("Cost of path for agent {}: {}".format(idx, agent_risk))
            print(
                "Cost of solution: {}".format(
                    all_cost_vectors[best_path_key][
                        config["edge_attributes"].index("step")
                    ]
                )
            )
            print("Accumulated risk: {}".format(np.min(path_risks)))
            # Save the all_paths and all_cost_vectors
            all_sols = []
            all_cvecs = []
            for key in all_paths:
                all_sols.append(all_paths[key])
                all_cvecs.append(all_cost_vectors[key])
            all_sols = np.asarray(all_sols, dtype=object)
            np.savez(
                "pud/mapf/unit_tests/logs/bocbs/bocbs.npz",
                all_sols=all_sols,
                all_cvecs=all_cvecs,
                allow_pickle=True,
            )
        else:
            print("No solution found")

        print("Number of expanded nodes: {}".format(solver.num_expanded))
        print("Number of generated nodes: {}".format(solver.num_generated))

        print("Min / Median / Max / Mean / Std of Low-Level RTs: {:.5f} / {:.5f} / {:.5f} / {:.5f} / {:.5f}".format(
            np.min(solver.single_agent_planner_times),
            np.median(solver.single_agent_planner_times),
            np.max(solver.single_agent_planner_times),
            np.mean(solver.single_agent_planner_times),
            np.std(solver.single_agent_planner_times)
        ))
        print("Min / Median / Max / Mean / Std of Low-Level Exp: {:.2f} / {:.2f} / {:.2f} / {:.2f} / {:.2f}".format(
            np.min(solver.single_agent_planner_expansions),
            np.median(solver.single_agent_planner_expansions),
            np.max(solver.single_agent_planner_expansions),
            np.mean(solver.single_agent_planner_expansions),
            np.std(solver.single_agent_planner_expansions)
        ))

        self.assertTrue(len(paths) == 5)
        self.assertTrue(detect_collisions(paths, self.pdist, 0.0) == [])

    def test_namocbs_paths(self):
        self.load_problem(self.filename)
        self.graph_waypoints = np.array(self.graph_waypoints)
        config = {
            "seed": 0,
            "max_time": 100,
            "max_distance": 1,
            "use_experience": False,
            "collision_radius": 0.0,
            "use_cardinality": False,
            "risk_attribute": "cost",
            "tree_save_frequency": 1000,
            "split_strategy": "disjoint",
            "edge_attributes": ["step", "cost"],
            "logdir": "pud/mapf/unit_tests/logs/namocbs",
        }

        solver = NAMultiOjectiveCBSSolver(
            graph=self.G,
            goals=self.goal_ids,
            starts=self.start_ids,
            pdist=self.pdist,
            config=config,
        )
        all_paths, all_cost_vectors, success = solver.find_paths()
        print("<" + "-" * 50 + "NA-MO-CBS" + "-" * 50 + ">")
        if success:
            keys = all_paths.keys()
            path_risks = np.zeros(len(all_paths))
            for idx, id in enumerate(all_paths):
                for path in all_paths[id]:
                    path_risks[idx] += self.compute_cost(path)
            best_path_idx = np.argmin(path_risks)
            best_path_key = list(keys)[best_path_idx]
            paths = all_paths[best_path_key]
            for idx, path in enumerate(paths):
                agent_risk = self.compute_cost(path)
                print("Cost of path for agent {}: {}".format(idx, agent_risk))
            print(
                "Cost of solution: {}".format(
                    all_cost_vectors[best_path_key][
                        config["edge_attributes"].index("step")
                    ]
                )
            )
            print("Accumulated risk: {}".format(np.min(path_risks)))
            # Save the all_paths and all_cost_vectors
            all_sols = []
            all_cvecs = []
            for key in all_paths:
                all_sols.append(all_paths[key])
                all_cvecs.append(all_cost_vectors[key])
            all_sols = np.asarray(all_sols, dtype=object)
            np.savez(
                "pud/mapf/unit_tests/logs/namocbs/namocbs.npz",
                all_sols=all_sols,
                all_cvecs=all_cvecs,
                allow_pickle=True,
            )
        else:
            print("No solution found")

        print("Number of expanded nodes: {}".format(solver.num_expanded))
        print("Number of generated nodes: {}".format(solver.num_generated))

        print("Min / Median / Max / Mean / Std of Low-Level RTs: {:.5f} / {:.5f} / {:.5f} / {:.5f} / {:.5f}".format(
            np.min(solver.single_agent_planner_times),
            np.median(solver.single_agent_planner_times),
            np.max(solver.single_agent_planner_times),
            np.mean(solver.single_agent_planner_times),
            np.std(solver.single_agent_planner_times)
        ))
        print("Min / Median / Max / Mean / Std of Low-Level Exp: {:.2f} / {:.2f} / {:.2f} / {:.2f} / {:.2f}".format(
            np.min(solver.single_agent_planner_expansions),
            np.median(solver.single_agent_planner_expansions),
            np.max(solver.single_agent_planner_expansions),
            np.mean(solver.single_agent_planner_expansions),
            np.std(solver.single_agent_planner_expansions)
        ))


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
