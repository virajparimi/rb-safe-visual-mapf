import time
import logging
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from pud.mapf.risk_bounded_cbs import RiskBoundedCBSSolver

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    graph = nx.DiGraph()
    for n in range(9):
        graph.add_node(n)
        graph.add_edge(n, n, weight=0, step=1, cost=0)
    for m in range(4):
        graph.add_edge(m, m + 1, weight=1, step=1, cost=0)
        graph.add_edge(m + 1, m, weight=1, step=1, cost=0)
    for m in range(5, 8):
        graph.add_edge(m, m + 1, weight=1, step=1, cost=0)
        graph.add_edge(m + 1, m, weight=1, step=1, cost=0)

    graph.add_edge(6, 2, weight=1, step=1, cost=0)
    graph.add_edge(2, 6, weight=1, step=1, cost=0)
    graph.add_edge(7, 2, weight=1, step=1, cost=0)
    graph.add_edge(2, 7, weight=1, step=1, cost=0)

    graph.add_edge(6, 7, weight=1, step=1, cost=10)
    graph.add_edge(7, 6, weight=1, step=1, cost=10)

    graph_waypoints = [[0, 0], [1, 1], [2, 2], [3, 2], [4, 2], [3, 1], [2, 1], [1, 2], [0, 3]]
    graph_waypoints = np.array(graph_waypoints)

    start_ids, goal_ids = [0, 5], [4, 8]

    print("Graph nodes:", graph.nodes())
    print("Graph edges:", graph.edges(data=True))
    print("Graph waypoints:", graph_waypoints)
    print("Start IDs:", start_ids)
    print("Goal IDs:", goal_ids)
    print("Start coordinates:", graph_waypoints[start_ids])
    print("Goal coordinates:", graph_waypoints[goal_ids])

    config = {
        "seed": 0,
        "max_time": 60,
        "max_distance": 1,
        "use_experience": True,
        "collision_radius": 0.0,
        "use_cardinality": True,
        "risk_attribute": "cost",
        "budget_allocater": "uniform",
        "risk_bound": 0.2,
        "tree_save_frequency": 1,
        "split_strategy": "disjoint",
        "edge_attributes": ["step", "cost"],
        "logdir": "pud/mapf/unit_tests/logs/cbs",
    }

    start = time.time()
    solver = RiskBoundedCBSSolver(
        graph=graph,
        goals=goal_ids,
        starts=start_ids,
        graph_waypoints=graph_waypoints,
        config=config,
    )
    solution = solver.find_paths()
    print("Time taken: {}".format(time.time() - start))
    paths = solution.paths  # type: ignore
    print(paths)

    for idx, path in enumerate(paths):
        print("Cost of path for agent {}: {}".format(idx, solver.compute_cost(path, risk=True)))

    print("Number of expanded nodes: {}".format(solver.num_expanded))
    print("Number of generated nodes: {}".format(solver.num_generated))

    plt.figure(figsize=(10, 6))
    for idx, path in enumerate(paths):
        x = [graph_waypoints[node][0] for node in path]
        y = [graph_waypoints[node][1] for node in path]
        plt.plot(x, y, marker='o', label=f'Agent {idx}')
    plt.title('Paths for Agents')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid()
    plt.show()
