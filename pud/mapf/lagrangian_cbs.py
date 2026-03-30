from networkx import Graph
from typing import Dict, List
from numpy.typing import NDArray

from pud.mapf.cbs import CBSSolver
from pud.mapf.single_agent_planner import LagrangianAStar


class LagrangianCBSSolver(CBSSolver):

    def __init__(
        self,
        graph: Graph,
        goals: List[int],
        starts: List[int],
        pdist: NDArray,
        config: Dict,
    ):

        self.lagrangian = config["lagrangian"]
        super().__init__(graph, goals, starts, pdist, config)

    def make_planners(self, config: Dict) -> None:
        self.single_agent_planners = {}
        for agent in range(self.num_agents):
            self.single_agent_planners[agent] = LagrangianAStar(
                config=config,
                agent_id=agent,
                graph=self.graph,
                goal=self.goals[agent],
                start=self.starts[agent],
                lagrangian=self.lagrangian,
            )

    def compute_cost(self, path: List[int], risk: bool = False) -> float:
        cost = 0.0
        for i in range(len(path) - 1):
            cost += (
                1 + self.lagrangian * self.graph[path[i]][path[i + 1]][self.risk_attribute]
            )
        return cost
