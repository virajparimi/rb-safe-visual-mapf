# Code adapted from https://github.com/wonderren/public_pymomapf/blob/master/libmomapf/mocbs.py
# Choosing to perform the tree-by-tree expansion strategy
from __future__ import annotations
from networkx import Graph
from typing import List, Dict
from numpy.typing import NDArray

from pud.mapf.mocbs import MultiObjectiveCBSSolver
from pud.mapf.single_agent_planner import BiObjectiveAStar


class BiObjectiveCBSSolver(MultiObjectiveCBSSolver):
    def __init__(
        self,
        graph: Graph,
        starts: List[int],
        goals: List[int],
        pdist: NDArray,
        config: Dict,
    ):
        super().__init__(graph, starts, goals, pdist, config)

    def make_planners(self, config: Dict) -> None:
        self.single_agent_planners = {}
        for agent in range(self.num_agents):
            self.single_agent_planners[agent] = BiObjectiveAStar(
                config=config,
                agent_id=agent,
                graph=self.graph,
                goal=self.goals[agent],
                start=self.starts[agent],
            )
