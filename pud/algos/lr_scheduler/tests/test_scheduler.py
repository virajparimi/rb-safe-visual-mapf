import unittest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from pud.algos.lr_scheduler.scheduler import (
    HybridScheduler,
    PiecewiseLinearSchedulerUp,
    PiecewiseLinearSchedulerDown,
)

"""
python pud/algos/lr_scheduler/tests/test_scheduler.py TestScheduler.test_linear_up
python pud/algos/lr_scheduler/tests/test_scheduler.py TestScheduler.test_linear_down
python pud/algos/lr_scheduler/tests/test_scheduler.py TestScheduler.test_hybrid_scheduler
"""


def plot(x, y, ax: Axes, start: float, stop: float):
    ax.plot(
        [start] * 100,
        np.linspace(-0.5, 1, 100),
        marker="o",
        color="r",
        linestyle="-",
        linewidth=2,
        markersize=2,
    )

    ax.plot(
        [stop] * 100,
        np.linspace(-0.5, 1, 100),
        marker="o",
        color="r",
        linestyle="-",
        linewidth=2,
        markersize=2,
    )

    ax.plot(
        x,
        y,
        marker="o",
        color="b",
        linestyle="-",
        linewidth=2,
        markersize=2,
        label="schedule",
    )

    ax.legend()


class TestScheduler(unittest.TestCase):
    def test_linear_up(self):
        start = 15
        stop = 25

        sched_u = PiecewiseLinearSchedulerUp(start=start, stop=stop)
        fig, ax = plt.subplots()
        fig.tight_layout()
        x = np.arange(0, 200)
        y = [sched_u(xi) for xi in x]
        plot(x, y, ax, start, stop)

        fname = "pud/algos/lr_scheduler/output/linear_sched_up.png"
        fig.savefig(fname=fname, dpi=320)
        plt.close(fig)

    def test_linear_down(self):
        start = 15
        stop = 25

        sched_u = PiecewiseLinearSchedulerDown(start=start, stop=stop)
        fig, ax = plt.subplots()
        fig.tight_layout()
        x = np.arange(0, 200)
        y = [sched_u(xi) for xi in x]
        plot(x, y, ax, start, stop)

        fname = "pud/algos/lr_scheduler/output/linear_sched_down.png"
        fig.savefig(fname=fname, dpi=320)
        plt.close(fig)

    def test_hybrid_scheduler(self):
        list_scheds = [
            PiecewiseLinearSchedulerUp(15, 25),
            PiecewiseLinearSchedulerDown(25, 35),
            PiecewiseLinearSchedulerUp(35, 45),
            PiecewiseLinearSchedulerDown(45, 55),
        ]

        sched = HybridScheduler(start=0, stop=100, list_scheds=list_scheds)

        fig, ax = plt.subplots()
        fig.tight_layout()
        x = np.arange(0, 200)
        y = [sched(xi) for xi in x]
        plot(x, y, ax, 0, 100)

        fname = "pud/algos/lr_scheduler/output/linear_sched_hybrid.png"
        fig.savefig(fname=fname, dpi=320)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
