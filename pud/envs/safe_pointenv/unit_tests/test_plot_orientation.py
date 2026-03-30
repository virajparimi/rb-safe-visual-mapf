import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from pud.envs.safe_pointenv.safe_pointenv import plot_safe_walls, plot_trajs

L = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
)


out_dir = Path("pud/envs/safe_pointenv/unit_tests/outputs")
out_dir.mkdir(parents=True, exist_ok=True)
L_reading = np.loadtxt("pud/envs/safe_pointenv/unit_tests/L_reading.txt", delimiter=",")

L90 = np.rot90(L)

L270 = np.rot90(L, k=3)

walls = L270
fname = "L270.jpg"

walls = L
fname = "L.jpg"

fname = out_dir.joinpath(fname)

fig, ax = plt.subplots()

ax = plot_safe_walls(walls=walls, cost_map=None, cost_limit=None, ax=ax)

traj = L_reading * np.array(list(walls.shape), dtype=float)

ax = plot_trajs(
    list_trajs=[traj],
    walls=walls,
    ax=ax,
    starts=np.array([traj[0]]),
    goals=np.array([traj[-1]]),
)

ax.set_xlim((0, 1))
ax.set_ylim((0, 1))
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect("equal", adjustable="box")
fig.savefig(fname=fname, dpi=320, bbox_inches="tight")
plt.close(fig)
