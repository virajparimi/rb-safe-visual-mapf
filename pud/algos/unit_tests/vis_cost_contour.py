import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from pud.envs.safe_pointenv.safe_pointenv import SafePointEnv, plot_safe_walls


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--maze_name", type=str, default="CentralObstacle", help="maze name"
    )
    parser.add_argument(
        "--cost_name", type=str, default="linear", help="cost function name"
    )
    parser.add_argument(
        "--radius", type=float, default=25, help="cost function radius from obstacle"
    )
    parser.add_argument("--cost_limit", type=float, default=2.0, help="cost limit")
    parser.add_argument(
        "--resize_factor", type=int, default=5, help="maze resize factor"
    )

    args = parser.parse_args()

    env = SafePointEnv(
        walls=args.maze_name,
        resize_factor=args.resize_factor,
        thin=False,
        cost_limit=args.cost_limit,
        cost_f_args={
            "name": args.cost_name,
            "radius": args.radius,
        },
    )

    cost_map = env.get_cost_map()
    map = env.get_map()

    x = np.arange(cost_map.shape[0]) / float(map.shape[0])
    y = np.arange(cost_map.shape[1]) / float(map.shape[1])

    X, Y = np.meshgrid(x, y)
    fig, ax = plt.subplots()
    # Somehow contourf plots a different orientation than pyplot.plot, have to switch axes
    CS = ax.contourf(Y, X, cost_map, cmap=mpl.colormaps["cool"], alpha=0.5)

    ax = plot_safe_walls(
        walls=env.get_map(), cost_map=env.get_cost_map(), cost_limit=0, ax=ax
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        "cost contour with resize = {:.2f} radius = {:.2f}".format(
            args.resize_factor, args.radius
        )
    )
    fig.colorbar(CS)
    fig.savefig(
        "temp/cost_f_contour_{}_{}_resize={}_r={}.jpg".format(
            args.maze_name, args.cost_name, args.resize_factor, args.radius
        ),
        dpi=300,
    )
    plt.close(fig)
