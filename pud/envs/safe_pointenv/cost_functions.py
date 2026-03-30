import numpy as np
from typing import Union
import matplotlib.pyplot as plt


def cost_from_linear_distance(
    d: Union[float, np.ndarray], r: float
) -> Union[float, np.ndarray]:
    """
    Steady transition between safe and unsafe
    """
    r"""
        J_{c}(d)\gets\begin{cases}
                (-2/r)(d)+2 & d\le r\\
                0 & d>r
                \end{cases}
    d: distance from obstacle
    r: radius that defines the vicinity of cost function
    """
    d_arr = d
    if isinstance(d, float):
        d_arr = np.array([d])

    assert isinstance(d_arr, np.ndarray)
    Jc = -2.0 / r * d_arr + 2.0
    Jc[d > r] = 0.0

    if len(Jc) == 1:
        Jc = float(np.squeeze(Jc))
    return Jc


def cost_from_cosine_distance(
    d: Union[float, np.ndarray], r: float
) -> Union[float, np.ndarray]:
    """
    fast transition between safe and unsafe, not good for learning intermediate values
    """
    r"""
        J_{c}(d)\gets\begin{cases}
                \cos\left(\frac{2\pi d}{2r}\right)+1 & d\le r\\
                0 & d>r
                \end{cases}
    """
    d_arr = d
    if isinstance(d, float):
        d_arr = np.array([d])

    Jc = np.cos(2.0 * np.pi * d_arr / (2.0 * r)) + 1.0
    Jc[d > r] = 0.0

    if len(Jc) == 1:
        Jc = float(np.squeeze(Jc))
    return Jc


def const_cost_from_distance(
    d: Union[float, np.ndarray], r: float
) -> Union[float, np.ndarray]:
    """
    simple constant cost when the distance is within the distance threshold r
    """
    r"""
        J_{c}(d)\gets\begin{cases}
                1 & d\le r\\
                0 & d>r
                \end{cases}
    """
    d_arr = d
    if isinstance(d, float):
        d_arr = np.array([d])

    Jc = np.ones_like(d_arr)
    Jc[d > r] = 0.0

    if len(Jc) == 1:
        Jc = float(np.squeeze(Jc))
    return Jc


if __name__ == "__main__":
    import functools
    from pathlib import Path

    r = 5
    d = np.linspace(0, 50, 100)
    Jc = cost_from_linear_distance(d, r)
    Jc_2 = cost_from_cosine_distance(0.0, 2.0)

    cost_f = functools.partial(cost_from_cosine_distance, r=r)

    out_dir = Path("pud/envs/safe_pointenv")
    out_dir.mkdir(exist_ok=True, parents=True)

    plt.plot(d, Jc)
    plt.savefig(out_dir.joinpath("cost_linear_d.png"), dpi=300)
