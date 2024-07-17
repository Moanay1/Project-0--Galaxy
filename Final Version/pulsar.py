import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.optimize as opt
import tools as tools

def pick_kick_velocity_component(v: np.ndarray) -> np.ndarray:
    """Parametrization from Faucher-Giguère et al. 2006 eq 7"""
    w, sigma_v1, sigma_v2 = 0.9, 160*1e5, 780*1e5
    norm = 10000000  # optimum normalization parameter
    # for the value selection process
    return (w*tools.make_gaussian(v, mu=0, sigma=sigma_v1)
            + (1-w)*tools.make_gaussian(v, mu=0, sigma=sigma_v2)) * norm # cm/s


def give_kick_velocity() -> np.ndarray:
    v_x = tools.give_random_value(pick_kick_velocity_component, -2000e5, 2000e5)
    v_y = tools.give_random_value(pick_kick_velocity_component, -2000e5, 2000e5)
    v_z = tools.give_random_value(pick_kick_velocity_component, -2000e5, 2000e5)
    return np.linalg.norm([v_x, v_y, v_z])


def test_kick_velocity() -> None:
    arr = []
    for _ in tqdm(range(10000)):
        arr.append(give_kick_velocity())

    fig = plt.figure()
    plt.hist(arr, histtype="step", density=True, bins=100)
    plt.xlabel(r"$v_\mathrm{k}$ [cm$\cdot$s$^{-1}$]")
    plt.ylabel("PDF")
    plt.xlim(np.min(arr), np.max(arr))
    plt.grid()
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":

    #test_kick_velocity()
    1