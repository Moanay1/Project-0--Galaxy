import firstordergalaxy
import zeroordergalaxy
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import os


def make_gaussian(x: float, mu: float, sigma: float, A: float) -> float:
    return A*np.exp(-(x-mu)**2/(2*sigma**2))


def binning(x: np.ndarray) -> np.ndarray:

    hist, edges = np.histogram(x, bins=20)
    edges = (edges + (edges[1] - edges[0])/2)[:-1]

    return hist, edges


def fit_gaussian(x: np.ndarray) -> np.ndarray:

    hist, edges = binning(x)

    pOpt, pCov = opt.curve_fit(make_gaussian, edges, hist,
                               p0=(np.mean(x), 5, 10))
    mu, sigma, A = pOpt
    dmu, dsigma, dA = np.sqrt(np.diag(pCov))
    return mu, sigma, A, edges


def shortest_interval_points(x: np.ndarray) -> np.ndarray:

    hist, edges = binning(x)

    A = 0
    all_count = np.sum(hist)
    imax, iL, iU = np.argmax(hist), 0, 0
    threshold = np.max(hist)

    while A/all_count < 0.90:  # 2 sigma
        threshold -= 1
        if imax-iL > 1:
            while hist[imax-iL] > threshold:
                iL += 1
        if imax+iU > 1:
            while hist[imax+iU] > threshold:
                iU += 1
        A = np.sum(hist[imax-iL:imax+iU])

    return edges[imax-iL], edges[imax+iU], A/all_count


def make_galaxies(n: int = 20) -> None:

    # Only add new galaxies to the directory withoutremoving the ancient ones.
    init = int(len(os.listdir('./Galaxies'))/2)

    print(n)

    for i in tqdm(range(init, init+n), desc="Number of Galaxies",):
        firstordergalaxy.create_file_first(name=i)
        zeroordergalaxy.create_file_zero(name=i)


def vary_age_galaxies(n: int = 5) -> None:

    init = int(len(os.listdir('./Age Evolution'))/len(t_arr))

    for n_ in tqdm(range(init, init+n)):
        for t_ in t_arr:
            firstordergalaxy.create_file_first(t=t_, name=n_)
            zeroordergalaxy.create_file_zero(t=t_, name=n_)


def give_inside_proportion() -> None:

    n = int(len(os.listdir('./Galaxies'))/2)
    proportion_0 = np.array([])
    proportion_1 = np.array([])

    for n_ in range(n):
        is_inside_0 = np.genfromtxt("Galaxies/Pulsars_0_{}.csv"
                                    .format(n_), usecols=6, dtype=bool,
                                    skip_header=1)
        proportion_0 = np.append(
            proportion_0, np.count_nonzero(is_inside_0)/len(is_inside_0)*100)

        is_inside_1 = np.genfromtxt("Galaxies/Pulsars_1_{}.csv"
                                    .format(n_), usecols=6, dtype=bool,
                                    skip_header=1)
        proportion_1 = np.append(
            proportion_1, np.count_nonzero(is_inside_1)/len(is_inside_1)*100)

    fig = plt.figure()
    plt.hist(proportion_0, histtype="step", bins=20, label=r"0 order")
    plt.hist(proportion_1, histtype="step", bins=20, label=r"1st order")
    plt.xlabel("Pulsars inside their SNR [%]")
    plt.legend()
    plt.grid()
    fig.tight_layout()
    plt.savefig(r"Project Summary/Images/is_inside_comparison.pdf")
    plt.show()

    for t_ in t_arr:
        for n_ in range(n):
            is_inside_0 = np.genfromtxt("Age Evolution/Pulsars_0_{}_{}_kyr.csv"
                                        .format(n_, t_/1e3), usecols=6,
                                        dtype=bool, skip_header=1)
            proportion_0 = np.append(
                proportion_0,
                np.count_nonzero(is_inside_0)/len(is_inside_0)*100)

            is_inside_1 = np.genfromtxt("Age Evolution/Pulsars_1_{}_{}_kyr.csv"
                                        .format(n_, t_/1e3), usecols=6,
                                        dtype=bool, skip_header=1)
            proportion_1 = np.append(
                proportion_1,
                np.count_nonzero(is_inside_1)/len(is_inside_1)*100)
            plt.plot(t_arr/1e3, proportion_0, linewidth=0.5, color="black",
                     label=r"0 order")
            plt.plot(t_arr/1e3, proportion_1, linewidth=0.5, color="red",
                     label=r"1st order")

    fig = plt.figure()

    plt.xscale("log")
    plt.xlabel("$t$ [kyr]")
    plt.ylabel("Pulsars inside their SNR [%]")
    plt.legend()
    plt.grid()
    fig.tight_layout()
    plt.savefig(r"Project Summary/Images/is_inside_evolution_with_time.pdf")
    plt.show()


def give_inside_proportion_with_time_same_age() -> None:

    colors = ["blue", "orange"]
    phases = ["ST", "PDS"]

    fig = plt.figure()

    for i in range(len(phases)):
        mu_arr, sigma_arr = np.array([]), np.array([])

        for t_ in tqdm(t_arr):
            proportion_arr = np.array([])
            for _ in range(100):
                proportion_arr = np.append(
                    proportion_arr,
                    zeroordergalaxy.give_is_inside_proportion(t_, n=100,
                                                              phase=phases[i]))
            mu, sigma, A, edges = fit_gaussian(proportion_arr)
            mu_arr = np.append(mu_arr, mu)
            sigma_arr = np.append(sigma_arr, sigma)

        plt.plot(t_arr/1e3, mu_arr, linewidth=0.5, color=colors[i],
                 label=f"{phases[i]}")
        plt.fill_between(t_arr/1e3, mu_arr-2*sigma_arr, mu_arr+2*sigma_arr,
                         alpha=0.2, color=colors[i],
                         label=r"2$\sigma$ "f"{phases[i]}")

    plt.axvline(x=342, linestyle="--", color="red",
                label=r"Geminga: 342 kyr")
    plt.axvline(x=110, linestyle="-.", color="red",
                label=r"Monogem: 110 kyr")

    # plt.plot(t_arr/1e3, max_arr, linewidth = 0.5, color="red")
    # plt.fill_between(t_arr/1e3, min_sigma_arr, min_sigma_arr, alpha = 0.2, color = "red", label = r"2 $\sigma$")

    plt.xscale("log")
    plt.xlabel("Pulsar age [kyr]")
    plt.ylabel("Pulsars inside their SNR [%]")
    plt.grid()
    plt.legend(fontsize=12)
    fig.tight_layout()
    plt.savefig(r"Project Summary/Images/is_inside_evolution_with_time\
                _same_age_comparison.pdf")
    plt.show()


t_arr = np.logspace(np.log10(10e3), np.log10(1e6), 20, endpoint=True)

if __name__ == "__main__":

    # make_galaxies()
    # give_inside_proportion()

    # vary_age_galaxies()
    # give_inside_proportion_evolution_with_time()

    give_inside_proportion_with_time_same_age()

    2
