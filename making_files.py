import firstordergalaxy
import zeroordergalaxy
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import scipy.optimize as opt
import os
import time


def make_lognormal(zz: np.ndarray, mu: float, sigma: float, A) -> np.ndarray:
    """Returns the log normal function of `zz` for parameters `mu` and 
    `sigma`.

    Args:
        zz (np.ndarray): point for which we want the lognormal
        mu (float): mean
        sigma (float): variance

    Returns:
        float: lognormal value
    """
    normal_std = np.log10(sigma)
    normal_mean = np.log(mu)
    return A/np.sqrt(2*np.pi*normal_std**2)\
            * np.exp(-(np.log(zz)-normal_mean)**2/(2*normal_std**2))


def make_gaussian(x: float, mu: float, sigma: float, A: float) -> float:
    return A*np.exp(-(x-mu)**2/(2*sigma**2))


def binning(x: np.ndarray) -> np.ndarray:

    hist, edges = np.histogram(x, bins=20)
    edges = (edges + (edges[1] - edges[0])/2)[:-1]

    return hist, edges


def fit_gaussian(x: np.ndarray) -> np.ndarray:

    hist, edges = binning(x)

    pOpt, pCov = opt.curve_fit(make_gaussian, edges, hist,
                               p0=(np.mean(x), 5, 10),
                               bounds=((0, 0, 0), (100, 1000, 1000)))
    mu, sigma, A = pOpt
    dmu, dsigma, dA = np.sqrt(np.diag(pCov))
    return mu, sigma, A, edges

def shortest_interval(x, y, beta):
    deltax = 1

    maxx = len(x)-2

    mode = np.max(y)
    p = np.where(y == mode)[0][0]
    xleft = p
    xright = p
    integral = mode * deltax
    all_count = np.sum(y)

    while (integral < beta):
        if xright == maxx:
            yright = y[xright]
        else:
            yright = y[xright + 1]

        if xleft == 0:
            yleft = y[xleft]
        else:
            yleft = y[xleft - 1]

        intleft = yleft * deltax
        intright = yright * deltax

        if (intleft > intright) and xleft != 0:
            integral += intleft
            if xleft == 0:
                xleft = 0
            else:
                xleft += -1

        elif (intleft < intright) and xright != maxx:
            integral += intright
            if xright == maxx:
                xright = maxx
            else:
                xright += +1

        else:
            integral += yleft * deltax + yright * deltax
            if xright == maxx:
                xright = maxx
            else:
                xright += +1

            if xleft == 0:
                xleft = 0
            else:
                xleft += -1

    # print(x[xleft], x[xright], integral/all_count)
    return x[xleft], x[xright], integral/all_count

def argmedian(x):
  return np.argpartition(x, len(x) // 2)[len(x) // 2]


def make_galaxies(n: int = 20) -> None:

    # Only add new galaxies to the directory without removing the ancient ones.
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

    colors = ["orange"]

    fig = plt.figure()

    number_of_pulsars = 1000
    realizations = 100

    mu_arr = np.array([])
    xL_arr, xU_arr = np.array([]), np.array([])

    file = open("Galaxies/Pulsars_n{number}_r{realizations}.csv".format(
        number=number_of_pulsars, realizations=realizations
    ), "w")
    file.write("t [yr], Percentage [%], Percentage err -, Percentage err +\n")

    for t_ in tqdm(t_arr):
        proportion_arr = np.array([])
        for _ in tqdm(range(realizations)):
            result = 0
            while result == 0:
                result = zeroordergalaxy.give_is_inside_proportion(
                    t_, n=number_of_pulsars, phase="PDS")
            proportion_arr = np.append(proportion_arr, result)

            mu = np.median(proportion_arr, axis=0)
            xL = np.percentile(proportion_arr, 5, axis=0)
            xU = np.percentile(proportion_arr, 95, axis=0)

            mu_arr = np.append(mu_arr, mu)
            xL_arr = np.append(xL_arr, xL)
            xU_arr = np.append(xU_arr, xU)

        file.write("{t:.2f}, {percentage:.2f}, {errmin:.2f}, {errmax:.2f}\n".
                   format(t=t_, percentage=mu, errmin=xL, errmax=xU))

    plt.plot(t_arr/1e3, mu_arr, linewidth=0.5, color=colors[0])
    plt.fill_between(t_arr/1e3, xL_arr, xU_arr,
                        alpha=0.2, color=colors[0],
                        label=r"2$\sigma$")

    plt.axvline(x=342, linestyle="--", color="red",
                label=r"Geminga: 342 kyr")
    plt.axvline(x=110, linestyle="-.", color="red",
                label=r"Monogem: 110 kyr")

    plt.xscale("log")
    plt.xlabel("Pulsar age [kyr]")
    plt.ylabel("Pulsars inside their SNR [%]")
    plt.grid()
    plt.legend(fontsize=12)
    fig.tight_layout()
    plt.savefig(r"Project Summary/Images/is_inside_evolution_with_time_same_age.pdf")
    plt.show()


def give_inside_proportion_with_time_varying_parameters() -> None:

    colors = ["blue", "orange"]
    variable = [True, False]
    labels = [r"Variance of $E_\mathrm{SN}$, $n_\mathrm{ISM}$", "No variance"]
    labels2 = ["Variance", "No variance"]

    fig = plt.figure()

    number_of_pulsars = 1000
    realizations = 100

    for i in range(len(variable)):

        time_pulsars = np.array([])
        mu_arr = np.array([])
        xL_arr, xU_arr = np.array([]), np.array([])

        file = open("Galaxies/Pulsars_n{number}_r{realizations}_varying_parameters_{parameters}.csv".
        format(number=number_of_pulsars, realizations=realizations, parameters=variable[i]), "w")
        file.write("t [yr], Percentage [%], Percentage err -, Percentage err +\n")

        for t_ in tqdm(t_arr):
            proportion_arr = np.array([])
            for _ in range(realizations):
                result = 0
                while result == 0:
                    start = time.process_time()
                    result = zeroordergalaxy.give_is_inside_proportion(
                        t_, n=number_of_pulsars,
                        variable_parameters=variable[i])
                    time_pulsars = np.append(time_pulsars,
                            (time.process_time() - start)/number_of_pulsars)
                proportion_arr = np.append(proportion_arr, result)
                
            mu = np.median(proportion_arr, axis=0)
            xL = np.percentile(proportion_arr, 5, axis=0)
            xU = np.percentile(proportion_arr, 95, axis=0)

            mu_arr = np.append(mu_arr, mu)
            xL_arr = np.append(xL_arr, xL)
            xU_arr = np.append(xU_arr, xU)

            file.write("{t:.2f}, {percentage:.2f}, {errmin:.2f}, {errmax:.2f}\n".
                   format(t=t_, percentage=mu, errmin=xL, errmax=xU))

        print(np.mean(time_pulsars))

        plt.plot(t_arr/1e3, mu_arr, linewidth=0.5, color=colors[i],
                 label=labels[i])
        plt.fill_between(t_arr/1e3, xL_arr, xU_arr,
                         alpha=0.2, color=colors[i],
                         label=labels2[i])

    plt.axvline(x=342, linestyle="--", color="red",
                label=r"Geminga: 342 kyr")
    plt.axvline(x=110, linestyle="-.", color="red",
                label=r"Monogem: 110 kyr")

    plt.xscale("log")
    plt.xlabel("Pulsar age [kyr]")
    plt.ylabel("Pulsars inside their SNR [%]")
    plt.grid()
    plt.legend(fontsize=12)
    fig.tight_layout()
    plt.savefig(r"Project Summary/Images/is_inside_evolution_with_time_varying_parameters.pdf")
    plt.show()


def check_number_of_realisations():

    t_arr = np.array([1, 2, 5])*1e5 #yr
    realisations_arr = np.logspace(0.5, 3, 20)
    number_of_pulsars = 1000

    fig = plt.figure()

    for t_ in t_arr:
        mu_arr = np.array([])
        xL_arr, xU_arr = np.array([]), np.array([])

        for realisations in tqdm(realisations_arr):
            proportion_arr = np.array([])
            for _ in range(int(realisations)):
                result = 0
                while result == 0:
                    result = zeroordergalaxy.give_is_inside_proportion(
                        t_, n=number_of_pulsars,
                        variable_parameters=True)
                proportion_arr = np.append(proportion_arr, result)

            mu = np.median(proportion_arr, axis=0)
            xL = np.percentile(proportion_arr, 5, axis=0)
            xU = np.percentile(proportion_arr, 95, axis=0)

            mu_arr = np.append(mu_arr, mu)
            xL_arr = np.append(xL_arr, xL)
            xU_arr = np.append(xU_arr, xU)

        plt.plot(realisations_arr, mu_arr, label=f"t={t_/1e3} kyr")
        plt.fill_between(realisations_arr, xL_arr, xU_arr,
                         alpha=0.2, color=plt.gca().lines[-1].get_color())
    
    plt.xscale("log")
    plt.xlabel("Number of realisations")
    plt.ylabel("Mean percentage value")
    plt.ylim(0,100)
    plt.grid()
    plt.legend()
    fig.tight_layout()
    plt.savefig(r"Project Summary/Images/Check_number_realisations_2.pdf")
    plt.show()


def give_characteristic_PSR_age(P, dPdt):
    return P / (2*dPdt) / (1e3*np.pi*1e7) # kyr


def plot_catalog_PSR_in_SNR():
    data = np.genfromtxt("Pulsars_in_SNRs.txt", delimiter=",", skip_header=2)
    period = data[:,1]
    period_derivative = data[:,2]*1e-15

    characteristic_age = give_characteristic_PSR_age(period, period_derivative)
    _, bins = np.histogram(characteristic_age, bins=20)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))

    fig = plt.figure()

    plt.scatter(period, period_derivative)
    plt.xlabel(r"$P$ [s]")
    plt.ylabel(r"$\mathrm{d}P/\mathrm{d}t$ [s/s]")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()
    fig.tight_layout()
    plt.savefig(r"Project Summary/Images/P_Pdot_in_SNR.pdf")
    plt.show()

    fig = plt.figure()

    plt.hist(characteristic_age, bins=logbins, histtype="step",
             label=r"Characteristic age")
    plt.xlabel(r"$t$ [kyr]")
    plt.ylabel(r"Number of pulsars")
    plt.xscale("log")
    plt.grid()
    plt.legend()
    fig.tight_layout()
    plt.savefig(r"Project Summary/Images/PSR_in_SNR_age.pdf")
    plt.show()


def plot_catalog_PSR_out_SNR(): 
    data = np.genfromtxt("Pulsars_out_SNRs.txt", delimiter=",", skip_header=2)
    period = data[:,3]
    period_derivative = np.abs(data[:,6])

    characteristic_age = give_characteristic_PSR_age(period, period_derivative)
    _, bins = np.histogram(characteristic_age, bins=200)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))

    fig = plt.figure()

    plt.scatter(period, period_derivative)
    plt.xlabel(r"$P$ [s]")
    plt.ylabel(r"$\mathrm{d}P/\mathrm{d}t$ [s/s]")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()
    fig.tight_layout()
    plt.savefig(r"Project Summary/Images/P_Pdot_out_SNR.pdf")
    plt.show()

    fig = plt.figure()

    plt.hist(characteristic_age, bins=logbins, histtype="step",
             label=r"Characteristic age")
    plt.xlabel(r"$t$ [kyr]")
    plt.ylabel(r"Number of pulsars")
    plt.xscale("log")
    plt.grid()
    plt.legend()
    fig.tight_layout()
    plt.savefig(r"Project Summary/Images/PSR_out_SNR_age.pdf")
    plt.show()


def plot_age_SNR():

    fig = plt.figure()

    data = np.genfromtxt("Pulsars_out_SNRs.txt", delimiter=",", skip_header=2)
    period = data[:,3]
    period_derivative = np.abs(data[:,6])

    characteristic_age = give_characteristic_PSR_age(period, period_derivative)
    _, bins = np.histogram(characteristic_age,
                           bins=20)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    plt.hist(characteristic_age, bins=logbins,
             weights=np.ones_like(characteristic_age)/len(characteristic_age),
             histtype="step",
             label=r"Outside SNR")
    
    

    data = np.genfromtxt("Pulsars_in_SNRs.txt", delimiter=",", skip_header=2)
    period = data[:,1]
    period_derivative = data[:,2]*1e-15

    characteristic_age = give_characteristic_PSR_age(period, period_derivative)
    _, bins = np.histogram(characteristic_age,
                           bins=20)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    plt.hist(characteristic_age, bins=logbins,
             weights=np.ones_like(characteristic_age)/len(characteristic_age),
             histtype="step",
             label=r"Inside SNR")


    plt.xlabel(r"$t$ [kyr]")
    plt.ylabel(r"Number of pulsars")
    plt.xscale("log")
    plt.grid()
    plt.legend()
    fig.tight_layout()
    plt.savefig(r"Project Summary/Images/PSR_in_out_SNR.pdf")
    plt.show()


def plot_morphology_PSR():

    fig = plt.figure()

    data = np.genfromtxt("Pulsars_in_SNRs.txt", delimiter=",", skip_header=2,
                         dtype=str)
    type = data[:,4]

    plt.hist(type, histtype="step")

    plt.xlabel(r"Type")
    plt.ylabel(r"Number of pulsars")
    plt.grid()
    fig.tight_layout()
    plt.savefig(r"Project Summary/Images/PSR_in_SNR_type.pdf")
    plt.show()


def plot_bow_shock_time_distribution():

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, 
                                   gridspec_kw={'height_ratios': [2, 1]})

    colors = ["blue", "orange"]
    variable = [True, False]
    labels = [r"Variance of $E_\mathrm{SN}$, $n_\mathrm{ISM}$", "No variance"]
    labels2 = ["Variance", "No variance"]

    for i in range(len(variable)):

        data = np.genfromtxt(f"Galaxies/Pulsars_n1000_r100_varying_parameters_{variable[i]}.csv", skip_header=1, delimiter=",")
        time_pulsars = data[:,0]
        percentage = data[:,1]
        xL = data[:,2]
        xU = data[:,3]

        ax1.plot(time_pulsars/1e3, percentage, linewidth=0.5, color=colors[i],
                 label=labels[i])
        ax1.fill_between(t_arr/1e3, xL, xU,
                         alpha=0.2, color=colors[i],
                         label=labels2[i])
        
    ax1.axvline(x=342, linestyle="--", color="red",
                label=r"Geminga: 342 kyr")
    ax1.axvline(x=110, linestyle="-.", color="red",
                label=r"Monogem: 110 kyr")
    
    ax1.set_xscale("log")
    ax1.set_xlabel("Pulsar age [kyr]")
    ax1.set_ylabel("Pulsars in SNR [%]")
    ax1.legend(fontsize = 9)
    ax1.grid()

    t_bs_arr = zeroordergalaxy.give_bow_shock_time(n = 100000)

    _, bins = np.histogram(t_bs_arr, bins=100)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))

    ax2.hist(t_bs_arr, bins=logbins, histtype="step")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$t_\mathrm{BS}$ [kyr]")
    ax2.set_ylabel(r"Pulsars")
    ax2.grid()
    fig.tight_layout()
    plt.savefig(r"Project Summary/Images/t_BS_evolution.pdf")
    plt.show()


def plot_ATNF_pulsars_t_BS():
    """Use the definition of Martin+2022 for the selection of pulsars
    from the ATNF catalog:
    Dist <= 1 kpc
    Age > 60e3 yr (and Age < 1e8 yr)
    Age < 1e8 yr
    Edot >= 1e33 erg.s-1
    """

    data = np.genfromtxt("Nearby Pulsars.txt", skip_header=3)

    names = data[:,1]
    distances = data[:,3] # kpc
    ages = data[:,4]/1e3 # kyr
    powers = data[:,5] # erg.s-1

    colors = [plt.cm.jet(i) for i in powers/np.max(powers)]

    fig, axes = plt.subplots(2, 1, sharex=True, 
                                   gridspec_kw={'height_ratios': [2, 1]})
    ax1, ax2 = axes

    ax1.scatter(ages, distances, linestyle = "--", color=colors)
    #sm = plt.cm.ScalarMappable(cmap=plt.get_cmap("jet"), norm=col.LogNorm(
        #vmin=np.min(powers), vmax=np.max(powers)))
    #cbar = fig.colorbar(sm, ax=axes.ravel().tolist())
    #cbar.set_label(r"$\dot{E}$ [erg s$^{-1}$]")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$t$ [kyr]")
    ax1.set_ylabel(r"$d$ [kpc]")
    ax1.grid()


    t_bs_arr = zeroordergalaxy.give_bow_shock_time(n = 10000)

    _, bins = np.histogram(t_bs_arr, bins=20)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))

    ax2.hist(t_bs_arr, bins=logbins, histtype="step")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$t_\mathrm{BS}$ [kyr]")
    ax2.set_ylabel(r"Pulsars")
    ax2.grid()
    fig.tight_layout()
    plt.savefig(r"Project Summary/Images/ATNF_pulsars_distribution.pdf")
    plt.show()


def plot_giacinti_pulsars():

    data = np.genfromtxt("Giacinti_pulsars.txt", delimiter=",", skip_header=1)

    luminosity = 10**data[:, 2] # erg/s
    characteristic_age = data[:,3] # kyr
    distance = data[:,4] # kpc

    fig = plt.figure()

    plt.scatter(characteristic_age[2:], luminosity[2:]/distance[2:])
    plt.scatter(characteristic_age[:2], luminosity[:2]/distance[:2], marker="*")
    plt.xlabel(r"Characteristic age [kyr]")
    plt.ylabel(r"$L/d$ [erg/s/kpc]")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()
    fig.tight_layout()
    plt.show()


def plot_flux_pulsars(zoom:bool = True):

    data = np.genfromtxt("flux(time)_catalog.txt", delimiter=",",
                         skip_header=4, dtype=str)

    name = data[:,1]
    characteristic_age = np.float64(data[:,10])/1e3 # kyr
    distance = np.float64(data[:,9]) # kpc
    luminosity = np.float64(data[:,11]) # erg/s
    Edotd2 = np.float64(data[:,12]) # erg/s/kpc^2

    indices_monogem_geminga = [i for i in range(len(name)) 
               if name[i] in [" B0656+14", " J0633+1746"]]
    
    index_B1055_52 = [i for i in range(len(name)) 
               if name[i] in [" B1055-52"]]

    # Computation of the number of pulsars that have similar properties
    #   to Geminga
    ####################################################################
    # FLUX (Luminosity over distance squared)
    ####################################################################

    tolerance = 3
    Geminga_like_pulsars = []

    for i in range(len(name)):
        if distance[i] < distance[indices_monogem_geminga[1]] * tolerance:
            if Edotd2[i] > Edotd2[indices_monogem_geminga[1]] / tolerance:
                Geminga_like_pulsars.append(i)

    size = np.log10(characteristic_age) / np.log10(np.max(characteristic_age))

    print(np.array(Geminga_like_pulsars)+1)

    print(characteristic_age[Geminga_like_pulsars])
    print(distance[Geminga_like_pulsars])
    print(Edotd2[Geminga_like_pulsars])

    fig = plt.figure()

    p = plt.scatter(distance, Edotd2, s=size*25, c=characteristic_age,
                norm=col.LogNorm(vmin=np.min(characteristic_age),
                                 vmax=np.max(characteristic_age)),
                cmap="gist_rainbow")
    plt.scatter(distance[indices_monogem_geminga[1]],
                Edotd2[indices_monogem_geminga[1]], s=100, marker="*",
                label="Geminga")
    plt.scatter(distance[indices_monogem_geminga[0]],
                Edotd2[indices_monogem_geminga[0]], s=100, marker="*",
                label="Monogem")
    plt.scatter(distance[index_B1055_52],
                Edotd2[index_B1055_52], s=100, marker="*",
                label="B1055-52")
    
    plt.axvline(x=distance[indices_monogem_geminga[1]]*tolerance,
               color="black", linestyle="--")
    plt.axhline(y=Edotd2[indices_monogem_geminga[1]]/tolerance,
               color="black", linestyle="--")

    plt.xlabel(r"Distance [kpc]")
    plt.ylabel(r"$\dot{E}/d^2$ [erg/s/kpc$^2$]")
    plt.xscale("log")
    plt.yscale("log")
    clb = plt.colorbar(p)
    clb.set_label(r"Pulsar age [kyr]")
    plt.grid()
    plt.legend(fontsize=10)
    if zoom:
        plt.xlim(right=distance[indices_monogem_geminga[1]]*tolerance)
        plt.ylim(bottom=Edotd2[indices_monogem_geminga[1]]/tolerance)
        fig.tight_layout()
        plt.savefig("Project Summary/Images/ATNF_pulsars_properties_zoom.pdf")
    else:
        fig.tight_layout()
        plt.savefig("Project Summary/Images/ATNF_pulsars_properties.pdf")
    plt.show()


def plot_luminosity_over_distance_pulsars(zoom:bool = True):

    data = np.genfromtxt("flux(time)_catalog.txt", delimiter=",",
                         skip_header=4, dtype=str)

    name = data[:,1]
    characteristic_age = np.float64(data[:,10])/1e3 # kyr
    distance = np.float64(data[:,9]) # kpc
    luminosity = np.float64(data[:,11]) # erg/s
    Edotd2 = np.float64(data[:,12]) # erg/s/kpc^2
    false_flux = luminosity/distance # erg/s/kpc

    indices_monogem_geminga = [i for i in range(len(name)) 
               if name[i] in [" B0656+14", " J0633+1746"]]
    
    index_B1055_52 = [i for i in range(len(name)) 
               if name[i] in [" B1055-52"]]

    # Computation of the number of pulsars that have similar properties
    #   to Geminga
    ####################################################################
    # FLUX (Luminosity over distance squared)
    ####################################################################

    tolerance = 3
    Geminga_like_pulsars = []

    for i in range(len(name)):
        if distance[i] < distance[indices_monogem_geminga[1]] * tolerance:
            if false_flux[i] > false_flux[indices_monogem_geminga[1]] / tolerance:
                Geminga_like_pulsars.append(i)

    size = np.log10(characteristic_age) / np.log10(np.max(characteristic_age))

    print(np.array(Geminga_like_pulsars)+1)

    print(characteristic_age[Geminga_like_pulsars])
    print(distance[Geminga_like_pulsars])
    print(false_flux[Geminga_like_pulsars])

    fig = plt.figure()

    p = plt.scatter(distance, false_flux, s=size*25, c=characteristic_age,
                norm=col.LogNorm(vmin=np.min(characteristic_age),
                                 vmax=np.max(characteristic_age)),
                cmap="gist_rainbow")
    plt.scatter(distance[indices_monogem_geminga[1]],
                false_flux[indices_monogem_geminga[1]], s=100, marker="*",
                label="Geminga")
    plt.scatter(distance[indices_monogem_geminga[0]],
                false_flux[indices_monogem_geminga[0]], s=100, marker="*",
                label="Monogem")
    plt.scatter(distance[index_B1055_52],
                false_flux[index_B1055_52], s=100, marker="*",
                label="B1055-52")
    
    plt.axvline(x=distance[indices_monogem_geminga[1]]*tolerance,
               color="black", linestyle="--")
    plt.axhline(y=false_flux[indices_monogem_geminga[1]]/tolerance,
               color="black", linestyle="--")

    plt.xlabel(r"Distance [kpc]")
    plt.ylabel(r"$\dot{E}/d$ [erg/s/kpc]")
    plt.xscale("log")
    plt.yscale("log")
    clb = plt.colorbar(p)
    clb.set_label(r"Pulsar age [kyr]")
    plt.grid()
    plt.legend(fontsize=10)
    if zoom:
        plt.xlim(right=distance[indices_monogem_geminga[1]]*tolerance)
        plt.ylim(bottom=false_flux[indices_monogem_geminga[1]]/tolerance)
        fig.tight_layout()
        plt.savefig("Project Summary/Images/ATNF_pulsars_properties_zoom_luminosity_over_distance.pdf")
    else:
        fig.tight_layout()
        plt.savefig("Project Summary/Images/ATNF_pulsars_properties_luminosity_over_distance.pdf")
    plt.show()


def plot_period_PSR():

    data = np.genfromtxt("flux(time)_catalog.txt", delimiter=",",
                         skip_header=4)
    
    P0 = data[:, 3]
    P0 = P0[P0 > 6e-2] # remove the millisecond pulsars

    counts, bins = np.histogram(P0, bins=200)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    logbins2 = [logbins[i] + (logbins[i+1] - logbins[i])/2
                for i in range(len(logbins)-1)]
    
    counts, _ = np.histogram(P0, bins=logbins, density=True)

    pOpt, pCov = opt.curve_fit(make_lognormal, logbins2, counts,
                               p0=(0.3, 10, 1.75))
    mu, sigma, A = pOpt
    print(pOpt, np.sqrt(np.diag(pCov)))

    x = np.logspace(-2, 1, 1000)
    y = make_lognormal(x, mu, sigma, A)

    fig = plt.figure()

    plt.hist(P0, bins=logbins, density=True, histtype="step", label="Data")
    plt.plot(x, y, label="Fit")
    plt.xlabel(r"Period [s]")
    plt.ylabel(r"Pulsars")
    plt.xscale("log")
    plt.grid()
    plt.legend()
    fig.tight_layout()
    plt.savefig(r"Project Summary/Images/fit_data_P.pdf")
    plt.show()





t_arr = np.logspace(np.log10(1e3), np.log10(1e7), 40, endpoint=True)

if __name__ == "__main__":

    # make_galaxies()
    # give_inside_proportion()

    # give_inside_proportion_with_time_same_age()
    # give_inside_proportion_with_time_varying_parameters()

    # check_number_of_realisations()

    # plot_age_SNR()
    # plot_morphology_PSR()

    plot_bow_shock_time_distribution()
    # plot_ATNF_pulsars_t_BS()

    # plot_giacinti_pulsars()
    # plot_flux_pulsars()
    # plot_luminosity_over_distance_pulsars()

    # plot_period_PSR()

    1
