from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from zeroordergalaxy import give_n_ISM
from sn_bubble import make_lognormal

def plot_H2_density_Mertsch():

    names = ["BEG03", "SBM15"]

    fig = plt.figure()

    for name in names:

        hdul = fits.open(f"ISM_density/H2_dens_mean_{name}.fits")

        data = hdul[0].data.flatten()
        data = data*1e6
        print(f"{name} mean n_ISM   = {np.mean(data)} cm-3")
        print(f"{name} median n_ISM = {np.median(data)} cm-3")

        _, bins = np.histogram(data, bins=5000)
        logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))

        plt.hist(data, bins=logbins, density=True, histtype="step",
                 label=f"Model {name}")
    
    n = give_n_ISM(10000)
    _, bins = np.histogram(n, bins=50)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    plt.hist(n, bins=logbins, histtype="step", density = True,
             label=f"Leahy 2020")
    plt.axvline(x=1)

    plt.xlabel(r"$n_\mathrm{ISM}$ [cm$^{-3}$]")
    plt.ylabel(r"Counts")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    fig.tight_layout()
    plt.savefig("Project Summary/Images/f(ISM)_H2_lines_Mertsch.pdf")
    plt.show()


def plot_SNRs_densities_Leahy():

    names = ["?", "CC", "Ia"]

    fig, ax = plt.subplots(3, 1, sharex=True)

    for i in range(len(names)):

        data = np.genfromtxt(f"ISM_density/Leahy_density_{names[i]}.txt",
                             delimiter=",", skip_header=1)

        E, Eu, El = data[:,1]*1e50, data[:,2]*1e50, data[:,3]*1e50
        n, nu, nl = data[:,4]*1e-2, data[:,5]*1e-2, data[:,6]*1e-2

        ax[i].errorbar(n, E, xerr=np.array([nu, nl]), yerr=np.array([Eu, El]),
                        fmt='o', label=f"Type {names[i]}")

        ax[i].set_xlabel(r"$n_\mathrm{ISM}$ [cm$^{-3}$]")
        ax[i].set_ylabel(r"$E_\mathrm{SN}$ [erg]")
        ax[i].set_xscale("log")
        ax[i].set_yscale("log")
        ax[i].legend()
        ax[i].grid()

    fig.tight_layout()
    plt.savefig("Project Summary/Images/f(ISM)_Leahy_SNRs.pdf")
    plt.show()


def plot_SNRs_densities_Leahy_hist():

    names = ["?", "CC", "Ia"]

    n_tot = np.array([])

    for i in range(len(names)):

        data = np.genfromtxt(f"ISM_density/Leahy_density_{names[i]}.txt",
                             delimiter=",", skip_header=1)

        n, nu, nl = data[:,4]*1e-2, data[:,5]*1e-2, data[:,6]*1e-2
        n_tot = np.append(n_tot, n)

        _, bins = np.histogram(n_tot, bins=int(len(n)/1.5))
        logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
        counts, edges = np.histogram(n_tot, bins=logbins)
        edges = [edges[i] + (edges[i+1] - edges[i])/2 for i in range(len(edges)-1)]
        counts = counts/np.sum(edges*counts)

        pOpt, pCov = opt.curve_fit(make_lognormal, edges, counts, p0=(0.069, 5.1))
        print(pOpt)

        x = np.logspace(np.log10(np.min(edges)), np.log10(np.max(edges)), 1000)

        fig = plt.figure()

        plt.plot(x, make_lognormal(x, *pOpt), linestyle="-", label=f"Fit {names[i]}")
        plt.plot(edges, counts, linestyle="", marker="o", label=f"Type {names[i]}")

        plt.xlabel(r"$n_\mathrm{ISM}$ [cm$^{-3}$]")
        plt.ylabel(r"SNRs")
        plt.xscale("log")
        plt.legend()
        plt.grid()

        fig.tight_layout()
        plt.savefig(f"Project Summary/Images/f(n_ISM)_Leahy_SNRs_type{names[i]}.pdf")
        plt.show()

    _, bins = np.histogram(n_tot, bins=int(len(n_tot)/3))
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    counts, edges = np.histogram(n_tot, bins=logbins)
    edges = [edges[i] + (edges[i+1] - edges[i])/2 for i in range(len(edges)-1)]
    counts = counts/np.sum(edges*counts)

    pOpt, pCov = opt.curve_fit(make_lognormal, edges, counts, p0=(0.069, 5.1))
    mu, sigma = pOpt
    print(pOpt)

    x = np.logspace(np.log10(np.min(edges)), np.log10(np.max(edges)), 1000)

    fig = plt.figure()

    plt.plot(x, make_lognormal(x, *pOpt), linestyle="-", label="Fit")
    plt.plot(edges, counts, linestyle="", marker="o", label=f"All types")

    plt.xlabel(r"$n_\mathrm{ISM}$ [cm$^{-3}$]")
    plt.ylabel(r"SNRs")
    plt.xscale("log")
    plt.legend()
    plt.grid()

    fig.tight_layout()
    plt.savefig(f"Project Summary/Images/f(n_ISM)_Leahy_SNRs_type.pdf")
    plt.show()


def plot_SNR_MHD_simulation_Iurii():
    data = np.genfromtxt("ISM_density/log_fileEL", skip_header=1)
    time = data[:,0]
    radius = data[:,5]/3e18 #pc
    velocity = data[:,6]

    fig = plt.figure()

    plt.plot(time, radius)
    plt.xlabel(r"Time [yr]")
    plt.ylabel(r"Radius [pc]")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()
    fig.tight_layout()
    plt.savefig(f"Project Summary/Images/SNR_MHD_Iurii_radius.pdf")
    plt.show()

if __name__ == "__main__":

    # plot_H2_density_Mertsch()
    plot_SNRs_densities_Leahy_hist()
    # plot_SNR_MHD_simulation_Iurii()

    1
