from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from zeroordergalaxy import give_n_ISM

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

        plt.hist(data, bins=logbins, density=True, histtype="step", label=f"Model {name}")
    
    n = give_n_ISM(10000)
    _, bins = np.histogram(n, bins=50)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    plt.hist(n, bins=logbins, histtype="step", density = True, label=f"Leahy 2020")
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


if __name__ == "__main__":

    plot_H2_density_Mertsch()

    1
