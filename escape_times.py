import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({'font.size': 17})
plt.rcParams["font.family"] = "serif"


def plot_escape_times():

    models = ["ISM", "CSM", "Superbubble"]
    linewidths = [1, 2, 2]
    total_data = np.array([])

    fig = plt.figure()

    for model in models:
        data = np.genfromtxt(f"Escape Times/{model}.csv")
        total_data = np.append(total_data, data)

    _, bins = np.histogram(total_data, bins=50)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))

    for i in range(len(models)):
        model = models[i]
        linewidth = linewidths[i]
        data = np.genfromtxt(f"Escape Times/{model}.csv")

        plt.hist(data, bins=logbins, histtype="step", linewidth=linewidth, label=f"{model}")

    plt.axvline(x=342, linestyle="--", color="red",
                label=r"Geminga: 342 kyr")
    plt.axvline(x=110, linestyle="-.", color="red",
                label=r"Monogem: 110 kyr")
    
    plt.xlabel("Escape Time [kyr]")
    plt.ylabel("Pulsars")
    plt.xscale("log")
    plt.legend(fontsize=11)
    plt.grid()
    fig.tight_layout()
    plt.savefig("Project Summary/Images/all escape times.pdf")
    plt.savefig("CSM_plots/all escape times.pdf")
    plt.show()


def plot_escape_times_ISM():

    models = ["ISM"]
    linewidths = [2]
    total_data = np.array([])

    fig = plt.figure()

    for model in models:
        data = np.genfromtxt(f"Escape Times/{model}.csv")
        total_data = np.append(total_data, data)

    _, bins = np.histogram(total_data, bins=50)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))

    for i in range(len(models)):
        model = models[i]
        linewidth = linewidths[i]
        data = np.genfromtxt(f"Escape Times/{model}.csv")

        plt.hist(data, bins=logbins, histtype="step", linewidth=linewidth, label=f"{model}")

    plt.axvline(x=342, linestyle="--", color="red",
                label=r"Geminga: 342 kyr")
    plt.axvline(x=110, linestyle="-.", color="red",
                label=r"Monogem: 110 kyr")
    
    plt.xlabel("Escape Time [kyr]")
    plt.ylabel("Pulsars")
    plt.xscale("log")
    plt.legend(fontsize=11)
    plt.grid()
    fig.tight_layout()
    plt.savefig("Project Summary/Images/ISM escape times.pdf")
    plt.savefig("CSM_plots/ISM escape times.pdf")
    plt.show()


if __name__ == "__main__":

    plot_escape_times()
    plot_escape_times_ISM()

    1
