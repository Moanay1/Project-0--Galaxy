import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({'font.size': 17})
plt.rcParams["font.family"] = "serif"


def plot_escape_times():

    models = ["CSM", "Superbubble"]
    linewidths = [2, 2]
    total_data = np.array([])

    fig = plt.figure()

    for model in models:
        data = np.genfromtxt(f"Escape Times/{model}.csv")
        total_data = np.append(total_data, data)

    _, bins = np.histogram(total_data, bins=500)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))

    for i in range(len(models)):
        model = models[i]
        linewidth = linewidths[i]
        data = np.genfromtxt(f"Escape Times/{model}.csv")

        plt.hist(data, bins=logbins, weights=np.ones_like(data) / len(data), histtype="step", linewidth=linewidth, cumulative=-1, label=f"{model}")

    plt.axvline(x=342, linestyle="--", color="red",
                label=r"Geminga: 342 kyr")
    plt.axvline(x=208, linestyle=":", color="red",
                label=r"J0622+3749: 208 kyr")
    plt.axvline(x=110, linestyle="-.", color="red",
                label=r"B0656+14: 110 kyr")
    
    plt.xlabel("Escape Time [kyr]")
    plt.ylabel("Probability of being inside")
    plt.xscale("log")
    plt.legend(fontsize=11)
    plt.xlim([np.min(logbins), np.max(logbins)])
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

    _, bins = np.histogram(total_data, bins=500)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))

    for i in range(len(models)):
        model = models[i]
        linewidth = linewidths[i]
        data = np.genfromtxt(f"Escape Times/{model}.csv")

        plt.hist(data, bins=logbins, weights=np.ones_like(data) / len(data), histtype="step", linewidth=linewidth, cumulative=-1, label=f"{model}")

    plt.axvline(x=342, linestyle="--", color="red",
                label=r"Geminga: 342 kyr")
    plt.axvline(x=208, linestyle=":", color="red",
                label=r"J0622+3749: 208 kyr")
    plt.axvline(x=110, linestyle="-.", color="red",
                label=r"B0656+14: 110 kyr")
    
    plt.xlabel("Escape Time [kyr]")
    plt.ylabel("Probability of being inside")
    plt.xscale("log")
    plt.legend(fontsize=11)
    plt.xlim([np.min(logbins), np.max(logbins)])
    plt.grid()
    fig.tight_layout()
    plt.savefig("Project Summary/Images/ISM escape times.pdf")
    plt.savefig("CSM_plots/ISM escape times.pdf")
    plt.show()


if __name__ == "__main__":

    plot_escape_times()
    plot_escape_times_ISM()

    1
