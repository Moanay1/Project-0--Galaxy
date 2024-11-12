import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({'font.size': 17})
plt.rcParams["font.family"] = "serif"


def plot_escape_times():

    models = ["CSM bubble", "Superbubble"]
    colors = ["blue", "red"]
    total_data = np.array([])

    fig = plt.figure()

    for model in models:
        data = np.genfromtxt(f"Escape Times/{model}.csv")
        total_data = np.append(total_data, data)

    _, bins = np.histogram(total_data, bins=500)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))

    for i in range(len(models)):
        model = models[i]
        color = colors[i]
        data = np.genfromtxt(f"Escape Times/{model}.csv")

        plt.hist(data, bins=logbins, weights=np.ones_like(data) / len(data), color=color, histtype="step", linewidth=3, cumulative=-1, label=f"{model}", zorder=5)

    plt.axvline(x=110, linewidth=5, alpha=0.75,  color="pink",
                label=r"B0656+14: 110 kyr")
    plt.axvline(x=208, linewidth=5, alpha=0.75,  color="violet",
                label=r"J0622+3749: 208 kyr")
    plt.axvline(x=342, linewidth=5, alpha=0.75, color="purple",
                label=r"Geminga: 342 kyr")
    
    plt.xlabel("Pulsar age [kyr]")
    plt.ylabel("Probability of being inside")
    plt.xscale("log")
    plt.legend(fontsize=11)
    plt.xlim([np.min(logbins), np.max(logbins)])
    #plt.grid()
    fig.tight_layout()
    plt.savefig("Project Summary/Images/all escape times.pdf")
    plt.savefig("CSM_plots/all escape times.pdf")
    plt.show()


def plot_escape_times40():

    models = ["CSM", "CSM40", "Superbubble", "Superbubble40"]
    legends = ["CSM 120 M$_\odot$", "CSM 40 M$_\odot$", "SB 120 M$_\odot$", "SB 40 M$_\odot$"]
    colors = ["blue", "green", "red", "orange"]
    total_data = np.array([])

    fig = plt.figure()

    for model in models:
        data = np.genfromtxt(f"Escape Times/{model}.csv")
        total_data = np.append(total_data, data)

    _, bins = np.histogram(total_data, bins=500)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))

    for i in range(len(models)):
        model = models[i]
        color = colors[i]
        legend = legends[i]
        data = np.genfromtxt(f"Escape Times/{model}.csv")

        plt.hist(data, bins=logbins, weights=np.ones_like(data) / len(data), color=color, histtype="step", linewidth=3, cumulative=-1, label=legend, zorder=5)

    plt.axvline(x=110, linewidth=5, alpha=0.75,  color="pink",
                label=r"B0656+14: 110 kyr")
    plt.axvline(x=208, linewidth=5, alpha=0.75,  color="violet",
                label=r"J0622+3749: 208 kyr")
    plt.axvline(x=342, linewidth=5, alpha=0.75, color="purple",
                label=r"Geminga: 342 kyr")
    
    plt.xlabel("Pulsar age [kyr]")
    plt.ylabel("Probability of being inside")
    plt.xscale("log")
    plt.legend(fontsize=11)
    plt.xlim([np.min(logbins), np.max(logbins)])
    #plt.grid()
    fig.tight_layout()
    plt.savefig("Project Summary/Images/escape_times_40.pdf")
    plt.savefig("CSM_plots/escape_times_40.pdf")
    plt.show()


def plot_escape_times_SNR():

    models = ["CSM bubble", "CSM SNR", "Superbubble bubble", "Superbubble SNR"]
    legends = ["CSM bubble", "CSM SNR", "SB bubble", "SB SNR"]
    colors = ["blue", "green", "red", "orange"]
    total_data = np.array([])

    fig = plt.figure()

    for model in models:
        data = np.genfromtxt(f"Escape Times/{model} 40.csv")
        total_data = np.append(total_data, data)

    _, bins = np.histogram(total_data, bins=500)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))

    for i in range(len(models)):
        model = models[i]
        color = colors[i]
        legend = legends[i]
        data = np.genfromtxt(f"Escape Times/{model} 40.csv")

        plt.hist(data, bins=logbins, weights=np.ones_like(data) / len(data), color=color, histtype="step", linewidth=3, cumulative=-1, label=legend, zorder=5)

    plt.axvline(x=110, linewidth=5, alpha=0.75,  color="pink",
                label=r"B0656+14: 110 kyr")
    plt.axvline(x=208, linewidth=5, alpha=0.75,  color="violet",
                label=r"J0622+3749: 208 kyr")
    plt.axvline(x=342, linewidth=5, alpha=0.75, color="purple",
                label=r"Geminga: 342 kyr")
    
    plt.xlabel("Pulsar age [kyr]")
    plt.ylabel("Probability of being inside")
    plt.xscale("log")
    plt.legend(fontsize=11)
    plt.xlim([np.min(logbins), np.max(logbins)])
    #plt.grid()
    fig.tight_layout()
    plt.savefig("Project Summary/Images/escape_times_SNR.pdf")
    plt.savefig("CSM_plots/escape_times_SNR.pdf")
    plt.show()


def plot_escape_times_ISM():

    models = ["ISM"]
    total_data = np.array([])

    fig = plt.figure()

    for model in models:
        data = np.genfromtxt(f"Escape Times/{model}.csv")
        total_data = np.append(total_data, data)

    _, bins = np.histogram(total_data, bins=500)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))

    for i in range(len(models)):
        model = models[i]
        data = np.genfromtxt(f"Escape Times/{model}.csv")

        plt.hist(data, bins=logbins, weights=np.ones_like(data) / len(data), color="black", histtype="step", linewidth=3, cumulative=-1, label=f"{model}", zorder=5)

    plt.axvline(x=110, linewidth=5, alpha=0.75,  color="pink",
                label=r"B0656+14: 110 kyr")
    plt.axvline(x=208, linewidth=5, alpha=0.75,  color="violet",
                label=r"J0622+3749: 208 kyr")
    plt.axvline(x=342, linewidth=5, alpha=0.75, color="purple",
                label=r"Geminga: 342 kyr")
    
    plt.xlabel("Pulsar age [kyr]")
    plt.ylabel("Probability of being inside")
    plt.xscale("log")
    plt.legend(fontsize=11)
    plt.xlim([np.min(logbins), np.max(logbins)])
    #plt.grid()
    fig.tight_layout()
    plt.savefig("Project Summary/Images/ISM escape times.pdf")
    plt.savefig("CSM_plots/ISM escape times.pdf")
    plt.show()


def plot_escape_times_galaxy():

    models = ["Galactic population"]
    total_data = np.array([])

    fig = plt.figure()

    for model in models:
        data = np.genfromtxt(f"Escape Times/{model}.csv")
        total_data = np.append(total_data, data)

    _, bins = np.histogram(total_data, bins=500)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))

    for i in range(len(models)):
        model = models[i]
        data = np.genfromtxt(f"Escape Times/{model}.csv")

        plt.hist(data, bins=logbins, weights=np.ones_like(data) / len(data), color="black", histtype="step", linewidth=3, cumulative=-1, label=f"{model}", zorder=5)

    plt.axvline(x=110, linewidth=5, alpha=0.75,  color="pink",
                label=r"B0656+14: 110 kyr")
    plt.axvline(x=208, linewidth=5, alpha=0.75,  color="violet",
                label=r"J0622+3749: 208 kyr")
    plt.axvline(x=342, linewidth=5, alpha=0.75, color="purple",
                label=r"Geminga: 342 kyr")
    
    plt.xlabel("Pulsar age [kyr]")
    plt.ylabel("Probability of being inside")
    plt.xscale("log")
    plt.legend(fontsize=11)
    plt.xlim([np.min(logbins), np.max(logbins)])
    #plt.grid()
    fig.tight_layout()
    plt.savefig("Project Summary/Images/Galactic escape times.pdf")
    plt.savefig("CSM_plots/Galactic escape times.pdf")
    plt.show()


if __name__ == "__main__":

    # plot_escape_times()
    # plot_escape_times40()
    plot_escape_times_SNR()
    # plot_escape_times_ISM()
    # plot_escape_times_galaxy()

    1
