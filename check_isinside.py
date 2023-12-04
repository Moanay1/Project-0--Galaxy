import numpy as np
import matplotlib.pyplot as plt

t = 100e3

fig, ax = plt.subplots(2, 1)

for i in [0, 1]:

    is_inside = np.genfromtxt("Pulsar_characteristics_{}_kyr_{}.csv".format(t/1e3, i), usecols=6, dtype=bool, skip_header=1)

    print(is_inside)

    ax[i].plot(range(len(is_inside)), is_inside, marker = ".", markersize = 1, linestyle = "", label = r"{:.2f}% inside SNR".format(np.count_nonzero(is_inside)/len(is_inside)*100))
    ax[i].legend()
    ax[i].set_ylabel("{}".format(i))

fig.tight_layout()
plt.savefig(r"Project Summary/Images/isinside_comparison.pdf")
plt.show()