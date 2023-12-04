import numpy as np
import matplotlib.pyplot as plt

t = 100e3

data = np.genfromtxt("Pulsar_characteristics_{}_kyr_0.csv".format(t/1e3), delimiter=",", skip_header=1)

x = data[:,0]
y = data[:,1]
z = data[:,2]

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(projection='3d')

ax.scatter(x, y, z, s=0.5)
ax.plot(8.5, 0, linewidth=0, marker=".", color="red")
ax.set_xlabel(r"x [kpc]")
ax.set_ylabel(r"y [kpc]")
ax.set_zlabel(r"z [kpc]")
ax.set_zlim(-15,15)
fig.tight_layout()
plt.show()
