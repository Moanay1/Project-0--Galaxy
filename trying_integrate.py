import numpy as np
import scipy.integrate as inte
import matplotlib.pyplot as plt

x = np.linspace(-100, 100, 100)
y = x

integral = np.array([inte.simpson(y[:i], x[:i]) for i in range(2, len(x))])

fig = plt.figure()

plt.plot(x[1:-1], integral+5000, label="Integration")
plt.plot(x, x**2/2, label="Analytical")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
fig.tight_layout()
plt.show()