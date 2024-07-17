import numpy as np
import matplotlib.pyplot as plt
import sn_bubble as sn
import cgs

def pressure_WR(r, M=50):

    mass_loss = sn.give_mass_loss_WC(M)*cgs.sun_mass/cgs.year
    wind_speed = sn.give_wind_speed_WC(M)*cgs.km

    return mass_loss*wind_speed/(4*np.pi*r**2)

def thermal_pressure(r, M=50, T=80000):

    mass_loss = sn.give_mass_loss_MS(M)*cgs.sun_mass/cgs.year
    wind_speed = sn.give_wind_speed_O(M)*cgs.km

    a = np.zeros(len(r)) + 0.1 # mass_loss/(4*np.pi*cgs.proton_mass*wind_speed*r**2)

    return a*cgs.k_boltzmann*T

r = np.geomspace(0.01*cgs.pc, 100*cgs.pc)

fig = plt.figure()

plt.plot(r/cgs.pc, pressure_WR(r), label="wind")
plt.plot(r/cgs.pc, thermal_pressure(r), label="thermal")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Radius [pc]")
plt.ylabel("Pressure [Ba]")
plt.grid()
plt.legend()
fig.tight_layout()
plt.show()
