import numpy as np
import scipy.integrate as inte
import matplotlib.pyplot as plt
import zeroordergalaxy as galaxy


def make_lognormal(zz: np.ndarray, mu: float, sigma: float) -> np.ndarray:
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
    return 1/np.sqrt(2*np.pi*normal_std**2)\
            * np.exp(-(np.log(zz)-normal_mean)**2/(2*normal_std**2))


def plot_convergence_available_energy():

    # Convergence
    steps_arr = np.int32(np.logspace(1, 7, 7))

    fig = plt.figure()
    plt.plot(steps_arr, [available_energy(steps=steps_) for steps_ in steps_arr], linestyle = "", marker ="o")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of integration steps")
    plt.ylabel("Total available energy [erg]")
    plt.grid()
    fig.tight_layout()
    plt.show()


def initial_frequency(P0:float = 1 # s
                      ):
    return 2*np.pi/(P0) # s-1


def moment_inertia(Ms:float = 1.4, # Msol
                   Rs:float = 10 # km
                   ):
    return 2/5*(Ms*Msol)*(Rs*km)**2 # g/cm2


# def spindown_time(P0:float = 100, # ms
#                   dP0:float = 1e-15 # s/s
#                   ):
#     return (P0*ms)/(2*dP0) # s


def spindown_time(P0:float = 1, # s
                  Bs:float = 10**(12.6), # G
                  Rs:float = 10 # km
                  ):
    global I
    return 3*c**3 * I / (Bs**2 * (Rs*km)**6 * initial_frequency(P0)**2) # s


def luminosity(t:float = 1, # yr
               P0:float = 1, # s
               ):
    global I
    Omega0 = initial_frequency(P0) # s-1
    tau0 = spindown_time(P0) # s
    return 1/2 * I * Omega0**2/tau0 * 1/(1 + t*yr/tau0)**2 # erg/s


def bowshock_time(E_SN:float = 1e51, # erg
                  n_ISM:float = 1, # cm-3
                  v_k:float = 280 # km/s
                  ):
    return 56e3*(E_SN/1e51)**(1/3) * (n_ISM)**(-1/3) * (v_k/280)**(-5/3) # yr


def initial_period_computed(P:float = 1, # s
                            t:float = 0, # yr
                            ):
    t_spindown = spindown_time(P0=P)
    return P * (1 + t*yr / t_spindown)**(-1/2) # s


def plot_initial_period_computed():

    t_arr = np.logspace(0, 9, 1000) # yr

    P = 1 # s
    P0_arr = initial_period_computed(t = t_arr)

    fig = plt.figure()
    plt.plot(t_arr, P0_arr, label=f"Observed period now: {P}"r"s")
    plt.xscale("log")
    plt.xlabel(r"Pulsar Age [yr]")
    plt.ylabel(r"Initial pulsar period [s]")
    plt.grid()
    plt.legend()
    fig.tight_layout()
    plt.show()


def available_energy(t_init:float = 0, # yr
                     P0:float = 1, # s
                    ):
    return inte.quad(lambda x:luminosity(t=x, P0=P0), t_init, np.inf)[0] # erg


def SimpsonIntegration(f, a:float, b:float, steps:int) -> float :

    # making the number of steps even so that the final number of points is odd
    if steps % 2 != 0:
        steps += 1

    x_axis = np.linspace(a, b, steps)
    h = x_axis[1] - x_axis[0]
    I = f(a) + f(b)

    I += 2*np.sum(f(x_axis[:steps-2:2])) + 4*np.sum(f(x_axis[1:steps-1:2]))
    
    I = h/3 * I
    return I


c = 3e10 #cm/s
Msol = 1.989e33 # g
km = 1e5 # cm
ms = 1e-3 # s
yr = np.pi*1e7 # s
I = moment_inertia()


def plot_available_energy_over_time(P0=100):

    print(f"bowshock time = {bowshock_time():.0f} yr")
    print(f"spindown time = {spindown_time(P0=P0)/yr:.0f} yr")
    energy_frac = available_energy(t_init=bowshock_time(), P0=P0)/available_energy(P0=P0)
    print(f"energy fraction = {energy_frac*100:.0f} %")

    t_arr = np.logspace(1, 8, 500) # yr

    available_energy_arr = np.array([available_energy(t_init=t, P0=P0)/available_energy(P0=P0)
                                     for t in t_arr])*100 # %


    fig = plt.figure()

    # plt.plot(t_arr, luminosity(t_arr))
    plt.plot(t_arr, available_energy_arr,
             label=r"$E(t_\mathrm{BS})/E(0)=$"f"{energy_frac*100:.0f} %")
    plt.axvline(x=bowshock_time(), c="orange", linestyle="--",
                label="Bowshock time")
    plt.axvline(x=spindown_time(P0=P0)/yr, c="red", linestyle="-.",
                label="Spindown time")
    plt.xscale("log")
    # plt.yscale("log")
    plt.xlabel(r"Pulsar age [yr]")
    plt.ylabel(r"Energy available [%]")
    plt.grid()
    plt.legend()
    fig.tight_layout()
    plt.savefig("Project Summary/Images/energybudget(t_age).pdf")
    plt.show()


def plot_efficiency_distribution_fixed_period():

    number_pulsars = 10000

    E_SNs = galaxy.give_E_SN(number_pulsars)
    n_ISMs = galaxy.give_n_ISM(number_pulsars)
    v_ks = np.array([galaxy.give_kick_velocity()
                     for _ in range(number_pulsars)])

    bowshock_times = bowshock_time(E_SNs, n_ISMs, v_ks)

    P0s = [0.1, 0.3] # s

    fig = plt.figure()

    for P0 in P0s:
        efficiencies = np.array([available_energy(t_init=time, P0=P0)/
                                 available_energy(P0=P0) for time in bowshock_times])*100
        
        efficiencies = efficiencies[efficiencies > 0]
        
        plt.hist(efficiencies, histtype="step", bins=50,
                 label=r"$P_0 =$"f"{P0*1e3:.0f}"r"ms")
    
    plt.xlabel("Injection efficiency [%]")
    plt.ylabel("Pulsars")
    plt.grid()
    plt.legend()
    fig.tight_layout()
    plt.savefig("Project Summary/Images/f(efficiencies)_fixed_periods.pdf")
    plt.show()


def plot_efficiency_distribution():

    number_pulsars = 10000

    E_SNs = galaxy.give_E_SN(number_pulsars)
    n_ISMs = galaxy.give_n_ISM(number_pulsars)
    v_ks = np.array([galaxy.give_kick_velocity()
                     for _ in range(number_pulsars)])

    bowshock_times = bowshock_time(E_SNs, n_ISMs, v_ks)

    Ps = galaxy.give_P_PSR(number_pulsars)
    P0s = initial_period_computed(P=Ps, t=bowshock_times)

    fig = plt.figure()

    efficiencies = np.array([available_energy(t_init=bowshock_times[i],
                                              P0=P0s[i]) /
                             available_energy(P0=P0s[i])
                             for i in range(number_pulsars)])*100

    efficiencies = efficiencies[efficiencies > 0]

    median = np.median(efficiencies)

    plt.hist(efficiencies, histtype="step", bins=100, density=True)
    plt.axvline(x=median, color="red", label=f"Median = {median:.2f} %")
    
    plt.xlabel("Injection efficiency [%]")
    plt.ylabel("Pulsars proportion per bin")
    plt.grid()
    plt.legend()
    fig.tight_layout()
    plt.savefig("Project Summary/Images/f(efficiencies).pdf")
    plt.show()


if __name__ == "__main__":

    # plot_initial_period_computed()

    # plot_convergence_available_energy() # Only use when Simpsons Rule in
    #                                     # available energy
    # plot_available_energy_over_time()
    # plot_efficiency_distribution_fixed_period()
    plot_efficiency_distribution()

    1


