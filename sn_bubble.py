import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as inte
import random

np.set_printoptions(precision=1)
plt.rcParams.update({'font.size': 17})
plt.rcParams["font.family"] = "serif"


def give_random_value(
    func,
    min: float,
    max: float
) -> float:
    """
    Returns a random value using the acceptance-rejection Monte-Carlo
    method following Bestehorn, Computational Physics (2018), between
    `min` and `max`.
    -----
    Input:
    -----
    x    : integer,
           number of stars in one cluster (for the normalization factor)

    -----
    Output:
    -----
    y     : float, random value
    """
    while True:
        x1 = np.random.uniform()
        x2 = np.random.uniform()
        y = min - (min - max)*x1
        if x2 <= func(y):  # type: ignore
            return y


def pick_IMF(m: np.ndarray) -> np.ndarray:
    return m**(-2.3)


def pick_IMF_exp(m: np.ndarray) -> np.ndarray:
    return m**(-2.3)*np.exp(-0.35/m)


def test_IMF() -> None:
    M = np.linspace(2, 300)
    arr = []
    for _ in range(1000):
        arr.append(give_random_value(pick_IMF, 2, 300))

    # Uniform histogram in log x-scale
    _, bins = np.histogram(M, bins=50)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))

    fig = plt.figure()
    plt.plot(M, pick_IMF(M)/inte.quad(pick_IMF, 2, 300)
             [0], label=r"Standard IMF with index $\alpha = 2.3$")
    plt.plot(M, pick_IMF_exp(M)/inte.quad(pick_IMF_exp, 2, 300)
             [0], label="IMF from Evoli+2021")
    plt.hist(arr, histtype="step", density=True, bins=logbins)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$M$ [M$_\odot$]")
    plt.ylabel(r"PDF")
    plt.legend(fontsize=12)
    fig.tight_layout()
    plt.savefig(r"Project Summary/Images/f(m).pdf")
    plt.show()


def give_MS_time(
    M_init: np.ndarray,
    model: str = "Seo"
) -> np.ndarray:
    """Choose from "Seo" and "Zakhozhay" in yr"""

    if model == "Seo":  # Seo et al 2018
        t = 10**(7.91 - 0.77*np.log10(M_init))
    if model == "Zakhozhay":  # Zakhozhay 2013
        t = 10**(9.96 - 3.32*np.log10(M_init) + 0.63*np.log10(M_init)
                 ** 2 + 0.19*np.log10(M_init)**3 - 0.057*np.log10(M_init)**4)
    return t


def give_RSG_time(M_init: np.ndarray) -> np.ndarray:
    """Parametrization from Seo et al. 2018"""
    t = 10**(-2.76*np.log10(M_init) + 9.38)
    return t


def give_WR_time(M: np.ndarray, model="Seo") -> np.ndarray:
    """Choose from "Seo" and "Zakhozhay" """

    if model == "Seo":  # Seo et al 2018
        t = 10**(7.91 - 0.77*np.log10(M))
    if model == "Zakhozhay":  # Zakhozhay 2013
        t = 10**(9.96 - 3.32*np.log10(M) + 0.63*np.log10(M) **
                 2 + 0.19*np.log10(M)**3 - 0.057*np.log10(M)**4)
    return t


def plot_MS_time() -> None:
    M = np.linspace(2, 150, 500)  # Solar masses

    fig = plt.figure()
    plt.plot(M, give_MS_time(M, model="Seo"), label="Seo et al. 2018")
    plt.plot(M, give_MS_time(M, model="Zakhozhay"), label="Zakhozhay 2013")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$M$ [M$_\odot$]")
    plt.ylabel(r"$t_\mathrm{MS}$ [yr]")
    plt.legend()
    fig.tight_layout()
    plt.show()


def give_wind_luminosity_O(M: np.ndarray) -> np.ndarray:
    """Parametrization from Seo et al. 2018"""
    L = 10**(-3.38*np.log10(M)**2 + 14.77*np.log10(M) + 21.21)  # erg.s-1
    return L


def give_wind_luminosity_B(M: np.ndarray) -> np.ndarray:
    """Parametrization from Seo et al. 2018"""
    L = 10**(-3.38*np.log10(M)**2 + 15.02*np.log10(M) + 20.36)  # erg.s-1
    return L


def give_wind_luminosity_RSG(M: np.ndarray) -> np.ndarray:
    """Parametrization from Seo et al. 2018"""
    L = 10**(5.86*np.log10(M) + 25.79)  # erg.s-1
    return L


def give_wind_luminosity_WN(M: np.ndarray) -> np.ndarray:
    """Parametrization from Seo et al. 2018"""
    L = 10**(1.63*np.log10(M) + 35.21)  # erg.s-1
    return L


def give_wind_luminosity_WC(M: np.ndarray) -> np.ndarray:
    """Parametrization from Seo et al. 2018"""
    L = 10**(2.58*np.log10(M) + 34.63)  # erg.s-1
    return L


def give_wind_speed_O(M: np.ndarray) -> np.ndarray:
    """Parametrization from Seo et al. 2018"""
    v = 10**(3.28 + 0.08*np.log10(M))  # km.s-1
    return v


def give_wind_speed_B(M: np.ndarray) -> np.ndarray:
    """Parametrization from Seo et al. 2018"""
    v = 10**(2.85 + 0.21*np.log10(M))  # km.s-1
    return v


def give_wind_speed_RSG(M: np.ndarray) -> np.ndarray:
    """Parametrization from Seo et al. 2018"""
    v = 1.9*M**0.85  # km.s-1
    return v


def give_wind_speed_WN(M: np.ndarray) -> np.ndarray:
    """Parametrization from Seo et al. 2018"""
    v = 10**(3.49 - 0.27*np.log10(M))  # km.s-1
    return v


def give_wind_speed_WC(M: np.ndarray) -> np.ndarray:
    """Parametrization from Seo et al. 2018"""
    v = 10**(2.64 + 0.63*np.log10(M))  # km.s-1
    return v


def give_mass_loss_MS(M: np.ndarray) -> np.ndarray:
    """Parametrization from Seo et al. 2018"""
    Mdot = 10**(-3.38*np.log10(M)**2 + 14.59*np.log10(M) - 20.84)  # Msol.yr-1
    return Mdot


def give_mass_loss_RSG(M: np.ndarray) -> np.ndarray:
    """Parametrization from Seo et al. 2018"""
    Mdot = 10**(4.16*np.log10(M) - 10.27)  # Msol.yr-1
    return Mdot


def give_mass_loss_WN(M: np.ndarray) -> np.ndarray:
    """Parametrization from Seo et al. 2018"""
    Mdot = 10**(2.17*np.log10(M) - 7.27)  # Msol.yr-1
    return Mdot


def give_mass_loss_WC(M: np.ndarray) -> np.ndarray:
    """Parametrization from Seo et al. 2018"""
    Mdot = 10**(1.32*np.log10(M) - 6.15)  # Msol.yr-1
    return Mdot


def give_bubble_density(
    M: np.ndarray,
    n_ISM: float = 0.069,
    t: float = 1e6
) -> np.ndarray:
    """Bubble density in cm-3"""
    L_arr = np.array([])
    for M_ in M:
        if M_ > 16:
            L_arr = np.append(L_arr, give_wind_luminosity_O(M_))  # erg.s-1
        else:
            L_arr = np.append(L_arr, give_wind_luminosity_B(M_))
    t_ = np.minimum(give_MS_time(M), t)  # yr
    n_b = 0.01 * (L_arr/1e36)**(6/35) * (n_ISM)**(19/35) * \
        (t_/1e6)**(-22/35)  # cm-3
    return n_b


def give_bubble_radius(
    M: np.ndarray,
    n_ISM: float = 0.069,
    t: int = None
) -> np.ndarray:
    """Bubble radius in pc. Parametrization from Weaver 1988."""
    L_arr = np.array([])
    for M_ in M:
        if M_ > 16:
            L_arr = np.append(L_arr, give_wind_luminosity_O(M_))  # erg.s-1
        else:
            L_arr = np.append(L_arr, give_wind_luminosity_B(M_))
    if t is None:
        t = give_MS_time(M)  # yr
    r_b = 28 * (L_arr/1e36)**(1/5) * (n_ISM)**(-1/5) * (t/1e6)**(3/5)  # pc
    return r_b


def give_wind_radius(
    M: np.ndarray,
    n_ISM: float = 0.069,
    t: int = None
) -> np.ndarray:
    """Wind radius in pc"""
    L_arr = np.array([])
    u_arr = np.array([])
    M_dot_arr = np.array([])
    for M_ in M:
        if M_ > 16:
            L_arr = np.append(L_arr, give_wind_luminosity_O(M_))  # erg.s-1
        else:
            L_arr = np.append(L_arr, give_wind_luminosity_B(M_))
    for M_ in M:
        if M_ < 10**1.6:
            M_dot_arr = np.append(
                M_dot_arr, give_mass_loss_RSG(M_)/(np.pi*1e7))  # Msol.s-1
            u_arr = np.append(u_arr, give_wind_speed_RSG(M_)*1e5)  # cm.s-1
        else:
            WR_type = random.choice(["WC", "WN"])
            if WR_type == "WC":
                M_dot_arr = np.append(
                    M_dot_arr, give_mass_loss_WC(M_)/(np.pi*1e7))  # Msol.s-1
                u_arr = np.append(u_arr, give_wind_speed_WC(M_)*1e5)  # cm.s-1
            if WR_type == "WN":
                M_dot_arr = np.append(
                    M_dot_arr, give_mass_loss_WN(M_)/(np.pi*1e7))  # Msol.s-1
                u_arr = np.append(u_arr, give_wind_speed_WN(M_)*1e5)  # cm.s-1
    if t is None:
        t = give_MS_time(M)  # yr
    r = 1.3 * (M_dot_arr/1e-5)**(1/2) * (u_arr/1e6)**(1/2) * \
        (L_arr/1e36)**(-7/35) * (n_ISM)**(-21/70) * (t/1e6)**(14/35)  # pc
    return r


def give_SN_PDS_time(
    E: float = 2.7e50,
    n: float = 0.069,
    chi: float = 1
) -> float:
    """Formula and parameters from Cioffi et al. 1988.
    Returns the Pressure Driven Snowplough time in yr"""
    t = 3.61e4 * (E/1e51)**(3/14) * (chi)**(-5/14) * (n)**(-4/7)  # yr
    return t/np.exp(1)


def give_SN_PDS_radius(
    t: np.ndarray,
    E: float = 2.7e50,
    n: float = 0.069,
    chi: float = 1
) -> np.ndarray:
    """Formula and parameters from Cioffi et al. 1988"""
    t_PDS = give_SN_PDS_time(E, n, chi)  # yr
    radius_beginning_PDS = 14.0 * \
        (E/1e51)**(2/7) * (n)**(-3/7) * (chi)**(-1/7)  # pc
    r = radius_beginning_PDS * (4/3*t/t_PDS - 1/3)**(3/10)  # pc
    return r


def give_SN_ST_radius(
    t: float = 100e3,
    E: float = 2.7e50,
    n: float = 0.069
) -> float:
    """
    Naive interpretation of the Sedov-Taylor phase of the SNR,
    time in yr"""
    # ST phase
    r = 0.3 * (E/1e51)**(1/5) * (n/1)**(-1/5) * (t/1)**(2/5)  # pc
    return r  # pc


def give_ISM_sound_speed(T: float = 1e2) -> float:
    """Temperature of the ISM may vary depending on the region
    (dust, cloud, gas, etc)"""
    return np.sqrt(gamma*kB*T/(mu_p*m_p))  # cm/s


def give_SN_merge_time(
    c_s: float = 1e6,
    E: float = 2.7e50,
    n: float = 0.069,
    chi: float = 1
) -> float:
    """Parametrization from Cioffi et al. 1988"""
    t_PDS = give_SN_PDS_time(E, n, chi)  # yr
    beta = 2
    t = 153 * t_PDS * ((E/1e51)**(1/14) * n**(1/7) * chi **
                       (3/14) / (beta * c_s/1e6))**(10/7)
    return t  # yr


def give_SN_radius(
    t: np.ndarray = 100e3,
    E: float = 2.7e50,
    n: float = 0.069,
    chi: float = 1,
    Mej: float = 1
) -> float:
    """Decides the phase of the SNR and gives the appropriate radius
    in pc. Requires a time in yr"""
    # Determining the SNR stage
    t_PDS = give_SN_PDS_time(E, n, chi)
    t_MCS = give_SN_MCS_time(E, n, chi, Mej)
    t_max = give_SN_merge_time(give_ISM_sound_speed())
    if t < t_PDS:
        r = give_SN_ST_radius(t, E, n)
    elif t < t_MCS:
        r = give_SN_PDS_radius(t, E, n, chi)
    elif t < t_max:
        r = give_SN_MCS_radius(t, E, n, chi, Mej)
    if t > t_max:
        r = 0
    return r


def give_SN_MCS_time(
    E: float = 2.7e50,
    n: float = 0.069,
    chi: float = 1, # metallicity
    Mej: float = 1 # Msol
) -> float:
    t_PDS = give_SN_PDS_time(E, n, chi)
    t = 61 * t_PDS * (10*(E/1e51/Mej)**1/2)**3 * \
        chi**(-3/14) * n**(-3/7) * (E/1e51)**(-3/14)  # yr
    return t  # yr


def give_SN_MCS_radius(
    t: np.ndarray = 100e3,
    E: float = 2.7e50,
    n: float = 0.069,
    chi: float = 1,
    Mej: float = 1
) -> np.ndarray:
    t_PDS = give_SN_PDS_time(E, n, chi)
    t_MCS = give_SN_MCS_time(E, n, chi, Mej)
    r_PDS = give_SN_PDS_radius(E, n, chi)
    r_MCS = (4.66*t_MCS/t_PDS*(1-0.939*(t_MCS/t_PDS) **
             (-0.17) - 0.153*(t_MCS/t_PDS)**(-1)))**(1/4)
    r = r_PDS * (4.66*(t/t_PDS - t_MCS/t_PDS) *
                 (1-0.779*(t_MCS/t_PDS)**(-0.17)) + r_MCS**4)**(1/4)
    return r


def plot_SN_radius(E: float = 1e51, n: float = 1, chi: float = 2) -> None:

    t_max = give_SN_merge_time(give_ISM_sound_speed(), E, n, chi)
    t_arr = np.logspace(1, 7, 50)  # yr
    r_arr = np.array([give_SN_radius(t_, E, n, chi) for t_ in t_arr])
    t_PDS = give_SN_PDS_time(E, n, chi)

    fig = plt.figure()
    plt.plot(t_arr, r_arr, linestyle="-", color="black",
             label="From Cioffi et al. 1988")
    plt.axvline(x=t_PDS, linestyle="--", color="red",
                label=r"$t_\mathrm{PDS} =$"f"${t_PDS/1e3:.1f}$"r" kyr")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(fontsize=12)
    plt.grid()
    plt.xlabel(r"$t$ [yr]")
    plt.ylabel(r"$R_\mathrm{s}(t)$ [pc]")
    fig.tight_layout()
    plt.savefig("Project Summary/Images/R_SN(t).pdf")
    plt.show()


def plot_wind_luminosity() -> None:
    M = np.linspace(2, 150, 500)  # Solar masses
    M_O = np.linspace(16, 150, 500)
    M_B = np.linspace(2, 16, 500)

    fig = plt.figure()
    plt.plot(M_O, give_wind_luminosity_O(M_O), label="O")
    plt.plot(M_B, give_wind_luminosity_B(M_B), label="B")
    plt.plot(M, give_wind_luminosity_RSG(M), label="RSG")
    plt.plot(M, give_wind_luminosity_WN(M), label="WN")
    plt.plot(M, give_wind_luminosity_WC(M), label="WC")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.xlabel(r"$M$ [M$_\odot$]")
    plt.ylabel(r"$L_\mathrm{w}$ [erg$\cdot$s$^{-1}$]")
    fig.tight_layout()
    plt.show()


def plot_wind_speed() -> None:
    M = np.linspace(2, 150, 500)  # Solar masses
    M_O = np.linspace(16, 150, 500)
    M_B = np.linspace(2, 16, 500)

    fig = plt.figure()
    plt.plot(M_O, give_wind_speed_O(M_O), label="O")
    plt.plot(M_B, give_wind_speed_B(M_B), label="B")
    plt.plot(M, give_wind_speed_RSG(M), label="RSG")
    plt.plot(M, give_wind_speed_WN(M), label="WN")
    plt.plot(M, give_wind_speed_WC(M), label="WC")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.xlabel(r"$M$ [M$_\odot$]")
    plt.ylabel(r"$v_\mathrm{w}$ [km$\cdot$s$^{-1}$]")
    fig.tight_layout()
    plt.show()


def plot_mass_loss() -> None:
    M = np.linspace(2, 150, 500)  # Solar masses

    fig = plt.figure()
    plt.plot(M, give_mass_loss_MS(M), label="MS")
    plt.plot(M, give_mass_loss_RSG(M), label="RSG")
    plt.plot(M, give_mass_loss_WN(M), label="WN")
    plt.plot(M, give_mass_loss_WC(M), label="WC")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.xlabel(r"$M$ [M$_\odot$]")
    plt.ylabel(r"$\dot{M}$ [M$_\odot\cdot$yr$^{-1}$]")
    fig.tight_layout()
    plt.show()


def plot_bubble_density() -> None:
    M = np.linspace(2, 150, 500)  # Solar masses

    fig = plt.figure()
    plt.plot(M, give_bubble_density(M, t=1e6),
             label=r"$t = {:.2f}$ Myr".format(1))
    plt.plot(M, give_bubble_density(M, t=5e6),
             label=r"$t = {:.2f}$ Myr".format(5))
    plt.plot(M, give_bubble_density(M, t=1e7),
             label=r"$t = {:.2f}$ Myr".format(10))
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.xlabel(r"$M$ [M$_\odot$]")
    plt.ylabel(r"$n_\mathrm{b}$ [cm$^{-3}$]")
    fig.tight_layout()
    plt.show()


def plot_bubble_radius() -> None:
    M = np.linspace(2, 150, 500)  # Solar masses

    fig = plt.figure()
    plt.plot(M, give_bubble_radius(M, t=1e6), color="red", linestyle="-",
             label=r"$t = {:.2f}$ Myr".format(1)+", r$_\mathrm{b}$")
    plt.plot(M, give_wind_radius(M, t=1e6), color="red", linestyle="--",
             label=r"$t = {:.2f}$ Myr".format(1)+", r$_\mathrm{w}$")
    plt.plot(M, give_bubble_radius(M, t=5e6), color="blue", linestyle="-",
             label=r"$t = {:.2f}$ Myr".format(5)+", r$_\mathrm{b}$")
    plt.plot(M, give_wind_radius(M, t=5e6), color="blue", linestyle="--",
             label=r"$t = {:.2f}$ Myr".format(5)+", r$_\mathrm{w}$")
    plt.plot(M, give_bubble_radius(M, t=1e7), color="green", linestyle="-",
             label=r"$t = {:.2f}$ Myr".format(10)+", r$_\mathrm{b}$")
    plt.plot(M, give_wind_radius(M, t=1e7), color="green", linestyle="--",
             label=r"$t = {:.2f}$ Myr".format(10)+", r$_\mathrm{w}$")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.xlabel(r"$M$ [M$_\odot$]")
    plt.ylabel(r"$r$ [pc]")
    fig.tight_layout()
    plt.show()


class Stars:
    def __init__(self, N: int, M_array: np.ndarray = None, t: float = 1e7):
        self.N = int(N)
        self.t = t
        # Define the initial masses and characteristic times associated
        # to the different phases of a star
        if M_array is None:
            self.init_mass = np.array(
                [give_random_value(pick_IMF, 2, 150) for _ in range(self.N)])
        else:
            self.init_mass = M_array
            self.N = len(self.init_mass)
        self.time_MS = give_MS_time(self.init_mass, model="Seo")
        self.time_RSG = give_RSG_time(self.init_mass)
        self.time_WR = give_WR_time(self.init_mass)
        self.type = np.array(
            ["MSO" if self.init_mass[i] > 16 else "MSB"
             for i in range(self.N)])

        # Give the stage in the life of the stars at input time t
        for i in range(self.N):
            if t > self.time_MS[i]:
                if self.init_mass[i] < 10**1.6:
                    self.type[i] = "RSG"
                    if t > self.time_RSG[i]:
                        self.type[i] = "SN"
                else:
                    self.type[i] = random.choice(["WC", "WN"])
                    if t > self.time_WR[i]:
                        self.type[i] = "SN"


gamma = 5/3  # adiabatic coefficient for monoatomic gas
m_p = 1.6726e-24  # g
mu_p = 1.4  # molecular fraction in the ISM
kB = 1.3807e-16  # erg/K Boltzmann constant

if __name__ == "__main__":
    # TESTS AND PLOTS

    # test_IMF()
    # plot_MS_time()
    # plot_wind_luminosity()
    # plot_wind_speed()
    # plot_mass_loss()
    # plot_bubble_density()
    # plot_bubble_radius()
    plot_SN_radius()

    # stars = Stars(1000, t=1e7)
    # print(stars.type)

    1
