import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.integrate as inte
import scipy.optimize as opt
from scipy.stats import lognorm
import random
import shock_speed as shock
from tqdm import tqdm
import cgs

np.set_printoptions(precision=3)
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


def give_MS_time(
    M_init: np.ndarray,
    model: str = "Zakhozhay"
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


def give_WR_time(M_init: np.ndarray) -> np.ndarray:

    t = 10**(5.51 - 0.042*np.log10(M_init))

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


def give_wind_luminosity_type(M: np.ndarray) -> np.ndarray:

    luminosity = np.array([])

    if type(M) in [float]:
        if M < 16:
            luminosity = np.append(luminosity, give_wind_luminosity_B(M))
        else:
            luminosity = np.append(luminosity, give_wind_luminosity_O(M))
        return luminosity

    else:
        for M_ in M:
            if M_ < 16:
                luminosity = np.append(luminosity, give_wind_luminosity_B(M_))
            else:
                luminosity = np.append(luminosity, give_wind_luminosity_O(M_))
        
        return luminosity
    


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


def give_wind_speed_type(M: np.ndarray, typee: str) -> np.ndarray:

    if typee == "B":
        return give_wind_speed_B(M)
    elif typee == "O":
        return give_wind_speed_O(M)
    elif typee == "RSG":
        return give_wind_speed_RSG(M)
    elif typee == "WN":
        return give_wind_speed_WN(M)
    elif typee == "WC":
        return give_wind_speed_WC(M)
    else:
        raise ValueError("Star type should be 'B', 'O', 'RSG, 'WN', 'WC'")


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


def give_mass_loss_type(M: np.ndarray, typee: str) -> np.ndarray:
    if typee == "MS":
        return give_mass_loss_MS(M)
    elif typee == "RSG":
        return give_mass_loss_RSG(M)
    elif typee == "WN":
        return give_mass_loss_WN(M)
    elif typee == "WC":
        return give_mass_loss_WC(M)
    else:
        raise ValueError("Star type should be 'MS', 'RSG, 'WN', 'WC'")


def give_bubble_density(
    M: np.ndarray,
    n_ISM: float = 1,
    t: float = 1e8
) -> np.ndarray:
    """Bubble density in cm-3"""
    L_arr = np.array([])
    for M_ in M:
        if M_ > 16:
            L_arr = np.append(L_arr, give_wind_luminosity_O(M_))  # erg.s-1
        else:
            L_arr = np.append(L_arr, give_wind_luminosity_B(M_))
    t_ = give_MS_time(M)  # yr
    n_b = 0.01 * (L_arr/1e36)**(6/35) * (n_ISM)**(19/35) * \
        (t_/1e6)**(-22/35)  # cm-3
    return n_b


def give_bubble_density_type(M, n_ISM=1):
    """Bubble number density from MacLow and McCray 1988"""

    L = give_wind_luminosity_type(M)
    
    t_MS = give_MS_time(M)  # yr
    n_b = 4e-3 * (0.22*L/1e38)**(6/35) * (n_ISM)**(19/35) * \
        (t_MS/1e7)**(-22/35)  # cm-3
    return n_b
    


def give_bubble_radius(
    M: np.ndarray,
    n_ISM: float = 1,
    t: int = None
) -> np.ndarray:
    """Bubble radius in pc. Parametrization from Weaver 1988."""
    L_arr = np.array([])
    for M_ in M:
        if M_ > 16:
            wind_speed = give_wind_speed_O(M_) * cgs.km
        else:
            wind_speed = give_wind_speed_B(M_) * cgs.km

        mass_loss = give_mass_loss_MS(M_) * cgs.sun_mass / cgs.year
        L_arr = np.append(L_arr, give_wind_kinetic_power(wind_speed, mass_loss))  # erg.s-1
    t_MS = give_MS_time(M)  # yr
    r_b = 21 * (0.22/0.22)**(1/5) * (L_arr/1e36)**(1/5) * (n_ISM)**(-1/5) * (t_MS/1e6)**(3/5)  # pc
    return r_b


def give_wind_kinetic_power(wind_speed, mass_loss):
    return mass_loss * wind_speed**2 / 2


def give_bubble_radius_type(M, n_ISM=1, typee="RSG"):

    wind_speed = give_wind_speed_type(M, typee=typee) * cgs.km
    mass_loss = give_mass_loss_MS(M) * cgs.sun_mass / cgs.year

    L = give_wind_kinetic_power(wind_speed, mass_loss)
    
    t_MS = give_MS_time(M)  # yr
    r_b = 21 * (0.22/0.22)**(1/5) * (L/1e36)**(1/5) * (n_ISM)**(-1/5) * (t_MS/1e6)**(3/5)  # pc
    return r_b


def give_wind_radius(
    M: np.ndarray,
    n_ISM: float = 1,
    t: int = None
) -> np.ndarray:
    """Wind radius in pc"""
    L_arr = np.array([])
    u_arr = np.array([])
    M_dot_arr = np.array([])
    type_arr = np.array([])
    t_arr = np.array([])

    for M_ in M:
        if M_ > 16:
            L_arr = np.append(L_arr, give_wind_luminosity_O(M_))  # erg.s-1
            type_arr = np.append(type_arr, "O")
        else:
            L_arr = np.append(L_arr, give_wind_luminosity_B(M_))
            type_arr = np.append(type_arr, "B")
    for M_ in M:
        if M_ < 40:
            M_dot_arr = np.append(
                M_dot_arr, give_mass_loss_RSG(M_))  # Msol.yr-1
            u_arr = np.append(u_arr, give_wind_speed_RSG(M_)*1e5)  # cm.s-1
            type_arr = np.append(type_arr, "RSG")
            t_arr = np.append(t_arr, give_RSG_time(M_))
        else:
            WR_type = random.choice(["WC", "WN"])
            t_arr = np.append(t_arr, give_WR_time(M_))
            if WR_type == "WC":
                M_dot_arr = np.append(
                    M_dot_arr, give_mass_loss_WC(M_))  # Msol.yr-1
                u_arr = np.append(u_arr, give_wind_speed_WC(M_)*1e5)  # cm.s-1
                type_arr = np.append(type_arr, WR_type)
            if WR_type == "WN":
                M_dot_arr = np.append(
                    M_dot_arr, give_mass_loss_WN(M_))  # Msol.s-1
                u_arr = np.append(u_arr, give_wind_speed_WN(M_)*1e5)  # cm.s-1
                type_arr = np.append(type_arr, WR_type)

    t_MS = give_MS_time(M)  # yr

    # t = np.minimum(np.array([t for _ in range(len(t_MS))]), t_MS)
    
    r = 1.3 * (M_dot_arr/1e-5)**(1/2) * (u_arr/1e6)**(1/2) * \
       (L_arr/1e36)**(-7/35) * (n_ISM)**(-21/70) * (t_MS/1e6)**(14/35)  # pc
    
    return r, type_arr


def give_wind_radius_type(M, typee, n_ISM=1):

    mass_loss = give_mass_loss_MS(M)
    wind_speed = give_wind_speed_type(M, typee)
    luminosity = give_wind_luminosity_type(M)

    t_MS = give_MS_time(M)  # yr

    r = 1.3 * (mass_loss/1e-5)**(1/2) * (wind_speed/1e6)**(1/2) * \
       (luminosity/1e36)**(-7/35) * (n_ISM)**(-21/70) * (t_MS/1e6)**(14/35)  # pc
    
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


def give_SN_PDS_time2(
    E: float = 2.7e50,
    n: float = 0.069,
) -> float:
    """Formula and parameters from Vink 2012.
    Returns the Pressure Driven Snowplough time in yr"""
    t = 44600 * (E/1e51)**(1/3) * (n)**(-1/3)   # yr
    return t


def give_SN_PDS_radius(
    t: np.ndarray,
    E: float = 2.7e50,
    n: float = 0.069,
    chi: float = 1,
    s:int = 0
) -> np.ndarray:
    """Formula and parameters from Cioffi et al. 1988"""
    t_PDS = give_SN_PDS_time(E, n, chi)  # yr
    # Formula from Cioffi 1988 but that has a tiny shift with ST
    # radius_beginning_PDS = 14.0 * \
    #     (E/1e51)**(2/7) * (n)**(-3/7) * (chi)**(-1/7)  # pc
    
    # Artificially stick the two parts together
    time_beginning_PDS = give_SN_PDS_time(E, n, chi)
    radius_beginning_PDS = give_SN_ST_radius(time_beginning_PDS, E, n, s)


    r = radius_beginning_PDS * (4/3*t/t_PDS - 1/3)**(3/10)  # pc
    return r


def give_SN_ST_radius(
    t: float = 100e3,
    E: float = 2.7e50,
    n: float = 0.069,
    s:int = 0
) -> float:
    """
    Naive interpretation of the Sedov-Taylor phase of the SNR,
    time in yr"""
    # ST phase
    if s==0:
        r = 2.026**(1/5) * (E)**(1/5) * (n*m_p)**(-1/5) * (t*yr)**(2/5)  # cm
    elif s==2:
        rho_s = 1e10 # Work on the progenitor star
        # DO A DENSITY CURVE
        r = 0.5**(1/(5-s)) * (E)**(1/(5-s)) * (rho_s)**(-1/(5-s)) * \
        (t*yr)**(2/(5-s))  # cm
    return r/pc  # pc


def give_ISM_sound_speed(T: float = 8000) -> float:
    """Temperature of the ISM may vary depending on the region
    (dust, cloud, gas, etc)"""
    return np.sqrt(gamma*kB*T/(mu_p*m_p))  # cm/s


def plot_ISM_sound_speed():

    T_arr = np.logspace(0, 7) # K

    fig = plt.figure()

    plt.plot(T_arr, give_ISM_sound_speed(T_arr)/1e5)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Temperature [K]")
    plt.ylabel("Sound speed [km/s]")
    plt.grid()
    fig.tight_layout()
    plt.show()


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


def give_SN_MCS_time(
    E: float = 2.7e50,
    n: float = 0.069,
    chi: float = 1, # metallicity
) -> float:
    t_PDS = give_SN_PDS_time(E, n, chi)
    t = 61 * t_PDS * (7)**3 * \
        chi**(-3/14) * n**(-3/7) * (E/1e51)**(-3/14)  # yr
    return t  # yr


def give_SN_MCS_radius(
    t: np.ndarray = 100e3,
    E: float = 2.7e50,
    n: float = 0.069,
    chi: float = 1,
) -> np.ndarray:
    t_PDS = give_SN_PDS_time(E, n, chi)
    t_MCS = give_SN_MCS_time(E, n, chi)
    r_PDS = give_SN_PDS_radius(E, n, chi)
    r_MCS = (4.66*(t_MCS/t_PDS)*(1-0.939*(t_MCS/t_PDS) **
             (-0.17) - 0.153*(t_MCS/t_PDS)**(-1)))**(1/4)
    r = r_PDS * (4.66*(t/t_PDS - t_MCS/t_PDS) *
                 (1 - 0.779*(t_MCS/t_PDS)**(-0.17)) + r_MCS**4)**(1/4)
    return r


def give_SN_radius(
    t: np.ndarray = 100e3,
    E: float = 2.7e50,
    n: float = 0.069,
    chi: float = 1,
    T: float = 8000,
    s:int = 0
) -> float:
    """Decides the phase of the SNR and gives the appropriate radius
    in pc. Requires a time in yr"""
    # Determining the SNR stage
    t_PDS = give_SN_PDS_time(E, n, chi)
    t_MCS = give_SN_MCS_time(E, n, chi)
    t_max = give_SN_merge_time(give_ISM_sound_speed(T), E, n)
    if t > t_max:
        r = 0
    elif t < t_PDS:
        r = give_SN_ST_radius(t, E, n, s)
    elif t < t_MCS:
        r = give_SN_PDS_radius(t, E, n, chi, s)
    else:
        r = give_SN_MCS_radius(t, E, n, chi)
    return r


def give_CSM_density(r:float, M:float = 8, n_ISM:float = 0.069):

    M = np.array([M])

    r_w = give_wind_radius(M)[0] # pc
    r_b = give_bubble_radius(M) # pc

    if r < r_w:
        n = 1e10/m_p * (r*pc)**(-2) # /cm3
    elif r < r_b:
        n = give_bubble_density(M, n_ISM)[0]
    else:
        n = n_ISM

    return n


def give_mass_radius(r:float, M:float = 8, n_ISM:float = 0.069):

    M = np.array([M])
    r = np.array([r])

    r_w = give_wind_radius(M)[0] # pc
    r_b = give_bubble_radius(M) # pc

    M_loss = give_mass_loss_RSG(M)*Msol/yr # g/s
    u_w = give_wind_speed_RSG(M)*1e5 # cm/s

    if r < r_w:
        M_shock = M_loss/u_w*(r*pc) # g
        return M_shock/Msol # Msol
    
    if r < r_b:
        n_bubble = give_bubble_density(M, n_ISM) # /cm3
        M_shock = M_loss/u_w*(r_w*pc) + 4*np.pi/3*n_bubble*m_p*(r*pc)**3 # g
        return M_shock/Msol # Msol

    M_shock = 4*np.pi/3*n_ISM*m_p*((r*pc)**3) # g, ISM

    return M_shock/Msol # Msol


def plot_CSM_density_mass():

    r_arr = np.logspace(-4, 4, 500) # pc

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    M_arr = np.linspace(8, 120, 5)

    for M_ in M_arr:
        n_arr = np.array([give_CSM_density(r_, M_) for r_ in r_arr]) # /cm3
        ax1.plot(r_arr, n_arr, label=f"$M={M_:.0f}$ M$_\odot$")

        mass_arr = np.array([give_mass_radius(r_, M_, n_ISM=1) for r_ in r_arr]) # Msol
        ax2.plot(r_arr, mass_arr)

    ax2.axhline(y=5, linestyle = "--", linewidth=0.5,
                color="black", label=r"$M_\mathrm{ej}$")

    ISM_density_mass = 4*np.pi/3*m_p*(r_arr*pc)**3/Msol
    ax2.plot(r_arr, ISM_density_mass, color="black", linestyle="-.",
             label=r"Mass accumulated in a"u"\n"r"normal homogeneous ISM")

    ax2.set_xlabel(r"$r$ [pc]")
    ax1.set_ylabel(r"$n$ [cm$^{-3}$]")
    ax2.set_xscale("log")
    ax1.set_yscale("log")
    ax1.grid()
    ax1.legend(fontsize=9)
    ax2.legend(fontsize=9)
    ax2.set_ylabel(r"$M$ [M$_\odot$]")
    ax2.set_yscale("log")
    ax2.grid()
    fig.tight_layout()
    plt.show()


def plot_SN_radius_varying_E(n: float = 0.069, chi: float = 1) -> None:
    """From Cioffi et al. 1988"""

    E_arr = np.array([0.1, 0.5, 1, 2, 5])*1e51

    fig = plt.figure()

    for E_ in E_arr:
        t_arr = np.logspace(1, 7, 500)  # yr
        r_arr = np.array([give_SN_radius(t_, E_, n, chi) for t_ in t_arr])


        plt.plot(t_arr, r_arr, linestyle="-",
                label=r"$E =$"f"${E_}$"r" erg")
        # plt.axvline(x=t_PDS, linestyle="--",
        #             label=r"$t_\mathrm{PDS} =$"f"${t_PDS/1e3:.1f}$"r" kyr")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(fontsize=12)
    plt.grid()
    plt.xlabel(r"$t$ [yr]")
    plt.ylabel(r"$R_\mathrm{s}(t)$ [pc]")
    fig.tight_layout()
    plt.savefig("Project Summary/Images/R_SN(t)_E.pdf")
    plt.show()


def plot_SN_radius_varying_n(E: float = 2.7e50, chi: float = 1) -> None:
    """From Cioffi et al. 1988"""

    n_arr = np.array([0.01, 0.05, 0.069, 0.1, 0.5])

    fig = plt.figure()

    for n_ in n_arr:
        t_max = give_SN_merge_time(give_ISM_sound_speed(), E, n_, chi)
        t_arr = np.logspace(1, 7, 500)  # yr
        r_arr = np.array([give_SN_radius(t_, E, n_, chi) for t_ in t_arr])
        t_PDS = give_SN_PDS_time(E, n_, chi)

    
        plt.plot(t_arr, r_arr, linestyle="-",
                label=r"$n_\mathrm{ISM} =$"f"${n_}$"r" cm$^{-3}$")
        # plt.axvline(x=t_PDS, linestyle="--",
        #             label=r"$t_\mathrm{PDS} =$"f"${t_PDS/1e3:.1f}$"r" kyr")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(fontsize=12)
    plt.grid()
    plt.xlabel(r"$t$ [yr]")
    plt.ylabel(r"$R_\mathrm{s}(t)$ [pc]")
    fig.tight_layout()
    plt.savefig("Project Summary/Images/R_SN(t)_n.pdf")
    plt.show()


def plot_SN_radius(E: float = 2.7e50,
                   n:float =0.069,
                   chi: float = 1,
                   s:int = 0
                   ) -> None:
    """From Cioffi et al. 1988"""

    n_arr = np.array([0.01, 0.05, 0.069, 0.1, 0.5])

    fig, ax = plt.subplots(1, 2, sharey=True)

    for n_ in n_arr:
        t_arr = np.logspace(1, 7, 500)  # yr
        r_arr = np.array([give_SN_radius(t_, E, n_, chi, s=s) for t_ in t_arr])
    
        ax[0].plot(t_arr, r_arr, linestyle="-",
                label=r"$n_\mathrm{ISM} =$"f"${n_}$"r" cm$^{-3}$")
        
    E_arr = np.array([0.1, 0.5, 1, 2, 5])*1e51

    for E_ in E_arr:
        t_arr = np.logspace(1, 7, 500)  # yr
        r_arr = np.array([give_SN_radius(t_, E_, n, chi, s=s) for t_ in t_arr])

        ax[1].plot(t_arr, r_arr, linestyle="-",
                label=r"$E_\mathrm{SN} =$"f"${E_}$"r" erg")

    ax[0].set_xscale("log")
    ax[1].set_xscale("log")
    plt.yscale("log")
    ax[0].legend(fontsize=12)
    ax[0].grid()
    ax[1].legend(fontsize=12)
    ax[1].grid()
    ax[0].set_xlabel(r"$t$ [yr]")
    ax[1].set_xlabel(r"$t$ [yr]")
    ax[0].set_ylabel(r"$R_\mathrm{s}(t)$ [pc]")
    fig.tight_layout()
    plt.savefig("Project Summary/Images/R_SN(t).pdf")
    plt.show()


def plot_SN_radius_comparison_medium():

    t_arr = np.logspace(1, 7, 500)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})

    r_arr1 = np.array([give_SN_radius(t_, s=0) for t_ in t_arr])
    r_arr2 = np.array([give_SN_radius(t_, s=2) for t_ in t_arr])
    ratio = r_arr2/r_arr1
    
    ax1.plot(t_arr, r_arr1, linestyle="-", label=r"$s =$"f"${0}$")
        
    ax1.plot(t_arr, r_arr2, linestyle="-", label=r"$s =$"f"${2}$")
    
    ax2.plot(t_arr, ratio, linestyle="-")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.legend(fontsize=12)
    ax1.grid()
    ax2.grid()
    ax2.set_xlabel(r"$t$ [yr]")
    ax1.set_ylabel(r"$R_\mathrm{s}(t)$ [pc]")
    ax2.set_ylabel(r"Ratio of $s_2/s_0$", fontsize=10)
    fig.tight_layout()
    plt.savefig("Project Summary/Images/R_SN(t)_comparing_CSM.pdf")
    plt.show()


def plot_SN_radius_comparison_Leahy() -> None:
    """From Cioffi et al. 1988"""

    t_arr = np.logspace(1, 7, 500)  # yr
    r_arr = np.array([give_SN_radius(t_, 2.7e50, 0.069, 1) for t_ in t_arr])

    data = np.genfromtxt("data Leahy.txt", delimiter=',', skip_header=1)
    t_leahy = data[:,0]
    r_leahy = data[:,1]

    fig = plt.figure()
    
    plt.plot(t_arr, r_arr, linestyle="-", label=r"This work")
    plt.plot(t_leahy, r_leahy, linestyle='-', label=r"Leahy et al. 2017")
    
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(fontsize=12)
    plt.grid()
    plt.xlabel(r"$t$ [yr]")
    plt.ylabel(r"$R_\mathrm{s}(t)$ [pc]")
    fig.tight_layout()
    plt.savefig("Project Summary/Images/R_SN(t)_comparison.pdf")
    plt.show()


def plot_SN_radius_extreme_cases(chi: float = 1) -> None:
    """From Cioffi et al. 1988"""

    t_arr = np.logspace(1, 8, 500)  # yr

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})

    r_arr1 = np.array([give_SN_radius(t_, 2e50, 0.17, chi) for t_ in t_arr])
    r_arr2 = np.array([give_SN_radius(t_, 5e50, 0.03, chi) for t_ in t_arr])
    ratio = r_arr2/r_arr1
    
    ax1.plot(t_arr, r_arr1, linestyle="-",
            label=r"$n_\mathrm{ISM} =$"f"${0.1}$"r" cm$^{-3}$"u"\n"
                  r"$E_\mathrm{SN} =$"f"${8e49}$"r" erg")
        
    ax1.plot(t_arr, r_arr2, linestyle="-",
            label=r"$n_\mathrm{ISM} =$"f"${0.01}$"r" cm$^{-3}$"u"\n"
                  r"$E_\mathrm{SN} =$"f"${6e50}$"r" erg")
    
    ax2.plot(t_arr, ratio, linestyle="-")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.legend(fontsize=12)
    ax1.grid()
    ax2.grid()
    ax2.set_xlabel(r"$t$ [yr]")
    ax1.set_ylabel(r"$R_\mathrm{s}(t)$ [pc]")
    ax2.set_ylabel(r"Ratio of max/min", fontsize=10)
    fig.tight_layout()
    plt.savefig("Project Summary/Images/R_SN(t)_extreme_cases.pdf")
    plt.show()


def make_multivariate_lognormal(x, y, mu1, mu2, sigma1, sigma2):
    mu1, mu2 = np.log(mu1), np.log(mu2)
    sigma1, sigma2 = np.log10(sigma1), np.log10(sigma2)
    return 1/(2*np.pi*sigma1*sigma2)*np.exp(-1/2*(
        ((np.log(x) - mu1)/sigma1)**2 + ((np.log(y) - mu2)/sigma2)**2))


def plot_SN_radius_varying_parameters(t:float = 1e3 # yr
                                      ) -> None:
    """From Cioffi et al. 1988"""

    # Preparing the 2D
    n_arr = np.logspace(np.log10(6e-3), np.log10(2))
    E_arr = np.logspace(np.log10(4e49), np.log10(2e51))

    nn, EE = np.meshgrid(n_arr, E_arr)

    radius_func = np.vectorize(give_SN_radius)
    r = radius_func(t, E = EE, n = nn)

    make_func = np.vectorize(make_multivariate_lognormal)
    pdf = make_func(nn, EE, 0.069, 2.7e50, 5.1, 3.5)/\
          np.max(make_func(nn, EE, 0.069, 2.7e50, 5.1, 3.5))


    fig = plt.figure()

    extent = [np.min(n_arr), np.max(n_arr), np.min(E_arr), np.max(E_arr)]

    CS = plt.contour(nn, EE, r, [70], colors="black", extent=extent,
                     linestyles="-")
    plt.clabel(CS, fmt=r"$70~\mathrm{pc}$", inline=False,
               manual=[(1, -3e50)])

    CS2 = plt.contour(nn, EE, pdf, [0.00269, 0.04550, 0.31731],
                      extent=extent, colors="black",
                      linestyles="--", linewidths=0.5)
    fmt = {}
    strs = ["3$\sigma$", "$2\sigma$", "$\sigma$"]
    for l, s in zip(CS2.levels, strs):
        fmt[l] = s
    plt.clabel(CS2, fontsize=10, fmt=fmt, inline=False)

    im = plt.contourf(nn, EE, r, int(np.max(r)-np.min(r)),
                 extent=extent, cmap="gist_rainbow")
    
    plt.scatter(0.069, 2.7e50, c="blue")
    plt.scatter(0.034, 8e50, c="red")
    plt.scatter(1, 1e51, c="black")
    plt.text(0.0225, 8.5e50, s="Monogem", fontsize=10)
    
    
    clb = plt.colorbar(im)
    clb.set_label(r"$R_\mathrm{s}(E_\mathrm{SN}, n_\mathrm{ISM})$ [pc]")
    clb.add_lines(CS)
    tick_locator = mpl.ticker.MaxNLocator(nbins=5)
    clb.locator = tick_locator
    clb.update_ticks()

    plt.xlabel(r"$n_\mathrm{ISM}$ [cm$^{-3}$]")
    plt.ylabel(r"$E_\mathrm{SN}$ [erg]")
    plt.xscale("log")
    plt.yscale("log")
    fig.tight_layout()
    plt.savefig("Project Summary/Images/R_SN(E_SN, n_ISM).pdf")
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
    M = np.linspace(8, 150, 500)  # Solar masses

    fig = plt.figure()
    plt.plot(M, give_bubble_radius(M), label=r"$r_\mathrm{b}$")
    plt.plot(M, give_wind_radius(M)[0], label=r"$r_\mathrm{w}$")
    plt.axvline(x=16, color = "black", linestyle="--", label=r"B$\rightarrow$O stars")
    plt.axvline(x=40, color = "green", linestyle="--", label=r"O$\rightarrow$WR stars")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(fontsize=12)
    plt.grid()
    plt.xlabel(r"$M$ [M$_\odot$]")
    plt.ylabel(r"$r$ [pc]")
    fig.tight_layout()
    plt.show()


def give_SN_ST_time(E:float = 2.7e50,
                    n:float = 0.069,
                    M:float = 8,
                    Mej:float = 1.4,
                    s:int = 0):
    """Time in s"""
    if s==0:
        t_SN = 423*(E/1e51)**(-1/2) * (Mej)**(5/6) * (n)**(-1/3)
    elif s==2:
        M_loss = give_mass_loss_RSG(M) # Msol/yr    
        v_w = give_wind_speed_RSG(M)*1e5 # cm/s
        t_SN = 1770*(E/1e51)**(-1/2) * (Mej)**(3/2) * \
                (M_loss/1e-5)**(-1) * (v_w/1e6)
    return t_SN # yr


def plot_characteristic_time_scales():

    M = np.array([16])

    r_w = 1.5*pc #give_wind_radius(M)[0]*pc # cm
    r_b = 25*pc #give_bubble_radius(M)*pc # cm

    t_w = shock.give_time_radius_integration(r_w, r_w, r_b)/yr
    t_b = shock.give_time_radius_integration(r_b, r_w, r_b)/yr
    t_PDS = give_SN_PDS_time()
    t_PDS2 = give_SN_PDS_time2()
    t_merger = give_SN_merge_time(give_ISM_sound_speed(80000))

    r_arr = np.logspace(np.log10(0.01*pc), np.log10(1000*pc), 10000) # cm
    t_arr = np.array([shock.give_time_radius_integration(r_, r_w, r_b)
                      for r_ in tqdm(r_arr)])/yr
    t_arr2 = np.array([shock.give_time_radius_integration_constant_density(r_)
                     for r_ in tqdm(r_arr)])/yr
    t_arr3 = np.array([shock.give_time_radius_integration_constant_density_Leahy(r_)
                     for r_ in tqdm(r_arr)])/yr
    r_arr = r_arr/pc

    fig = plt.figure()
    plt.plot(t_arr/1e3, r_arr, linewidth="3", label="$R(t)$ in CSM")
    plt.plot(t_arr2/1e3, r_arr, linewidth="2", label="$R(t)$ in ISM")
    plt.plot(t_arr3/1e3, r_arr, linewidth="2", label="$R(t)$ in ISM (Leahy params)")
    plt.axhline(y=r_w/pc, linestyle = ":", color="black", label=r"$r_\mathrm{w}$")
    plt.axhline(y=r_b/pc, linestyle = "-.", color="black", label=r"$r_\mathrm{b}$")
    plt.axvline(x=t_PDS/1e3, linestyle = "--", color="purple", label=r"$t_\mathrm{PDS}$ Cioffi+1988")
    plt.axvline(x=t_PDS2/1e3, linestyle = "--", color="pink", label=r"$t_\mathrm{PDS}$ Vink 2012")
    plt.axvline(x=t_merger/1e3, linestyle = "--", color="orange", label=r"$t_\mathrm{merger}$")
    plt.xlabel(r"$t$ [kyr]")
    plt.ylabel(r"$R$ [pc]")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()
    plt.legend(fontsize=9)
    fig.tight_layout()
    plt.savefig(f"Project Summary/Images/Characteristic_Timescales_M={M[0]}.pdf")
    plt.show()


def give_total_mass_lost(M = 8,
                         postMS_type = "RSG"):
    
    MS_time = give_MS_time(M)
    MS_mass_loss = give_mass_loss_MS(M) * cgs.sun_mass

    if postMS_type == "RSG":
        postMS_time = give_RSG_time(M)
        postMS_mass_loss = give_mass_loss_RSG(M) * cgs.sun_mass
    elif postMS_type in ["WC", "WN"]:
        postMS_time = give_WR_time(M)
        if postMS_type == "WC":
            postMS_mass_loss = give_mass_loss_WC(M) * cgs.sun_mass
        if postMS_type == "WN":
            postMS_mass_loss = give_mass_loss_WN(M) * cgs.sun_mass

    result = MS_time * MS_mass_loss + postMS_time * postMS_mass_loss
    return result


def give_total_mass_lost_MS(M = 8):
    return 10**(2.36*np.log10(M) - 3.15) * cgs.sun_mass


def give_total_mass_lost_RSG(M = 8):
    return 10**(0.85*np.log10(M) - 0.24) * cgs.sun_mass


def give_total_mass_lost_WR(M = 8):
    return 10**(1.64*np.log10(M) - 1.57) * cgs.sun_mass


def give_total_mass_lost_all_phases(M = 8):

    total_mass = 0
    total_mass += give_total_mass_lost_MS(M)

    if M < 40:
        total_mass += give_total_mass_lost_RSG(M)
    else:
        total_mass += give_total_mass_lost_WR(M)

    return total_mass


def give_wind_density(M = 8, typee = "RSG"):

    mass_loss = give_mass_loss_type(M, typee) / (cgs.sun_mass/cgs.year)
    wind_speed = give_wind_speed_type(M, typee) * cgs.km

    wind_density = mass_loss/(4*np.pi *  wind_speed)/cgs.proton_mass

    return wind_density


def plot_distributions():

    n = 10000

    bubble_density = []
    ejected_mass = []
    wind_speed = []
    bubble_radius = []
    wind_radius = []
    mass_loss = []
    wind_density = []
    luminosity = []
    MS_time = []

    for _ in tqdm(range(n)):
        M = give_random_value(pick_IMF, 8, 120)
        star = Star(M, n_ISM=1)
        bubble_density.append(star.bubble_density/(cgs.proton_mass))
        ejected_mass.append(star.ejected_mass/cgs.sun_mass)
        wind_speed.append(star.wind_speed/cgs.km)
        bubble_radius.append(star.bubble_radius/(cgs.pc))
        wind_radius.append(star.wind_radius/(cgs.pc))
        mass_loss.append(star.mass_loss/(cgs.sun_mass/cgs.year))
        wind_density.append(star.wind_density/(star.wind_radius/cgs.pc)**2/cgs.proton_mass)
        luminosity.append(star.luminosity)
        MS_time.append(star.MS_time/cgs.Myr)
        
    mass_loss = np.array(mass_loss)
    wind_radius = np.array(wind_radius)
    bubble_radius = np.array(bubble_radius)
    wind_speed = np.array(wind_speed)
    ejected_mass = np.array(ejected_mass)
    bubble_density = np.array(bubble_density)
    wind_density = np.array(wind_density)
    luminosity = np.array(luminosity)
    MS_time = np.array(MS_time)

    _, bins = np.histogram(mass_loss, bins=50)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))

    fig = plt.figure()
    plt.hist(mass_loss, histtype="step", bins=logbins, label="")
    plt.xscale("log")
    plt.xlabel("Mass loss [M$_\odot$/yr]")  
    plt.ylabel("Stars")
    plt.grid()
    fig.tight_layout()
    plt.savefig("Project Summary/Images/Mass loss.pdf")
    plt.savefig("CSM_plots/Mass loss.png")
    plt.savefig("CSM_plots/Mass loss.pdf")
    plt.show()

    fig = plt.figure()
    plt.hist(wind_radius, histtype="step", bins=50, label="")
    plt.xlabel("Wind radius [pc]")
    plt.ylabel("Stars")
    plt.grid()
    fig.tight_layout()
    plt.savefig("Project Summary/Images/Wind radius.pdf")
    plt.savefig("CSM_plots/Wind radius.png")
    plt.savefig("CSM_plots/Wind radius.pdf")
    plt.show()
    
    fig = plt.figure()
    plt.hist(bubble_radius, histtype="step", bins=50, label="")
    plt.xlabel("Bubble radius [pc]")
    plt.ylabel("Stars")
    plt.grid()
    fig.tight_layout()
    plt.savefig("Project Summary/Images/Bubble radius.pdf")
    plt.savefig("CSM_plots/Bubble radius.png")
    plt.savefig("CSM_plots/Bubble radius.pdf")
    plt.show()
    
    fig = plt.figure()
    plt.hist(wind_speed, histtype="step", bins=50, label="")
    plt.xlabel("Wind Speed [km/s]")
    plt.ylabel("Stars")
    plt.grid()
    fig.tight_layout()
    plt.savefig("Project Summary/Images/Wind Speed.pdf")
    plt.savefig("CSM_plots/Wind Speed.png")
    plt.savefig("CSM_plots/Wind Speed.pdf")
    plt.show()
    
    fig = plt.figure()
    plt.hist(ejected_mass, histtype="step", bins=50, label="")
    plt.xlabel("SNR ejecta Mass [M$_\odot$]")
    plt.ylabel("Stars")
    plt.grid()
    fig.tight_layout()
    plt.savefig("Project Summary/Images/Ejected Mass.pdf")
    plt.savefig("CSM_plots/Ejected Mass.png")
    plt.savefig("CSM_plots/Ejected Mass.pdf")
    plt.show()

    _, bins = np.histogram(bubble_density, bins=50)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    
    fig = plt.figure()
    plt.hist(bubble_density, histtype="step", bins=logbins, label="")
    plt.xscale("log")
    plt.xlabel("Bubble density [cm$^{-3}$]")
    plt.ylabel("Stars")
    plt.grid()
    fig.tight_layout()
    plt.savefig("Project Summary/Images/Bubble density.pdf")
    plt.savefig("CSM_plots/Bubble density.png")
    plt.savefig("CSM_plots/Bubble density.pdf")
    plt.show()

    _, bins = np.histogram(wind_density, bins=50)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    
    fig = plt.figure()
    plt.hist(wind_density, histtype="step", bins=logbins, label="")
    plt.xscale("log")
    plt.xlabel("Wind density [cm$^{-3}$]")
    plt.ylabel("Stars")
    plt.grid()
    fig.tight_layout()
    plt.savefig("Project Summary/Images/Wind density.pdf")
    plt.savefig("CSM_plots/Wind density.png")
    plt.savefig("CSM_plots/Wind density.pdf")
    plt.show()
    
    _, bins = np.histogram(luminosity, bins=50)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    
    fig = plt.figure()
    plt.hist(luminosity, histtype="step", bins=logbins, label="")
    plt.xscale("log")
    plt.xlabel("Luminosity [erg/s]")
    plt.ylabel("Stars")
    plt.grid()
    fig.tight_layout()
    plt.savefig("Project Summary/Images/Luminosity.pdf")
    plt.savefig("CSM_plots/Luminosity.png")
    plt.savefig("CSM_plots/Luminosity.pdf")
    plt.show()

    fig = plt.figure()
    plt.hist(MS_time, histtype="step", bins=50, label="")
    plt.xlabel("MS time [Myr]")
    plt.ylabel("Stars")
    plt.grid()
    fig.tight_layout()
    plt.savefig("Project Summary/Images/MS time.pdf")
    plt.savefig("CSM_plots/MS time.png")
    plt.savefig("CSM_plots/MS time.pdf")
    plt.show()




class Star:
    def __init__(self, M = 8, n_ISM:float=1):

        self.init_mass = M
        
        self.ism_density = n_ISM
        self.MS_type = "O" if self.init_mass > 16 else "B"
        self.postMS_type = "RSG" if self.init_mass < 40 \
                                 else random.choice(["WC", "WN"])
        
        self.MS_time = give_MS_time(self.init_mass) * cgs.year

        # self.total_mass_lost = give_total_mass_lost(self.init_mass, self.postMS_type)
        self.total_mass_lost = give_total_mass_lost_all_phases(self.init_mass)
        self.mass_loss = give_mass_loss_MS(self.init_mass) * cgs.sun_mass / cgs.year
        self.ejected_mass = np.abs(self.init_mass*cgs.sun_mass - self.total_mass_lost - cgs.pulsar_mass)
        self.wind_speed = give_wind_speed_type(self.init_mass, self.MS_type) * cgs.km
        self.wind_speed_postMS = give_wind_speed_type(self.init_mass, self.postMS_type) * cgs.km
        self.wind_radius = give_wind_radius_type(np.array([self.init_mass]), self.MS_type, self.ism_density)[0] * cgs.pc
        self.bubble_radius = give_bubble_radius_type(np.array([self.init_mass]), self.ism_density, self.MS_type)[0] * cgs.pc
        self.wind_density =  give_wind_density(self.init_mass, self.postMS_type)
        self.bubble_density = give_bubble_density_type(np.array([self.init_mass]), self.ism_density)[0] * cgs.proton_mass
        self.luminosity = give_wind_luminosity_type([self.init_mass])

        self.wind_kinetic_power = self.mass_loss * self.wind_speed**2 / 2



gamma = 5/3  # adiabatic coefficient for monoatomic gas
m_p = 1.6726e-24  # g
mu_p = 1.4  # molecular fraction in the ISM
kB = 1.3807e-16  # erg/K Boltzmann constant
pc = 3e18  # cm/pc
Msol = 1.989e33  # g/Msol
yr = np.pi * 1e7  # s/yr
AGE_GEMINGA = 342e3 # yr


if __name__ == "__main__":
    # TESTS AND PLOTS

    # test_IMF()
    # plot_MS_time()
    # plot_wind_luminosity()
    # plot_wind_speed()
    # plot_mass_loss()
    # plot_bubble_density()
    # plot_bubble_radius()
    # plot_SN_radius(s=0)
    # plot_SN_radius_comparison_medium()
    # plot_CSM_density_mass()

    # plot_ISM_sound_speed()

    # plot_characteristic_time_scales()

    # plot_SN_radius_comparison_Leahy()
    # plot_SN_radius_extreme_cases()
    # plot_SN_radius_varying_parameters(AGE_GEMINGA)


    plot_distributions()



    1
