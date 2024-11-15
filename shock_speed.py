import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from tqdm import tqdm

np.set_printoptions(precision=1)
plt.rcParams.update({'font.size': 17})
plt.rcParams["font.family"] = "serif"


def give_ST_radius(t: np.ndarray) -> np.ndarray:
    """t in yr, from ST paper"""
    m_p = 1.6726e-24  # g
    E = 2.7e50  # erg.s-1
    rho_ISM = m_p * 0.069  # g.cm-3
    t = t * np.pi * 1e7  # s
    r = (2.026*E/rho_ISM)**(1/5)*t**(2/5)
    return r/3e18  # pc


def give_SN_PDS_time2(
    E: float = 2.7e50,
    n: float = 0.069,
) -> float:
    """Formula and parameters from Vink 2012.
    Returns the Pressure Driven Snowplough time in yr"""
    t = 44600 * (E/1e51)**(1/3) * (n)**(-1/3)   # yr
    return t


def give_mass_radius_analytical(
        r: np.ndarray,
        r_w: float,
        r_b: float,
        M=5*1.989e33
) -> np.ndarray:
    r"""Gives the mass at the shock depending on the SNR radius `r`.
    The SNR sweeps the matter that is carved by the stellar winds before
    the SN. The computation is analytical.

    We use a three-zone model, separated by two characteristic radii,
    called `r_w`, the wind termination shock radius during the MS life 
    of the star, and `r_b`, the radius of the bubble that is blown by
    the star. The model is detailed in Cristofari et al. 2020.

    These two parameters are computed in the package `SN_bubble.py`.

    Args:
        r (np.ndarray): cm, radius at which we want to know the mass
        r_w (float): cm, wind termination shock radius
        r_b (float): cm, bubble radius

    Returns:
        np.ndarray: g, mass at the shock when it passes at radius `r`
    """
    M_arr = []  # g

    if type(r) in [np.float64, int, float]:
        if r < r_w:
            M_arr.append(M + M_loss/u_w*r)
        else:
            if r < r_b:
                M_arr.append(M
                             + M_loss/u_w*r_w
                             + 4*np.pi/3*rho_b*(r**3 - r_w**3))
            else:
                M_arr.append(M
                             + M_loss/u_w*r_w
                             + 4*np.pi/3*rho_b*(r_b**3 - r_w**3)
                             + 4*np.pi/3*rho_ISM*(r**3 - r_b**3))
    else:
        for r_ in r:
            if r_ < r_w:
                M_arr.append(M + M_loss/u_w*r_)
            else:
                if r_ < r_b:
                    M_arr.append(M
                                 + M_loss/u_w*r_w
                                 + 4*np.pi/3*rho_b*(r_**3 - r_w**3))
                else:
                    M_arr.append(M
                                 + M_loss/u_w*r_w
                                 + 4*np.pi/3*rho_b*(r_b**3 - r_w**3)
                                 + 4*np.pi/3*rho_ISM*(r_**3 - r_b**3))

    return np.array(M_arr)  # g


def give_speed_radius_analytical(
        r: np.ndarray,
        r_w: float,
        r_b: float,
        E: float = 1e51,
        M: float = 5*1.989e33
) -> np.ndarray:
    """Gives the speed of the shock at a radius `r`, for a SNR
    propagating in the three-zone model discussed in
    `give_mass_radius_analytical`. The computation is
    analytical and follows Cristofari et al. 2020.

    Args:
        r (np.ndarray): cm, radius at which we want to know the mass
        r_w (float): cm, wind termination shock radius
        r_b (float): cm, bubble radius
        E (float, optional): erg, energy emitted during the SN.
            Defaults to E_SN = 1e51 erg.
        M (float, optional): g, mass ejected during the SN.
            Defaults to M_ej = 1.989e33 g.

    Returns:
        np.ndarray: cm/s, speed of the shock at radius `r`.
    """
    u_arr = []
    
    if type(r) in [np.float64, int, float]:
        factor = 2*alpha*E \
            / (give_mass_radius_analytical(r, r_w, r_b, M)**2 * r**alpha)
        if r < r_w:
            u_element = factor\
                * ((M/alpha) * r**alpha
                   + (M_loss/u_w) * r**(alpha+1)/(alpha+1))
        elif r > r_w:
            if r < r_b:
                u_element = factor\
                    * (M*r_w**alpha/alpha
                       + M_loss/u_w*r_w**(alpha+1)/(alpha+1)
                       + (M + M_loss/u_w*r_w - 4*np.pi/3*rho_b*r_w**3)
                        * (r**alpha - r_w**alpha)/alpha
                        + 4*np.pi/3*rho_b*(r**(alpha+3)
                                           - r_w**(alpha+3))/(alpha+3))
            else:
                u_element = factor\
                    * (M*r_b**alpha/alpha
                       + M_loss/u_w*r_w**(alpha+1)/(alpha+1)
                       + (M + M_loss/u_w*r_w
                           - 4*np.pi/3*rho_b*(r_b**3 - r_w**3))
                       * (r_b**alpha - r_w**alpha)/alpha
                       + (M_loss/u_w*r_w
                          + 4*np.pi/3*rho_b*(r_b**3 - r_w**3)
                          - 4*np.pi/3*rho_ISM*r_b**3)
                       * (r**alpha - r_b**alpha)/alpha
                       + 4*np.pi/3*rho_ISM*(r**(alpha+3)
                                            - r_b**(alpha+3))/(alpha+3))

        u_arr = u_element

    else:
        u_arr = np.array([])
        for r_ in r:
            factor = 2*alpha*E\
                / (give_mass_radius_analytical(r_, r_w, r_b, M)**2 * r_**alpha)
            if r_ < r_w:
                u_element = factor\
                    * ((M/alpha) * r_**alpha
                       + (M_loss/u_w) * r_**(alpha+1)/(alpha+1))
            elif r_ > r_w:
                if r_ < r_b:
                    u_element = factor\
                        * (M*r_w**alpha/alpha
                           + M_loss/u_w*r_w**(alpha+1)/(alpha+1)
                           + (M + M_loss/u_w*r_w - 4*np.pi/3*rho_b*r_w**3)
                           * (r_**alpha - r_w**alpha)/alpha
                           + 4*np.pi/3*rho_b*(r_**(alpha+3)
                                              - r_w**(alpha+3))/(alpha+3))
                else:
                    u_element = factor\
                        * (M*r_b**alpha/alpha
                           + M_loss/u_w*r_w**(alpha+1)/(alpha+1)
                           + (M + M_loss/u_w*r_w
                              + 4*np.pi/3*rho_b*(r_b**3 - r_w**3))
                           * (r_b**alpha - r_w**alpha)/alpha
                           + (M_loss/u_w*r_w
                              + 4*np.pi/3*rho_b*(r_b**3 - r_w**3)
                              - 4*np.pi/3*rho_ISM*r_b**3)
                           * (r_**alpha - r_b**alpha)/alpha
                           + 4*np.pi/3*rho_ISM*(r_**(alpha+3)
                                                - r_b**(alpha+3))/(alpha+3))

            u_arr = np.append(u_arr, u_element)

    u_arr = (gamma+1)/2 * np.abs(u_arr)**(1/2)

    return u_arr  # cm.s-1


def give_speed_radius_analytical_constant_density_Leahy(
        r: np.ndarray,
        M: float = 1.989e33*5
) -> np.ndarray:
    """Gives the speed of the shock at a radius `r`, for a SNR
    propagating in a constant density ISM. The computation is
    analytical and follows Cristofari et al. 2020.

    Args:
        r (np.ndarray): cm, radius at which we want to know the mass
        E (float, optional): erg, energy emitted during the SN.
            Defaults to E_SN = 1e51 erg.
        M (float, optional): g, mass ejected during the SN.
            Defaults to M_ej = 1.989e33 g.

    Returns:
        np.ndarray: cm/s, speed of the shock at radius `r`.
    """
    u_arr = []
    rho_ISM_Leahy = 0.069*m_p # g/cm3
    E_Leahy = 2.7e50 # erg
    # THIS IS COUNTING FOR THE EFFECTIVE VALUES FOUND BY LEAHY+2020

    mass = M + 4*np.pi/3*rho_ISM_Leahy*r**3

    if type(r) in [np.float64, int, float]:
        factor = 2*alpha*E_Leahy / (mass**2 * r**alpha)
        
        u_element = factor\
            * (M*r**alpha/alpha
                + 4*np.pi/3*rho_ISM_Leahy*r**(alpha+3)/(alpha+3))

        u_arr = u_element

    else:
        for r_ in r:
            factor = 2*alpha*E_Leahy / (mass**2 * r_**alpha)
        
            u_element = factor\
                * (M*r_**alpha/alpha
                    + 4*np.pi/3*rho_ISM_Leahy*r_**(alpha+3)/(alpha+3))

            u_arr.append(u_element)

    u_arr = (gamma+1)/2 * (np.abs(u_arr))**(1/2)

    return u_arr  # cm.s-1


def give_speed_radius_analytical_constant_density(
        r: np.ndarray,
        M: float = 1.989e33*5,
        E: float = 1e51
) -> np.ndarray:
    """Gives the speed of the shock at a radius `r`, for a SNR
    propagating in a constant density ISM. The computation is
    analytical and follows Cristofari et al. 2020.

    Args:
        r (np.ndarray): cm, radius at which we want to know the mass
        E (float, optional): erg, energy emitted during the SN.
            Defaults to E_SN = 1e51 erg.
        M (float, optional): g, mass ejected during the SN.
            Defaults to M_ej = 1.989e33 g.

    Returns:
        np.ndarray: cm/s, speed of the shock at radius `r`.
    """
    u_arr = []

    mass = M + 4*np.pi/3*rho_ISM*r**3

    if type(r) in [np.float64, int, float]:
        factor = 2*alpha*E/ (mass**2 * r**alpha)
        
        u_element = factor\
            * (M*r**alpha/alpha
                + 4*np.pi/3*rho_ISM*r**(alpha+3)/(alpha+3))

        u_arr = u_element

    else:
        for r_ in r:
            factor = 2*alpha*E / (mass**2 * r_**alpha)
        
            u_element = factor\
                * (M*r_**alpha/alpha
                    + 4*np.pi/3*rho_ISM*r_**(alpha+3)/(alpha+3))

            u_arr.append(u_element)

    u_arr = (gamma+1)/2 * (np.abs(u_arr))**(1/2)

    return u_arr  # cm.s-1


def integrate_simpson(
        f,
        a: float,
        b: float,
        steps: float
) -> float:
    if steps % 2 != 0:
        steps += 1

    x_axis = np.linspace(a, b, steps)
    h = x_axis[1] - x_axis[0]
    I = f(a) + f(b)

    for i in range(steps):
        if i % 2 != 0:  # odd
            I += 4 * f(x_axis[i])
        else:        # even
            I += 2 * f(x_axis[i])
    I = h/3 * I
    return I


def give_time_radius_integration(
        r: float,
        r_w: float,
        r_b: float,
        M=5*1.989e33
) -> np.ndarray:
    """Finds the relationship between time and position of the SNR by
    integrating on the inverse of the SNR speed.

    Args:
        r_w (float): cm, wind termination shock radius
        r_b (float): cm, bubble radius

    Returns:
        float: time (in s) float.
    """

    t =  integrate.quad(lambda x: 1 /
         give_speed_radius_analytical(x, r_w, r_b, M=M), 0.001, r)[0]

    return t


def give_time_radius_integration_constant_density(
        r: float,
        M=5
) -> np.ndarray:
    """Finds the relationship between time and position of the SNR by
    integrating on the inverse of the SNR speed.

    Args:

    Returns:
        float: time (in s) float.
    """

    t =  integrate.quad(lambda x: 1 /
         give_speed_radius_analytical_constant_density(x, M=M), 0.001, r)[0]

    return t


def give_time_radius_integration_constant_density_Leahy(
        r: float,
) -> np.ndarray:
    """Finds the relationship between time and position of the SNR by
    integrating on the inverse of the SNR speed.

    Args:

    Returns:
        float: time (in s) float.
    """

    t =  integrate.quad(lambda x: 1 /
         give_speed_radius_analytical_constant_density_Leahy(x), 0.001, r)[0]

    return t


def give_time_radius_integration2(
        r_w: float,
        r_b: float,
        t: float,
        E: float = 2.7e50,
        M: float = 1
) -> float:
    """Finds the relationship between time and position of the SNR by
    integrating on the inverse of the SNR speed. The integration steps
    is not fixed anymore as we start from the ST approximation to 
    find the complete integration result by slowly increasing the
    upper limit on the radius.

    Args:
        r_w (float): cm, wind termination shock radius
        r_b (float): cm, bubble radius
        t (float): yr, time at which we want the radius.
        E (float, optional): erg, energy emitted during the SN.
            Defaults to E_SN = 1e51 erg.
        M (float, optional): Msol, mass ejected during the SN.
            Defaults to M_ej = 1.989e33 g.

    Returns:
        float: cm, radius of the SNR at time `t`.
    """
    r_element = give_ST_radius(t)*pc  # cm
    t_element = 0

    while t_element < t*(np.pi*1e7):
        r_element += 0.1*pc  # cm
        t_element = integrate.quad(
            lambda x: 1 /
            give_speed_radius_analytical(x, r_w, r_b, E, M*Msol),
            0,
            r_element)[0]

    return r_element


def plot_mass_radius_analytical() -> None:

    r_arr = np.logspace(np.log10(0.01*pc), np.log10(100*pc), 5000)  # cm

    M_arr = (give_mass_radius_analytical(r_arr, r_w, r_b))/Msol  # Msol
    r_arr = r_arr/pc  # pc

    fig = plt.figure()
    plt.plot(r_arr, M_arr, label=r"Analytical")
    plt.axvline(x=r_w/pc, color='red', linestyle="--",
                label=r"$r_\mathrm{w}$")
    plt.axvline(x=r_b/pc, color='green', linestyle="-.",
                label=r"$r_\mathrm{b}$")
    plt.xlabel(r"$r$ [pc]")
    plt.ylabel(r"$M(r)$ [M$_\odot$]")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    fig.tight_layout()
    plt.savefig("CSM_plots/shock_speed mass structure.pdf")
    plt.show()


def plot_speed_radius_analytical() -> None:

    r_arr = np.logspace(np.log10(0.1*pc), np.log10(100*pc), 1000) # cm

    u_arr = give_speed_radius_analytical(r_arr, r_w, r_b)
    # u_arr2 = give_speed_radius_analytical_constant_density(r_arr)
    r_arr = r_arr/pc

    fig = plt.figure()
    plt.plot(r_arr, u_arr, label=r"Analytical CSM")
    # plt.plot(r_arr, u_arr2, label=r"Analytical constant ISM")
    plt.axvline(x=r_w/pc, color='red', linestyle="--",
                label=r"$r_\mathrm{w}$")
    plt.axvline(x=r_b/pc, color='green', linestyle="-.",
                label=r"$r_\mathrm{b}$")
    plt.xlabel(r"$r$ [pc]")
    plt.ylabel(r"$u(r)$ [cm$\cdot$s$^{-1}$]")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    fig.tight_layout()
    plt.savefig("CSM_plots/shock_speed speed structure.pdf")
    plt.show()


def plot_radius_time_integration() -> None:

    r_arr = np.logspace(np.log10(0.1*pc), np.log10(100*pc), 1000) # cm

    fig = plt.figure()
    t_arr = np.array([give_time_radius_integration(r_, r_w, r_b)
                      for r_ in r_arr])
    t_arr = t_arr/yr

    t_arr2 = np.array([give_time_radius_integration_constant_density(r_)
                     for r_ in r_arr])
    t_arr2 = t_arr2/yr

    t_arr3 = np.array([give_time_radius_integration_constant_density_Leahy(r_)
                     for r_ in r_arr])
    t_arr3 = t_arr3/yr
    r_arr = r_arr/pc  # pc

    data = np.genfromtxt("ISM_density/log_fileEL", skip_header=1)
    time = data[:,0]
    radius = data[:,5]/3e18 #pc

    # plt.plot(time, radius, label="Accurate MHD Sim")
    
    plt.plot(t_arr, r_arr, label="CSM")
    plt.plot(t_arr2, r_arr, label="Constant ISM")
    plt.plot(t_arr3, r_arr, label="Constant ISM with Leahy parameters")


    plt.axhline(y=r_w/pc, color='black', linestyle="--",
                label=r"$r_\mathrm{w}$")
    plt.axhline(y=r_b/pc, color='black', linestyle="-.",
                label=r"$r_\mathrm{b}$")
    plt.ylabel(r"$r$ [pc]")
    plt.xlabel(r"$t$ [yr]")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(fontsize=12)
    plt.grid()
    fig.tight_layout()
    plt.savefig("Project Summary/Images/R(t)_comparison_ambient_medium.pdf")
    plt.show()


def plot_speed_time_integration() -> None:

    r_arr = np.logspace(np.log10(0.001*pc), np.log10(100*pc), 10000) # cm

    t_arr = np.array([give_time_radius_integration(r_, r_w, r_b)
                      for r_ in tqdm(r_arr)])
    u_arr = give_speed_radius_analytical(r_arr, r_w, r_b)
    t_arr = t_arr/(1e3*yr)  # kyr

    # Computation of the characteristic times
    t_w = give_time_radius_integration(r_w, r_w, r_b)/(1e3*yr)  # kyr
    t_b = give_time_radius_integration(r_b, r_w, r_b)/(1e3*yr)  # kyr

    t_rad = give_SN_PDS_time2(E=1e51, n=1)/1e3

    fig = plt.figure()
    
    plt.plot(t_arr, u_arr)

    plt.axhline(y=2e7, linestyle=":", color = "black",
                label=r"$v_\mathrm{rad} = 200$ km/s")
    plt.axvline(x=t_rad, linestyle="-", color= "black",
                label=r"$t_\mathrm{rad}$, Vink 2012")

    plt.axvline(x=t_w, linestyle="--", color = "red",
                label=r"$t(r_\mathrm{w})$")
    plt.axvline(x=t_b, linestyle="-.", color = "red",
                label=r"$t(r_\mathrm{b})$")

    plt.ylabel(r"$u(t)$ [cm/s]")
    plt.xlabel(r"$t$ [kyr]")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(1e-3, 1e3)
    plt.ylim(2e5, 5e9)
    plt.legend(fontsize=10)
    plt.grid()
    fig.tight_layout()
    plt.savefig(r"Project Summary/Images/u_s(t).pdf")
    plt.show()


pc = 3e18  # cm/pc
Msol = 1.989e33  # g/Msol
yr = np.pi * 1e7  # s/yr
r_w = 1.5*pc  # cm
r_b = 28*pc  # cm
M_ej = 15*Msol  # g
u_w = 3e6  # cm.s-1
M_loss = 1e-5*Msol/yr  # g.s-1
m_p = 1.6726e-24  # g
rho_b = m_p*1e-2  # g.cm-3
rho_ISM = m_p*1  # g.cm-3
xi = 0.1  # fraction of energy that goes into the acceleration of CRs
E_SN = 1e51  # erg
gamma = 5/3  # adiabatic coefficient for monoatomic gas
alpha = 6*(gamma-1)/(gamma+1)


if __name__ == "__main__":
    # TESTS AND PLOTS

    # plot_mass_radius_analytical()
    # plot_speed_radius_analytical()
    plot_radius_time_integration()
    # plot_speed_time_integration()

    #  print(give_speed_time_integration(1))

    1
