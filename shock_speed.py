import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from tqdm import tqdm

from zeroordergalaxy import give_ST_radius

np.set_printoptions(precision=1)


def give_mass_radius_analytical(
        r: np.ndarray,
        r_w: float,
        r_b: float
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
            M_arr.append(M_ej + M_loss/u_w*r)
        elif r > r_w:
            if r < r_b:
                M_arr.append(M_ej
                             + M_loss/u_w*r_w
                             + 4*np.pi/3*rho_b*(r**3 - r_w**3))
            else:
                M_arr.append(M_ej
                             + M_loss/u_w*r_w
                             + 4*np.pi/3*rho_b*(r_b**3 - r_w**3)
                             + 4*np.pi/3*rho_ISM*(r**3 - r_b**3))
    else:
        for r_ in r:
            if r_ < r_w:
                M_arr.append(M_ej + M_loss/u_w*r_)
            elif r_ > r_w:
                if r_ < r_b:
                    M_arr.append(M_ej
                                 + M_loss/u_w*r_w
                                 + 4*np.pi/3*rho_b*(r_**3 - r_w**3))
                else:
                    M_arr.append(M_ej
                                 + M_loss/u_w*r_w
                                 + 4*np.pi/3*rho_b*(r_b**3 - r_w**3)
                                 + 4*np.pi/3*rho_ISM*(r_**3 - r_b**3))

    return np.array(M_arr)  # g


def give_speed_radius_analytical(
        r: np.ndarray,
        r_w: float,
        r_b: float,
        E: float = 2.7e50,
        M: float = 1.989e33
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
            / (give_mass_radius_analytical(r, r_w, r_b)**2 * r**alpha)
        if r < r_w:
            u_element = factor\
                * ((M/alpha) * r**alpha
                   + (M_loss/u_w) * r**(alpha+1)/(alpha+1))
        elif r > r_w:
            if r < r_b:
                u_element = factor\
                    * (M*r**alpha/alpha
                       + M_loss/u_w*r_w**(alpha+1)/(alpha+1)
                       + (M_loss/u_w*r_w - 4*np.pi/3*rho_b*r_w**3)
                        * (r**alpha - r_w**alpha)/alpha
                        + 4*np.pi/3*rho_b*(r**(alpha+3)
                                           - r_w**(alpha+3))/(alpha+3))
            else:
                u_element = factor\
                    * (M*r**alpha/alpha
                       + M_loss/u_w*r_w**(alpha+1)/(alpha+1)
                       + (M_loss/u_w*r_w
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
        for r_ in r:
            factor = 2*alpha*E\
                / (give_mass_radius_analytical(r_, r_w, r_b)**2 * r_**alpha)
            if r_ < r_w:
                u_element = factor\
                    * ((M/alpha) * r_**alpha
                       + (M_loss/u_w) * r_**(alpha+1)/(alpha+1))
            elif r_ > r_w:
                if r_ < r_b:
                    u_element = factor\
                        * (M*r_**alpha/alpha
                           + M_loss/u_w*r_w**(alpha+1)/(alpha+1)
                           + (M_loss/u_w*r_w - 4*np.pi/3*rho_b*r_w**3)
                           * (r_**alpha - r_w**alpha)/alpha
                           + 4*np.pi/3*rho_b*(r_**(alpha+3)
                                              - r_w**(alpha+3))/(alpha+3))
                else:
                    u_element = factor\
                        * (M*r_**alpha/alpha
                           + M_loss/u_w*r_w**(alpha+1)/(alpha+1)
                           + (M_loss/u_w*r_w
                              + 4*np.pi/3*rho_b*(r_b**3 - r_w**3))
                           * (r_b**alpha - r_w**alpha)/alpha
                           + (M_loss/u_w*r_w
                              + 4*np.pi/3*rho_b*(r_b**3 - r_w**3)
                              - 4*np.pi/3*rho_ISM*r_b**3)
                           * (r_**alpha - r_b**alpha)/alpha
                           + 4*np.pi/3*rho_ISM*(r_**(alpha+3)
                                                - r_b**(alpha+3))/(alpha+3))

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
        r_b: float
) -> np.ndarray:
    """Finds the relationship between time and position of the SNR by
    integrating on the inverse of the SNR speed.

    Args:
        r_w (float): cm, wind termination shock radius
        r_b (float): cm, bubble radius
        n (int, optional): Number of integration steps. Defaults to 100.

    Returns:
        float: time (in s) float.
    """

    t =  integrate.quad(lambda x: 1 /
                        give_speed_radius_analytical(x, r_w, r_b), 0.001, r)[0]

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

    r_arr = np.logspace(np.log10(1e17), np.log10(1e22), 5000)  # cm

    M_arr = give_mass_radius_analytical(r_arr, r_w, r_b)/Msol  # Msol
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
    plt.show()


def plot_speed_radius_analytical() -> None:

    r_arr = np.logspace(np.log10(1e15), np.log10(1e22), 1000)  # cm

    u_arr = give_speed_radius_analytical(r_arr, r_w, r_b)
    r_arr = r_arr/pc

    fig = plt.figure()
    plt.plot(r_arr, u_arr, label=r"Analytical")
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
    plt.show()


def give_speed_time_integration(
        t: np.ndarray,
        r_w: float,
        r_b: float
) -> np.ndarray:
    """t in kyr, returns u(t) in cm.s-1"""

    r_arr = np.logspace(13, 22, 100) # cm

    t_arr = np.array([give_time_radius_integration(r_, r_w, r_b)
                      for r_ in r_arr])

    u_arr = give_speed_radius_analytical(r_arr, r_w, r_b)
    t_arr = t_arr/(1e3*yr)  # kyr

    for i in range(len(t_arr)):
        if t_arr[i] > t:
            return u_arr[i]


def plot_radius_time_integration() -> None:

    r_arr = np.logspace(13, 22, 100) # cm

    fig = plt.figure()
    t_arr = np.array([give_time_radius_integration(r_, r_w, r_b)
                      for r_ in r_arr])
    t_arr = t_arr/yr
    r_arr = r_arr/pc  # pc

    plt.plot(t_arr, r_arr, label="full integration")

    t_arr = np.logspace(-3, 12) # yr

    r_arr = np.array([give_time_radius_integration2(r_w, r_b, t_)
                     for t_ in t_arr])
    r_arr = r_arr/pc
    
    plt.plot(t_arr, r_arr, label="2nd integration method")


    plt.axhline(y=r_w/pc, color='black', linestyle="--",
                label=r"$r_\mathrm{w}$")
    plt.axhline(y=r_b/pc, color='black', linestyle="-.",
                label=r"$r_\mathrm{b}$")
    plt.ylabel(r"$r$ [pc]")
    plt.xlabel(r"$t$ [yr]")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    fig.tight_layout()
    plt.savefig("Project Summary/Images/R(t)_integration.pdf")
    plt.show()


def plot_speed_time_integration() -> None:

    r_arr = np.logspace(13, 22, 100) # cm

    t_arr = np.array([give_time_radius_integration(r_, r_w, r_b)
                      for r_ in r_arr])
    u_arr = give_speed_radius_analytical(r_arr, r_w, r_b)
    t_arr = t_arr/(1e3*yr)  # kyr

    fig = plt.figure()
    plt.plot(t_arr, u_arr, label=r"Analytical")
    plt.ylabel(r"$u(t)$ [cm$\cdot$s$^{-1}$]")
    plt.xlabel(r"$t$ [kyr]")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(1e-3, 3e1)
    plt.ylim(2e7, 5e9)
    plt.legend()
    plt.grid()
    fig.tight_layout()
    plt.savefig(r"Project Summary/Images/u_s(t).pdf")
    plt.show()


pc = 3e18  # cm/pc
Msol = 1.989e33  # g/Msol
yr = np.pi * 1e7  # s/yr
r_w = 1.5*pc  # cm
r_b = 28*pc  # cm
M_ej = 1*Msol  # g
u_w = 1e6  # cm.s-1
M_loss = 1e-5*Msol/yr  # g.s-1
m_p = 1.6726e-24  # g
rho_b = m_p*1e-2  # g.cm-3
rho_ISM = m_p*0.069  # g.cm-3
xi = 0.1  # fraction of energy that goes into the acceleration of CRs
E_SN = 2.7e50  # erg
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
