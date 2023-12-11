import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as inte
import random
import sn_bubble as SN

plt.rcParams.update({'font.size': 17})
plt.rcParams["font.family"] = "serif"


def make_gaussian(zz: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    r"""
    Returns the gaussian function of 'zz',
    for the mean 'mu' and variance 'sigma'

    -----
    Input:
    -----
    zz    : np.ndarray, point for which we want the Gaussian
    mu    : float, mean of the Gaussian
    sigma : float, variance of the Gaussian

    -----
    Output:
    -----
    y     : float, Gaussian value
    """
    return 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-(zz-mu)**2/(2*sigma**2))

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
    return 1/np.sqrt(2*np.pi*normal_std * zz**2)\
            * np.exp(-(np.log(zz)-normal_mean)**2/(2*normal_std**2))

def give_E_SN(n_: int = 1) -> np.ndarray:
    """Returns an array of length `n` for the SN energy, following
    Lehay et al. 2020.

    Args:
        n_ (int): Desired length of the final energy array

    Returns:
        np.ndarray: erg, E_SN array
    """
    mu = 2.7e50 # erg
    sigma = 3.5

    normal_std = np.log10(sigma)
    normal_mean = np.log(mu)

    result = np.random.lognormal(normal_mean, normal_std, size=n_)

    return result


def give_n_ISM(n_: int = 1) -> np.ndarray:
    """Returns an array of length `n` for the ISM density, following
    Lehay et al. 2020.

    Args:
        n_ (int): Desired length of the final density array

    Returns:
        np.ndarray: cm-3, n_ISM array
    """
    mu = 0.069 # cm-3
    sigma = 5.1

    normal_std = np.log10(sigma)
    normal_mean = np.log(mu)

    result = np.random.lognormal(normal_mean, normal_std, size=n_)

    return result


def make_maxwellian(vv: float, sigma: float) -> float:
    return np.sqrt(2/np.pi)*sigma**(-3)*vv**2 * np.exp(-(vv)**2/(2*sigma**2))


def give_SNR_radius(
        t: np.ndarray = 100e3,
        E: float = 2.7e50,
        n: float = 0.069
) -> np.ndarray:
    """Naive interpretation of the SNR, time in yr"""
    # ST phase, IMPLEMENT SNOWPLOUGH PHASE
    r = 0.3 * (E/1e51)**(1/5) * (n/1)**(-1/5) * (t/1)**(2/5)
    return r  # pc


def give_random_value(function, min: float, max: float) -> float:
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
        if x2 <= function(y):
            return y


def give_z(n_: int) -> np.ndarray:
    """Parametrization from Steiman-Cameron et al. 2010"""
    sigma_z = 70  # pc
    zz = np.random.normal(0, sigma_z, int(n_))
    return zz * 1e-3  # kpc


def pick_density_profile(R: np.ndarray) -> np.ndarray:
    """Parametrization from Lorimer et al. 2006 Model C (clumpy)"""
    A, B, C = 41, 1.9, 5.0
    R_sun = 8.5  # kpc
    norm = 1e-3  # optimum normalization parameter
    # for the value selection process
    return norm * A * (R/R_sun)**B * np.exp(-C*(R-R_sun)/R_sun)


def pick_kick_velocity_component(v: np.ndarray) -> np.ndarray:
    """Parametrization from Faucher-Giguère et al. 2006 eq 7"""
    w, sigma_v1, sigma_v2 = 0.90, 160, 780
    norm = 100  # optimum normalization parameter
    # for the value selection process
    return (w*make_gaussian(v, mu=0, sigma=sigma_v1)
            + (1-w)*make_gaussian(v, mu=0, sigma=sigma_v2)) * norm


def give_kick_velocity() -> np.ndarray:
    v_x = give_random_value(pick_kick_velocity_component, -2000, 2000)
    v_y = give_random_value(pick_kick_velocity_component, -2000, 2000)
    v_z = give_random_value(pick_kick_velocity_component, -2000, 2000)
    return np.linalg.norm([v_x, v_y, v_z])


def pick_initial_period(P0: float) -> np.ndarray:
    """Parametrization from Evoli et al. 2021"""
    mu_P0, sigma_P0 = 100, 50  # ms
    return make_gaussian(P0, mu_P0, sigma_P0)


def pick_arm() -> str:
    density_sagittarius = 169    # 1
    density_scutum = 266         # 2
    density_perseus = 339        # 3
    density_norma = 176          # 4
    return random.choices(
        ['sagittarius', 'scutum', 'perseus', 'norma'],
        weights=[density_sagittarius,
                 density_scutum,
                 density_perseus,
                 density_norma])[0]


def give_arm_angle(r: float = 3) -> float:
    """Parametrization from Steiman-Cameron et al. 2010"""
    arm = pick_arm()
    if arm == "perseus":
        a, b = 0.449, 0.249
    elif arm == "scutum":
        a, b = 0.608, 0.279
    elif arm == "norma":
        a, b = 0.378, 0.240
    elif arm == "sagittarius":
        a, b = 0.246, 0.242
    return np.log(r/a)/b - np.pi/2  # radians


def give_bar_angle() -> float:
    """Parametrization of Churchwell et al. 2009"""
    bar_angle = 90-20  # degrees
    return random.choice([bar_angle*np.pi/180,
                          bar_angle*np.pi/180+np.pi]) - np.pi/2  # radians


def convert_cartesian(r: float, phi: float) -> np.ndarray:
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    return x, y


def blur_phi(phi: np.ndarray) -> np.ndarray:
    """Dispersion around the arm line"""
    return np.random.normal(phi, 15*np.pi/180)


def give_ST_radius(t: np.ndarray) -> np.ndarray:
    """t in yr, from ST paper"""
    m_p = 1.6726e-24  # g
    E = 2.7e50  # erg.s-1
    rho_ISM = m_p * 0.069  # g.cm-3
    t = t * np.pi * 1e7  # s
    r = (2.026*E/rho_ISM)**(1/5)*t**(2/5)
    return r/3e18  # pc


def is_pulsar_in_SNR(
        vk: np.ndarray,
        r_SNR: float = 30,
        t: float = 100e3
) -> np.ndarray:
    # Unit conversions
    vk = vk*1e5/(3e18)*(np.pi*1e7)  # conversion from km/s to pc/yr
    return vk*t < r_SNR


def test_density_profile() -> None:
    R = np.linspace(0, 20, 1000)
    arr = []
    for _ in range(1000):
        arr.append(give_random_value(pick_density_profile, 0, 20))

    fig = plt.figure()
    plt.plot(R,
             pick_density_profile(R)/inte.quad(pick_density_profile, 0, 20)[0])
    plt.hist(arr, histtype="step", density=True, bins=50)
    plt.xlabel("$r$ [kpc]")
    plt.ylabel("PDF")
    plt.xlim(np.min(R), np.max(R))
    fig.tight_layout()
    plt.savefig(r"Project Summary/Images/f(R).pdf")
    # plt.show()


def test_ST_radius() -> None:
    t = np.logspace(np.log10(1), np.log10(3e5), 1000)
    r = give_ST_radius(t)

    fig = plt.figure()
    plt.plot(t, r)
    plt.xlabel("$t$ [yr]")
    plt.ylabel("$R(t)$ [pc]")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(np.min(t), np.max(t))
    fig.tight_layout()
    plt.savefig(r"Project Summary/Images/R(t)_ST.pdf")
    # plt.show()


def test_E_SN() -> None:
    z = np.logspace(48, 52, 100)  # erg
    arr = give_E_SN(10000)
    print(arr)

    # Uniform histogram in log x-scale
    _, bins = np.histogram(arr, bins=50)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))

    fig = plt.figure()
    plt.plot(z, make_lognormal(z, 2.7e50, 3.5)/inte.quad(
    lambda x: make_lognormal(x, 2.7e50, 3.5), 1e48, 1e52)[0])
    plt.hist(arr, histtype="step", density=True, bins=logbins)
    plt.xlabel(r"$E_\mathrm{SN}$ [erg]")
    plt.ylabel("PDF")
    plt.xlim(np.min(z), np.max(z))
    plt.xscale("log")
    fig.tight_layout()
    plt.savefig(r"Project Summary/Images/f(E_SN).pdf")
    plt.show()


def test_n_ISM() -> None:
    z = np.logspace(-4, 1, 1000)  # cm-3
    arr = give_n_ISM(1000)

    # Uniform histogram in log x-scale
    _, bins = np.histogram(arr, bins=50)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))

    fig = plt.figure()
    plt.plot(z, make_lognormal(z, 0.069, 5.1)/inte.quad(
        lambda x: make_lognormal(x, 0.069, 5.1), 1e-4, 1e1)[0])
    plt.hist(arr, histtype="step", density=True, bins=logbins)
    plt.xlabel(r"$n_\mathrm{ISM}$ [cm$^{-3}$]")
    plt.ylabel("PDF")
    plt.xlim(np.min(z), np.max(z))
    plt.xscale("log")
    fig.tight_layout()
    plt.savefig(r"Project Summary/Images/f(n_ISM).pdf")
    plt.show()


def test_z() -> None:
    z = np.linspace(-300e-3, 300e-3, 1000)  # kpc
    arr = give_z(1000)

    fig = plt.figure()
    plt.plot(z, make_gaussian(z, 0, 70e-3)/inte.quad(
        lambda x: make_gaussian(x, 0, 70e-3), -300e-3, 300e-3)[0])
    plt.hist(arr, histtype="step", density=True, bins=50)
    plt.xlabel("$z$ [kpc]")
    plt.ylabel("PDF")
    plt.xlim(np.min(z), np.max(z))
    fig.tight_layout()
    plt.savefig(r"Project Summary/Images/f(z).pdf")
    # plt.show()


def test_initial_period() -> None:
    P = np.linspace(0, 300, 1000)
    arr = []
    for _ in range(1000):
        arr.append(give_random_value(pick_initial_period, 0, 300))

    fig = plt.figure()
    plt.plot(P, pick_initial_period(P) /
             inte.quad(pick_initial_period, 0, 300)[0])
    plt.hist(arr, histtype="step", density=True, bins=50)
    plt.xlabel(r"$P_0$ [ms]")
    plt.ylabel("PDF")
    plt.xlim(np.min(P), np.max(P))
    fig.tight_layout()
    plt.savefig(r"Project Summary/Images/f(P0).pdf")
    # plt.show()


def test_kick_velocity() -> None:
    arr = []
    for _ in range(10000):
        arr.append(give_kick_velocity())

    fig = plt.figure()
    plt.hist(arr, histtype="step", density=True, bins=50)
    plt.xlabel(r"$v_\mathrm{k}$ [km$\cdot$s$^{-1}$]")
    plt.ylabel("PDF")
    plt.xlim(np.min(arr), np.max(arr))
    fig.tight_layout()
    plt.savefig(r"Project Summary/Images/f(vk).pdf")
    # plt.show()


def test_pick_arm() -> None:
    arr = []
    N = 10000
    for _ in range(N):
        arr.append(pick_arm())

    fig = plt.figure()
    plt.hist(arr, histtype="step")
    fig.tight_layout()
    plt.savefig(r"Project Summary/Images/pick_arm.pdf")
    # plt.show()


def create_galactic_coordinates(n_: int, plot: bool = True) -> None:

    x_arr_, y_arr_ = np.array([]), np.array([])

    for _ in range(int(n_)):
        r_ = give_random_value(pick_density_profile, 0, 20)
        if r_ >= 2.3:
            arr = blur_phi(give_arm_angle(r_))
            x_arr_ = np.append(x_arr_, convert_cartesian(r_, arr)[0])
            y_arr_ = np.append(y_arr_, convert_cartesian(r_, arr)[1])
        else:
            r_ = np.random.uniform(0, 3)
            arr = give_bar_angle()
            x_arr_ = np.append(x_arr_, blur_phi(convert_cartesian(r_, arr)[0]))
            y_arr_ = np.append(y_arr_, blur_phi(convert_cartesian(r_, arr)[1]))

    if plot:
        fig = plt.figure()
        plt.plot(x_arr_, y_arr_, linewidth=0, marker=".", markersize=0.5)
        plt.plot(8.5, 0, linewidth=0, marker=".", color="red")
        plt.xlim([-20, 20])
        plt.ylim([-20, 20])
        plt.xlabel(r"$x$ [kpc]")
        plt.ylabel(r"$y$ [kpc]")
        plt.gca().set_aspect('equal')
        fig.tight_layout()
        plt.savefig(r"Project Summary/Images/galaxy.pdf")
        # plt.show()

    return x_arr_, y_arr_


def create_file_zero(
        t: float = 100e3,
        pulsar_rate: float = 1/100,
        name: int = None
) -> None:
    x_arr, y_arr, z_arr, P0_arr, vk_arr, is_inside_arr = \
        np.array([]), np.array([]), np.array([]), np.array([]), np.array([]),\
            np.array([], dtype="bool")
    age_arr = np.arange(0, t, 1/pulsar_rate)
    n = len(age_arr)
    r_SNR = give_ST_radius(t=age_arr)

    for i in range(n):
        P0_arr = np.append(P0_arr,
                           give_random_value(pick_initial_period, 0, 300))
        vk_arr = np.append(vk_arr, give_kick_velocity())
        is_inside_arr = np.append(
            is_inside_arr,
            is_pulsar_in_SNR(vk_arr[-1], r_SNR=r_SNR[i], t=age_arr[i]))

    x_arr, y_arr = create_galactic_coordinates(n_=n, plot=False)
    z_arr = give_z(n_=n)
    print("{:.2f}% pulsars are within their SNR at 100 kyr"
          .format(np.count_nonzero(is_inside_arr)/len(is_inside_arr)*100))

    # Writing
    if name is not None:
        if t == 100e3:
            file = open("Galaxies/Pulsars_0_{}.csv".format(name), "w")
        else:
            file = open("Age Evolution/Pulsars_0_{}_{}_kyr.csv"
                        .format(name, t/1e3), "w")
    else:
        file = open("Age Evolution/Pulsar_characteristics_{}_kyr_0.csv"
                    .format(t/1e3), "w")
    file.write("x [kpc], y [kpc], z [kpc], P0 [ms], vk [km.s-1], r_SNR [pc],\
                is_inside\n")

    for i in range(int(n)):
        file.write("{x:.2f}, {y:.2f}, {z:.2f}, {PO:.2f}, {vk:.2f}, {r_SNR},\
                   {isinside}\n"
                   .format(x=x_arr[i], y=y_arr[i], z=z_arr[i],
                           PO=P0_arr[i], vk=vk_arr[i], r_SNR=r_SNR[i],
                           isinside=is_inside_arr[i]))

    file.close()


def give_is_inside_proportion(
        t: float = 100e3,
        n: int = 100,
        phase: str = "ST",
        variable_parameters: bool = False,
        E = 2.7e50,
        n_ = 0.069
) -> float:
    """Choose phase between "ST" and "PDS"
    (for pressure driven snowplough)."""
    vk_arr, is_inside_arr = np.array([]), np.array([])

    if variable_parameters:
        E_SN = give_E_SN()
        n_ISM = give_n_ISM()
        if phase == "ST":
            r_SNR = give_SNR_radius(t=t, E=E_SN, n=n_ISM)  # Only with ST
        elif phase == "PDS":
            # With snowplough phase and merging time
            r_SNR = SN.give_SN_radius(t=t, E=E_SN, n=n_ISM)
    else:
        if phase == "ST":
            r_SNR = give_SNR_radius(t=t, E=E, n=n_)  # Only with ST
        elif phase == "PDS":
            # With snowplough phase and merging time
            r_SNR = SN.give_SN_radius(t=t, E=E, n=n_)

    for n_ in range(n):
        vk_arr = np.append(vk_arr, give_kick_velocity())
        is_inside_arr = np.append(
            is_inside_arr, is_pulsar_in_SNR(vk_arr[-1], r_SNR=r_SNR, t=t))

    proportion = np.count_nonzero(is_inside_arr)/len(is_inside_arr)*100

    return proportion


if __name__ == "__main__":
    # TESTS

    # test_z()
    # test_ST_radius()
    # test_density_profile()
    # test_initial_period()
    # test_kick_velocity()
    # test_pick_arm()

    test_E_SN()
    test_n_ISM()

    # create_galactic_coordinates(1e4)

    # create_file_zero()

    1
