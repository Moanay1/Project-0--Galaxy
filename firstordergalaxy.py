# We use the same functions as the 0 order galaxy, but instead of using just the ST phase of the SNR, we use an accurate description of the shock in the medium surrounding the pulsar, carved by stellar winds.

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
# Custom packages
import zeroordergalaxy as zero
import sn_bubble as snr
import shock_speed


def give_SNR_radius(
        t_age: np.ndarray,
        M_MS: np.ndarray = None,
        E_SN: float = 2.7e50,
        M_ej: float = 1
) -> np.ndarray:
    """Gives the SNR radius in pc.
    Use the wind and bubble radii at a certain time for a pulsar of age
    t_age (yr). If M_MS is not None,
    it must be the same dimension as t_age."""
    if M_MS is None:
        M_MS = []
        for _ in range(len(t_age)):
            M_MS.append(snr.give_random_value(snr.pick_IMF, 2, 300))
        M_MS = np.asarray(M_MS)
    else:
        if len(t_age) != len(M_MS):
            print("len(t_age) and len(M_MS) must be of same dimension!")
            exit()
    r_w = snr.give_wind_radius(M_MS)
    r_b = snr.give_bubble_radius(M_MS)

    radii2 = np.array([])

    for i in tqdm(range(len(t_age))):
        radii2 = np.append(
            radii2,
            shock_speed.give_time_radius_integration2(r_w[i]*pc, r_b[i]*pc,
                                                      t_age[i], E_SN, M_ej))

    return radii2/pc


def test_SNR_radius(age_arr: np.ndarray, r: np.ndarray) -> None:

    fig = plt.figure()
    plt.plot(age_arr, r, label=r"From integration")
    plt.plot(age_arr, zero.give_ST_radius(age_arr), label=r"Theoretical ST")
    plt.grid()
    plt.xlabel(r"$t_\mathrm{age}$ [yr]")
    plt.ylabel(r"$r$ [pc]")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    fig.tight_layout()
    # plt.savefig(r"Project Summary/Images/R(t)_comparison.pdf")
    # plt.show()


def test_SNR_radius_constant_mass(
        mass: np.ndarray,
        t: float = 100e3,
        pulsar_rate: float = 1/100
) -> None:

    fig = plt.figure()
    for mass_ in mass:
        age_arr = np.arange(0, t, 1/pulsar_rate)
        mass_arr = np.full(len(age_arr), mass_)
        r_arr = give_SNR_radius(age_arr, mass_arr)
        plt.plot(age_arr, r_arr, label=r"$m = {}$ M$_\odot$".format(mass_))

    plt.plot(age_arr, zero.give_ST_radius(age_arr), color="black",
             linewidth=2, linestyle="--", label=r"Theoretical ST")
    plt.grid()
    plt.xlabel(r"$t_\mathrm{age}$ [yr]")
    plt.ylabel(r"$r$ [pc]")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(fontsize=12)
    fig.tight_layout()
    plt.savefig(r"Project Summary/Images/R(t)_mass_comparison2.pdf")
    plt.show()


def test_SNR_radius_varying_energy(
        E_SN: np.ndarray,
        mass: float = 5,
        t: float = 100e3,
        pulsar_rate: float = 1/100
) -> None:

    fig = plt.figure()
    for E_ in E_SN:
        age_arr = np.arange(0, t, 1/pulsar_rate)
        mass_arr = np.full(len(age_arr), mass)
        r_arr = give_SNR_radius(age_arr, mass_arr, E_)
        plt.plot(age_arr, r_arr, label=r"$E_\mathrm{SN} = "f"{E_}"r"$ erg")

    plt.plot(age_arr, zero.give_ST_radius(age_arr), color="black",
             linewidth=2, linestyle="--", label=r"Theoretical ST")
    plt.grid()
    plt.xlabel(r"$t_\mathrm{age}$ [yr]")
    plt.ylabel(r"$r$ [pc]")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(fontsize=12)
    fig.tight_layout()
    plt.savefig(r"Project Summary/Images/R(t)_energy_comparison.pdf")
    plt.show()


def test_SNR_radius_varying_ejecta_mass(
        M_ej: np.ndarray,
        mass: float = 5,
        t: float = 100e3,
        pulsar_rate: float = 1/100
) -> None:
    fig = plt.figure()
    for M_ in M_ej:
        age_arr = np.arange(0, t, 1/pulsar_rate)
        mass_arr = np.full(len(age_arr), mass)
        r_arr = give_SNR_radius(age_arr, mass_arr, M_ej=M_)
        plt.plot(age_arr, r_arr,
                 label=r"$M_\mathrm{ej} = "f"{M_}"r"$ M$_\odot$")

    plt.plot(age_arr, zero.give_ST_radius(age_arr), color="black",
             linewidth=2, linestyle="--", label=r"Theoretical ST")
    plt.grid()
    plt.xlabel(r"$t_\mathrm{age}$ [yr]")
    plt.ylabel(r"$r$ [pc]")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(fontsize=12)
    fig.tight_layout()
    plt.savefig(r"Project Summary/Images/R(t)_ejecta_mass_comparison.pdf")
    plt.show()


def create_file_first(
        t: float = 100e3,
        pulsar_rate: float = 1/100,
        name: int = None
) -> None:
    x_arr, y_arr, z_arr, P0_arr, vk_arr, is_inside_arr =\
        np.array([]), np.array([]), np.array([]), np.array([]), np.array([]),
    np.array([], dtype="bool")
    age_arr = np.arange(0, t, 1/pulsar_rate)
    n = len(age_arr)
    r_SNR = give_SNR_radius(age_arr)  # pc

    test_SNR_radius(age_arr, r_SNR)

    for i in range(n):
        P0_arr = np.append(
            P0_arr, zero.give_random_value(zero.pick_initial_period, 0, 300))
        vk_arr = np.append(vk_arr, zero.give_kick_velocity())
        is_inside_arr = np.append(
            is_inside_arr,
            zero.is_pulsar_in_SNR(vk_arr[-1], r_SNR=r_SNR[i], t=age_arr[i]))

    x_arr, y_arr = zero.create_galactic_coordinates(n_=n, plot=False)
    z_arr = zero.give_z(n_=n)
    print("{:.2f}% pulsars are within their SNR at 100 kyr"
          .format(np.count_nonzero(is_inside_arr)/len(is_inside_arr)*100))

    # Writing
    if name is not None:
        if t == 100e3:
            file = open("Galaxies/Pulsars_1_{}.csv".format(name), "w")
        else:
            file = open("Age Evolution/Pulsars_1_{}_{}_kyr.csv"
                        .format(name, t/1e3), "w")
    else:
        file = open("Age Evolution/Pulsar_characteristics_{}_kyr_1.csv"
                    .format(t/1e3), "w")
    file.write(
        "x [kpc], y [kpc], z [kpc], P0 [ms], vk [km.s-1], r_SNR [pc],\
             is_inside\n")

    for i in range(int(n)):
        file.write("{x:.2f}, {y:.2f}, {z:.2f}, {PO:.2f}, {vk:.2f}, {r_SNR},\
                    {isinside}\n".format(x=x_arr[i], y=y_arr[i],
                                         z=z_arr[i], PO=P0_arr[i],
                                         vk=vk_arr[i], r_SNR=r_SNR[i],
                                         isinside=is_inside_arr[i]))

    file.close()


pc = 3e18  # cm/pc

if __name__ == "__main__":

    # test_SNR_radius_constant_mass([5, 10, 25, 50, 75])
    # test_SNR_radius_varying_energy(np.array([0.1, 0.5, 1, 2, 5])*1e51)
    test_SNR_radius_varying_ejecta_mass([0.5, 1, 2, 3, 4])

    # create_file_first()

    1
