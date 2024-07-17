import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pulsar
import shock_speed as shock
import sn_bubble as bubble
import scipy.integrate as inte
import accurate_csm as csm
import cgs
import time

np.set_printoptions(precision=3)
plt.rcParams.update({'font.size': 17})
plt.rcParams["font.family"] = "serif"


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


def make_cluster_IMF(xx:float, M_cutoff:float=2e5*cgs.sun_mass):

    return 1e72*xx**(-2) * np.exp(-xx/M_cutoff)


def give_cluster_mass() -> np.ndarray:
    M = give_random_value(make_cluster_IMF, 1e3*cgs.sun_mass, 1e5*cgs.sun_mass)
    return M


def test_cluster_mass() -> None:
    R = np.geomspace(1e3*cgs.sun_mass, 1e5*cgs.sun_mass, 1000)
    arr = []
    for _ in tqdm(range(1000)):
        arr.append(give_random_value(make_cluster_IMF, 1e3*cgs.sun_mass, 1e5*cgs.sun_mass))

    # Uniform histogram in log x-scale
    _, bins = np.histogram(arr, bins=50)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))

    fig = plt.figure()
    plt.plot(R,
             make_cluster_IMF(R)/inte.quad(make_cluster_IMF, 1e3*cgs.sun_mass, 1e5*cgs.sun_mass)[0])
    plt.hist(arr, histtype="step", density=True, bins=logbins)
    plt.xlabel("$M_\mathrm{cl}$ [M$_\odot$]")
    plt.ylabel("PDF")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(np.min(R), np.max(R))
    fig.tight_layout()
    # plt.savefig(r"Project Summary/Images/f(M_cl).pdf")
    plt.show()


def pick_IMF(m: np.ndarray) -> np.ndarray:
    return 1e1*m**(-2.3)


def pick_IMF_exp(m: np.ndarray) -> np.ndarray:
    return m**(-2.3)*np.exp(-0.35/m)


def test_IMF() -> None:
    M = np.linspace(2, 300)
    arr = []
    for _ in tqdm(range(1000)):
        arr.append(give_random_value(pick_IMF, 2, 150))

    # Uniform histogram in log x-scale
    _, bins = np.histogram(M, bins=50)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))

    fig = plt.figure()
    plt.plot(M, pick_IMF(M)/inte.quad(pick_IMF, 2, 150)
             [0], label=r"Standard IMF with index $\alpha = 2.3$")
    plt.plot(M, pick_IMF_exp(M)/inte.quad(pick_IMF_exp, 2, 150)
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


def make_stellar_population(M_cluster:float=1e4*cgs.sun_mass):

    M_arr = np.array([])

    while np.sum(M_arr) < M_cluster:
        M_arr = np.append(M_arr, give_random_value(pick_IMF, 2, 150)*cgs.sun_mass)

    return M_arr

def test_stellar_population():

    arr = make_stellar_population()/cgs.sun_mass

    # Uniform histogram in log x-scale
    _, bins = np.histogram(arr, bins=50)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))

    fig = plt.figure()
    plt.hist(arr, histtype="step", density=True, bins=logbins,
             label=f"Total mass: {np.sum(arr):.0f} M$_\odot$")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$M$ [M$_\odot$]")
    plt.ylabel(r"Counts")
    plt.legend(fontsize=12)
    fig.tight_layout()
    # plt.savefig(r"Project Summary/Images/f(m).pdf")
    plt.show()

def pick_random_massive_star(stars):

    massive_stars = stars[stars > 8*cgs.sun_mass]

    random_star = np.random.choice(massive_stars)

    return random_star

def give_superbubble_radius(n_ISM:float=100/cgs.cm3,
                            luminosity:float=1e37*cgs.erg/cgs.sec,
                            time:float=10*cgs.Myr) -> float:
    return 174 * (n_ISM)**(-1/5) * (luminosity/1e37)**(1/5) * (time/(10*cgs.Myr))**(3/5) * cgs.pc


class Superbubble:
    def __init__(self, n_ISM = 100/cgs.cm3) -> None:

        self.ism_number_density = n_ISM

        self.cluster_mass = give_random_value(make_cluster_IMF, 
                                              1e3*cgs.sun_mass,
                                              1e5*cgs.sun_mass)
        self.star_masses = make_stellar_population(self.cluster_mass)
        self.stars_number = len(self.star_masses)
        self.mean_luminosity = 1.6e36 *cgs.erg / cgs.sec
        self.total_luminosity = self.stars_number*self.mean_luminosity

    def explode_star(self):

        self.star_mass = pick_random_massive_star(self.star_masses)
        self.explosion_time = bubble.give_MS_time(self.star_mass/cgs.sun_mass)*cgs.year

        self.pulsar_time = np.geomspace(cgs.year, 1*cgs.Gyr)
        self.time_arr = self.explosion_time + self.pulsar_time

        self.bubble_radius_explosion = give_superbubble_radius(self.ism_number_density,
                                                     self.total_luminosity,
                                                     self.explosion_time)

        self.bubble_radius = give_superbubble_radius(self.ism_number_density,
                                                     self.total_luminosity,
                                                     self.time_arr)

    def initialize_pulsar(self):

        self.kick_velocity = pulsar.give_kick_velocity()

        self.radius_pulsar = self.pulsar_time*self.kick_velocity

    def compare_pulsar(self):

        self.initialize_pulsar()
        self.escape_time = pulsar.find_exact_crossing_point(self.bubble_radius,
                                                            self.radius_pulsar,
                                                            self.pulsar_time)
        
    def is_pulsar_inside(self, t:float):

        self.compare_pulsar()

        if t < self.escape_time:
            return True
        else:
            return False
        
    def give_escape_time(self):

        self.initialize_pulsar()
        self.compare_pulsar()

        return self.escape_time
        
    def give_pulsar_population_inside(self, t:float, n_pulsars:int=1000):

        is_inside_arr = np.array([])

        for _ in range(n_pulsars):
            is_inside_arr = np.append(is_inside_arr, self.is_pulsar_inside(t))

        proportion = np.count_nonzero(is_inside_arr)/len(is_inside_arr)*100

        return proportion


def evaluate_one_system():

    superbubble = Superbubble()

    superbubble.explode_star()

    superbubble.give_escape_time()

    return superbubble.escape_time


def plot_escape_time_distribution():

    systems_number = 10000
    escape_times = np.array([])


    for _ in tqdm(range(systems_number)):
        escape_times = np.append(escape_times, evaluate_one_system()/cgs.kyr)

    fig = plt.figure()

    _, bins = np.histogram(escape_times, bins=100)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))

    plt.hist(escape_times, bins=logbins, histtype="step")
    plt.xscale("log")
    plt.xlabel(r"$t_\mathrm{BS}$ [kyr]")
    plt.ylabel(r"Pulsars")
    plt.grid()
    fig.tight_layout()
    plt.savefig("Project Summary/Images/t_BS Superbubble.pdf")
    plt.show()


if __name__ == "__main__":

    # test_cluster_mass()
    # test_IMF()
    # test_stellar_population()

    plot_escape_time_distribution()

    1
