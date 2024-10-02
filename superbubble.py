import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pulsar
import shock_speed as shock
import sn_bubble as bubble
import scipy.integrate as inte
import accurate_csm as csm
import leaving_the_cradle as cradle
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
    for _ in tqdm(range(10000)):
        arr.append(give_random_value(make_cluster_IMF, 1e3*cgs.sun_mass, 1e5*cgs.sun_mass)/cgs.sun_mass)

    # Uniform histogram in log x-scale
    _, bins = np.histogram(arr, bins=50)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))

    fig = plt.figure()
    plt.plot(R/cgs.sun_mass,
             make_cluster_IMF(R)*cgs.sun_mass/inte.quad(make_cluster_IMF, 1e3*cgs.sun_mass, 1e5*cgs.sun_mass)[0])
    plt.hist(arr, histtype="step", density=True, bins=logbins)
    plt.xlabel(r"$M_\mathrm{cl}$ [M$_\odot$]")
    plt.ylabel("PDF")
    plt.xscale("log")
    plt.yscale("log")
    fig.tight_layout()
    plt.savefig(r"Project Summary/Images/f(M_cl).pdf")
    plt.show()


def pick_IMF(m: np.ndarray) -> np.ndarray:
    return 1e1*m**(-2.3)


def pick_IMF_exp(m: np.ndarray) -> np.ndarray:
    return 1*m**(-2.3)*np.exp(-0.35/m)


def test_IMF() -> None:
    M = np.linspace(2, 300)
    arr = []
    for _ in tqdm(range(1000)):
        arr.append(give_random_value(pick_IMF_exp, 2, 150))

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
        M_arr = np.append(M_arr, give_random_value(pick_IMF_exp, 2, 150)*cgs.sun_mass)

    return M_arr

def test_stellar_population():

    arr = make_stellar_population()/cgs.sun_mass

    # Uniform histogram in log x-scale
    _, bins = np.histogram(arr, bins=50)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))

    fig = plt.figure()
    plt.hist(arr, histtype="step", density=True, bins=logbins,
             label=f"Total mass: {np.sum(arr):.0f} "r"M$_\odot$")
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
    massive_stars = massive_stars[massive_stars < 40*cgs.sun_mass]

    random_star = np.random.choice(massive_stars)

    return random_star


def give_total_star_mass_loss_rate(stars):

    mass_loss = 0

    for star in stars/cgs.sun_mass:
        mass_loss += bubble.give_mass_loss_MS(star)*cgs.sun_mass/cgs.year

    return mass_loss


def give_total_star_wind_speed(stars):

    mass_loss_tot = 0
    wind_speed_tot = 0

    for star in stars/cgs.sun_mass:
        typee = "O" if star > 16 else "B"
        mass_loss = bubble.give_mass_loss_MS(star)*cgs.sun_mass/cgs.year
        mass_loss_tot += mass_loss
        wind_speed = bubble.give_wind_speed_type(star, typee)*cgs.km/cgs.sec
        wind_speed_tot += mass_loss*wind_speed

    wind_speed_tot *= mass_loss_tot**(-1)

    return wind_speed_tot


def give_total_luminosity_computed(cluster_mass_loss, cluster_wind_speed):

    cluster_luminosity = cluster_mass_loss*cluster_wind_speed**2/2

    return cluster_luminosity



def give_superbubble_radius(n_ISM:float=100/cgs.cm3,
                            luminosity:float=1e37*cgs.erg/cgs.sec,
                            time:float=10*cgs.Myr) -> float:
    """The 22% rescaling parameter comes from Vieu 2022, to match the 
    observational proofs.

    Args:
        n_ISM (float, optional): ISM density. Defaults to 100/cgs.cm3.
        luminosity (float, optional): total cluster density. Defaults to 1e37*cgs.erg/cgs.sec.
        time (float, optional): age of the cluster. Defaults to 10*cgs.Myr.

    Returns:
        float: radius of the superbubble
    """
    return 174 * (n_ISM)**(-1/5) * (0.22 * luminosity/1e37)**(1/5) * (time/(10*cgs.Myr))**(3/5) * cgs.pc


def give_superbubble_density(M, total_luminosity = 1e37, n_ISM=100):
    """Bubble number density from MacLow and McCray 1988"""
    
    t_MS = bubble.give_MS_time(M/cgs.sun_mass)  # yr
    n_b = 4e-3 * (0.22*total_luminosity/1e38)**(6/35) * (n_ISM)**(19/35) * \
        (t_MS/1e7)**(-22/35)  # cm-3
    return n_b


def give_wind_termination_shock_radius(n_ISM:float=100/cgs.cm3,
                                 cluster_luminosity:float=1e37*cgs.erg/cgs.sec,
                                 cluster_mass_loss:float=1e-4*cgs.sun_mass/cgs.year,
                                 cluster_wind_speed:float=2e3*cgs.km/cgs.sec,
                                 time:float=10*cgs.Myr):
    
    r = 1.3 * (cluster_mass_loss/(1e-5*cgs.sun_mass/cgs.year))**(1/2) * (cluster_wind_speed/1e6)**(1/2) * \
       (cluster_luminosity/1e36)**(-7/35) * (n_ISM)**(-21/70) * (time/(1*cgs.Myr))**(14/35) * cgs.pc
    
    return r


class Superbubble:
    def __init__(self, n_ISM = 100/cgs.cm3) -> None:

        self.ism_number_density = n_ISM

        self.cluster_mass = give_random_value(make_cluster_IMF, 
                                              1e3*cgs.sun_mass,
                                              1e5*cgs.sun_mass)
        self.star_masses = make_stellar_population(self.cluster_mass)
        self.stars_number = len(self.star_masses)
        self.stars_luminosity = bubble.give_wind_luminosity_type(self.star_masses/cgs.sun_mass)
        self.total_luminosity = np.sum(self.stars_luminosity)
        self.total_mass_loss = give_total_star_mass_loss_rate(self.star_masses)
        self.total_wind_speed = give_total_star_wind_speed(self.star_masses)
        #self.computed_luminosity = give_total_luminosity_computed(cluster_mass_loss=self.total_mass_loss,
        #                                                          cluster_wind_speed=self.total_wind_speed)

    def explode_star(self):

        self.star_mass = pick_random_massive_star(self.star_masses)
        self.explosion_time = bubble.give_MS_time(self.star_mass/cgs.sun_mass)*cgs.year

        self.pulsar_time = np.geomspace(cgs.year, 1*cgs.Gyr)
        self.time_arr = self.explosion_time + self.pulsar_time

        self.bubble_radius_explosion = give_superbubble_radius(self.ism_number_density,
                                                     self.total_luminosity,
                                                     self.explosion_time)
        self.superbubble_density = give_superbubble_density(self.star_mass, self.total_luminosity, self.ism_number_density)
        
        self.wind_radius_explosion = give_wind_termination_shock_radius(self.ism_number_density,
                                                                  self.total_luminosity,
                                                                  self.total_mass_loss,
                                                                  self.total_wind_speed,
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


def evaluate_one_system(boundary="bubble"):

    sb = Superbubble()
    sb.explode_star()

    star = bubble.Star(sb.star_mass/cgs.sun_mass)
    wind_density = sb.total_mass_loss/(4*np.pi *  sb.total_wind_speed)/cgs.proton_mass
    system = cradle.PSR_SNR_System(n_ISM=100,
                                m_ej=star.ejected_mass,
                                mass_loss=sb.total_mass_loss,
                                wind_speed=sb.total_wind_speed,
                                wind_radius=sb.wind_radius_explosion,
                                wind_density=wind_density,
                                bubble_radius=sb.bubble_radius_explosion,
                                bubble_density=sb.superbubble_density*cgs.proton_mass)
    
    system.evolve(boundary=boundary)
    system.give_escape_time()

    return system.escape_time


def plot_SNR_in_SB_evolution():

    ratio = []

    systems_number = 10

    for _ in tqdm(range(systems_number)):

        sb = Superbubble()
        sb.explode_star()

        star = bubble.Star(sb.star_mass/cgs.sun_mass)
        wind_density = sb.total_mass_loss/(4*np.pi *  sb.total_wind_speed)/cgs.proton_mass
        system = cradle.PSR_SNR_System(n_ISM=100,
                                    m_ej=star.ejected_mass,
                                    mass_loss=sb.total_mass_loss,
                                    wind_speed=sb.total_wind_speed,
                                    wind_radius=sb.wind_radius_explosion,
                                    wind_density=wind_density,
                                    bubble_radius=sb.bubble_radius_explosion,
                                    bubble_density=sb.superbubble_density*cgs.proton_mass)
        
        system.evolve(boundary="SNR")
        system.give_escape_time()

        ratio.append(system.merger_radius/sb.bubble_radius_explosion)

    ratio = np.array(ratio)

    fig = plt.figure()
    plt.hist(ratio, histtype="step", bins=50, weights=np.ones_like(ratio) / len(ratio), label="")
    plt.xlabel(r"Merger radius/Bubble radius [n.u.]")  
    plt.ylabel("Proportion of clusters")
    plt.grid()
    fig.tight_layout()
    # plt.savefig("Project Summary/SB_plots/Merger radius over bubble radius.pdf")
    # plt.savefig("SB_plots/Merger radius over bubble radius.pdf")
    plt.show()
    


def plot_mass_loss_rate_distribution():
    
    mass_loss = []

    systems_number = 1000

    for _ in tqdm(range(systems_number)):
        system = Superbubble()
        mass_loss.append(system.total_mass_loss/(cgs.sun_mass/cgs.year))

    mass_loss = np.array(mass_loss)

    _, bins = np.histogram(mass_loss, bins=50)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))

    fig = plt.figure()
    plt.hist(mass_loss, histtype="step", bins=logbins, weights=np.ones_like(mass_loss) / len(mass_loss), label="")
    plt.xscale("log")
    plt.xlabel(r"Mass loss [M$_\odot$/yr]")  
    plt.ylabel("Proportion of clusters")
    plt.grid()
    fig.tight_layout()
    plt.savefig("Project Summary/SB_plots/Mass Loss.pdf")
    plt.savefig("SB_plots/Mass Loss.pdf")
    plt.show()


def plot_wind_speed_distribution():
    
    wind_speed = []

    systems_number = 1000

    for _ in tqdm(range(systems_number)):
        system = Superbubble()
        wind_speed.append(system.total_wind_speed/cgs.km)

    wind_speed = np.array(wind_speed)

    fig = plt.figure()
    plt.hist(wind_speed, histtype="step", bins=50, weights=np.ones_like(wind_speed) / len(wind_speed), label="")
    plt.xlabel("Wind speed [km/s]")  
    plt.ylabel("Proportion of clusters")
    plt.grid()
    fig.tight_layout()
    plt.savefig("Project Summary/SB_plots/Wind Speed.pdf")
    plt.savefig("SB_plots/Wind Speed.pdf")
    plt.show()


def plot_superbubble_density_distribution():
    
    density = []

    systems_number = 1000

    for _ in tqdm(range(systems_number)):
        sb = Superbubble()
        sb.explode_star()
        density.append(sb.superbubble_density)

    density = np.array(density)

    fig = plt.figure()
    plt.hist(density, histtype="step", bins=50, weights=np.ones_like(density) / len(density), label="")
    plt.xlabel("Superbubble density [cm$^{-3}$]")  
    plt.ylabel("Proportion of clusters")
    plt.grid()
    fig.tight_layout()
    plt.savefig("Project Summary/SB_plots/Superbubble density.pdf")
    plt.savefig("SB_plots/Superbubble density.pdf")
    plt.show()


def plot_luminosity_distribution():
    
    luminosity_computed = []
    luminosity_addition = []

    systems_number = 1000

    for _ in tqdm(range(systems_number)):
        system = Superbubble()
        luminosity_addition.append(system.total_luminosity)
        luminosity_computed.append(system.computed_luminosity)

    luminosity_addition = np.array(luminosity_addition)
    luminosity_computed = np.array(luminosity_computed)

    _, bins = np.histogram(luminosity_addition, bins=50)
    logbins1 = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))

    _, bins = np.histogram(luminosity_computed, bins=50)
    logbins2 = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))

    fig = plt.figure()
    plt.hist(luminosity_addition, histtype="step", bins=logbins1, weights=np.ones_like(luminosity_addition) / len(luminosity_addition), label="Addition")
    plt.hist(luminosity_computed, histtype="step", bins=logbins2, weights=np.ones_like(luminosity_computed) / len(luminosity_computed), label="Computed")
    plt.xlabel("Cluster luminosity [erg/s]")  
    plt.ylabel("Proportion of clusters")
    plt.xscale("log")
    plt.grid()
    plt.legend()
    fig.tight_layout()
    plt.savefig("Project Summary/SB_plots/Cluster Luminosity.pdf")
    plt.savefig("SB_plots/Cluster Luminosity.pdf")
    plt.show()


def plot_escape_time_distribution():

    systems_number = 10000
    escape_times = np.array([])
    boundaries = ["SNR", "bubble"]

    fig = plt.figure()

    for boundary in boundaries:
        file = open(f"Escape Times/Superbubble {boundary}.csv", "w")

        for _ in tqdm(range(systems_number)):
            escape_time = evaluate_one_system(boundary=boundary)/cgs.kyr
            file.write(f"{escape_time}\n")
            escape_times = np.append(escape_times, escape_time)

        _, bins = np.histogram(escape_times, bins=100)
        logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))

        plt.hist(escape_times, bins=logbins, histtype="step", label=boundary)
    plt.xscale("log")
    plt.xlabel(r"$t_\mathrm{BS}$ [kyr]")
    plt.ylabel(r"Pulsars")
    plt.grid()
    plt.legend()
    fig.tight_layout()
    # plt.savefig("Project Summary/Images/t_BS Superbubble.pdf")
    plt.show()


def plot_SB_radius_distribution():
    
    radius = []

    systems_number = 1000

    for _ in tqdm(range(systems_number)):
        sb = Superbubble()
        sb.explode_star()
        radius.append(sb.bubble_radius_explosion/cgs.pc)

    radius = np.array(radius)

    fig = plt.figure()
    plt.hist(radius, histtype="step", bins=50, weights=np.ones_like(radius) / len(radius), label="")
    plt.xlabel("Superbubble radius [pc]")
    plt.ylabel("Proportion of clusters")
    plt.grid()
    fig.tight_layout()
    plt.savefig("Project Summary/Images/Superbubble radius.pdf")
    plt.savefig("CSM_plots/Superbubble radius.png")
    plt.savefig("CSM_plots/Superbubble radius.pdf")
    plt.show()


def plot_wind_radius_distribution():
    
    radius = []

    systems_number = 1000

    for _ in tqdm(range(systems_number)):
        sb = Superbubble()
        sb.explode_star()
        radius.append(sb.wind_radius_explosion/cgs.pc)

    radius = np.array(radius)

    fig = plt.figure()
    plt.hist(radius, histtype="step", bins=50, weights=np.ones_like(radius) / len(radius), label="")
    plt.xlabel("Wind termination shock radius [pc]")
    plt.ylabel("Proportion of clusters")
    plt.grid()
    fig.tight_layout()
    plt.savefig("Project Summary/SB_plots/Wind termination shock radius.pdf")
    plt.savefig("SB_plots/Wind termination shock radius.pdf")
    plt.show()



if __name__ == "__main__":

    # test_cluster_mass()   
    # test_IMF()
    # test_stellar_population()
    plot_SB_radius_distribution()
    # plot_mass_loss_rate_distribution()
    # plot_wind_speed_distribution()
    # plot_luminosity_distribution()
    # plot_wind_radius_distribution()
    # plot_superbubble_density_distribution()

    # plot_SNR_in_SB_evolution()

    # plot_escape_time_distribution()


    1
