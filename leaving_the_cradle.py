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



def convergence_radius():

    n_array = 2**np.arange(1, 20, 1)

    r_w = 1.5*cgs.pc
    r_b = 25*cgs.pc

    fig = plt.figure()

    r_arr = np.geomspace(0.01*cgs.pc, 1000*cgs.pc, num=n_array[-1]) # cm
    t_arr_reference = np.array([shock.give_time_radius_integration(r_, r_w, r_b)
                      for r_ in tqdm(r_arr)])/cgs.year
    integral_t_arr_reference = inte.simpson(t_arr_reference, r_arr)
    
    
    difference = []

    for n_ in tqdm(n_array):
        r_arr = np.geomspace(0.01*cgs.pc, 1000*cgs.pc, n_) # cm
        t_arr = np.array([shock.give_time_radius_integration(r_, r_w, r_b)
                      for r_ in r_arr])/cgs.year
        integral_t_arr = inte.simpson(t_arr, r_arr)
        diff = (integral_t_arr - integral_t_arr_reference)/integral_t_arr_reference
        difference.append(diff)

    difference = np.abs(difference)

    print(n_array)
    print(r"%%%%%%%%%%%%%%%%%%%")
    print(difference)

    plt.plot(n_array, difference)

    plt.xlabel("Number of points in r_arr")
    plt.ylabel(r"$\frac{I - I_\mathrm{expected}}{I_\mathrm{expected}}$")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()
    fig.tight_layout()
    # plt.savefig("Project Summary/Images/Convergence_radius_points.pdf")
    plt.show()



def cooling_time(temperature:float = 8000, #K
                 number_density:float = 1 #cm-3
                 ):
    cooling_function = 1e-22 * (temperature/1e6)**(-0.7) # erg/cm3/s # MacLow & McCray 1988
    return 3/2 * cgs.k_boltzmann * temperature / (number_density * cooling_function)


def density_profile(r:np.ndarray,
                    r_w:float = 1.5*cgs.pc,
                    r_b:float = 25*cgs.pc,
                    number_density:float = 1/cgs.cm3,
                    n_bubble:float = 0.01/cgs.cm3,
                    r_shell:float = 0.3*cgs.pc
                    ):
    
    density_arr = []

    for r_ in r:
        if r_ < r_w:
            n = 1e-5*cgs.sun_mass/cgs.year/(4*np.pi * 1e6) \
                /cgs.proton_mass * (r_)**(-2) # /cm3
        elif r_ < r_b:
            # n = n_bubble # Simple solution
            n = n_bubble*((1 - r_/r_b) / (1 - r_w/r_b))**(-2/5) # From Weaver+1977
        elif r_ < r_b+r_shell:
            n = 17
        else:
            n = number_density
        density_arr.append(n)
    
    return np.array(density_arr)


def temperature_profile(r:np.ndarray,
                        r_w:float = 1.5*cgs.pc,
                        r_b:float = 25*cgs.pc,
                        r_shell:float = 0.3*cgs.pc
                        ):
    
    temperature_arr = []

    for r_ in r:
        if r_ < r_w:
            T = 1e4*cgs.K
        elif r_ < r_b:
            # T = 1e6*cgs.K # From Recchia+2022
            T = 1e6*cgs.K*((1 - r_/r_b) / (1 - r_w/r_b))**(2/5) # From Weaver+1977
        elif r_ < r_b + r_shell:
            T = 8000*cgs.K
        else:
            T = 8000*cgs.K
        temperature_arr.append(T)
    
    return np.array(temperature_arr)


def test_density_temperature_profile():

    r_arr = np.geomspace(0.1*cgs.pc, 100*cgs.pc, 1000)

    time_array = np.array([shock.give_time_radius_integration(r_, 1.5*cgs.pc, 25*cgs.pc)
                      for r_ in r_arr])

    fig, (ax1, ax3) = plt.subplots(2, 1, sharex=True)

    color = 'tab:red'
    ax1.set_ylabel(r"Density [cm$^{-3}$]", color=color)
    ax1.plot(r_arr/cgs.pc, density_profile(r_arr), color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xscale("log")
    ax1.set_yscale("log")

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(r"Temperature [K]", color=color)  # we already handled the x-label with ax1
    ax2.plot(r_arr/cgs.pc, temperature_profile(r_arr), color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_yscale("log")

    ax1.axvline(x=1.5, linestyle="--", alpha=0.5, linewidth=1,
                color="black", label="Wind radius")
    ax1.axvline(x=25, linestyle="-.", alpha=0.5, linewidth=1,
                color="black", label="Bubble radius")
    ax1.grid()
    ax1.legend(fontsize=10)


    ax3.plot(r_arr/cgs.pc, cooling_time(temperature_profile(r_arr), density_profile(r_arr))/cgs.kyr, label="Cooling time")
    ax3.plot(r_arr/cgs.pc, time_array/cgs.kyr, label=r"$t(R)$")
    ax3.set_xlabel("Radius [pc]")
    ax3.set_ylabel("Time [kyr]")
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.legend(fontsize=10)
    ax3.grid()
    fig.tight_layout()
    plt.savefig("Project Summary/CSM_plots/temperature density profiles.pdf")
    plt.show()


def sound_speed(temperature: float = 8000) -> float:
    """Temperature of the ISM may vary depending on the region
    (dust, cloud, gas, etc)"""
    mu_p = 1.4 # molecular fraction in the ISM
    gamma = 5/3  # adiabatic coefficient for monoatomic gas
    return np.sqrt(gamma*cgs.k_boltzmann*temperature/(mu_p*cgs.proton_mass))  # cm/s


def test_sound_speed():

    system = PSR_SNR_System(n=10000, m_ej=5*cgs.sun_mass)
    system.give_time()
    system.radiative_phase()

    r_arr = system.radius_arr

    fig = plt.figure()

    plt.plot(r_arr/cgs.pc, sound_speed(temperature=temperature_profile(r_arr))/cgs.km, label="Sound speed")
    plt.plot(r_arr/cgs.pc, system.speed_arr/cgs.km, label="Shock speed")
    plt.axvline(x=1.5, linestyle="--", alpha=0.5, linewidth=1,
                color="black", label="Wind radius")
    plt.axvline(x=25, linestyle="-.", alpha=0.5, linewidth=1,
                color="black", label="Bubble radius")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Radius [pc]")
    plt.ylabel("Sound speed [km/s]")
    plt.legend(fontsize=12)
    plt.grid()
    fig.tight_layout()
    plt.show()


class PSR_SNR_System:
    def __init__(self,
                 E_SN = 1e51*cgs.erg,
                 n_ISM = 1,
                 m_ej = 15*cgs.sun_mass, 
                 mass_loss = 1e-5*cgs.sun_mass/cgs.year, 
                 wind_speed = 3e6*cgs.cm/cgs.second, n = 500,
                 wind_radius = 1.5*cgs.pc,
                 bubble_radius = 25*cgs.pc,
                 bubble_density = 0.01*cgs.proton_mass,
                 weaver:bool = False, model_shell:bool = True) -> None:
        
        self.supernova_energy = E_SN
        self.ism_density = n_ISM*cgs.proton_mass
        self.bubble_density = bubble_density
        self.wind_radius = wind_radius
        self.bubble_radius = bubble_radius
        self.shell_width = 1*cgs.pc
        self.shell_density = 17*cgs.proton_mass
        self.shell_radius = self.bubble_radius + self.shell_width
        self.ejected_mass = m_ej
        self.stellar_mass_loss = mass_loss
        self.wind_speed = wind_speed

        self.integration_points = n
        self.radius_arr_ref = np.geomspace(1*cgs.pc, 200*cgs.pc, num=n)
        self.radius_arr = self.radius_arr_ref

        self.weaver = weaver
        self.model_shell = model_shell


    def associate_values(self):

        self.temperature = temperature_profile(self.radius_arr,
                                               self.wind_radius,
                                               self.bubble_radius)
        self.density = density_profile(self.radius_arr,
                                       self.wind_radius,
                                       self.bubble_radius,
                                       self.ism_density,
                                       self.bubble_density)
        self.coolingtime = cooling_time(self.temperature,
                                        self.density)
        self.soundspeed = sound_speed(self.temperature)

    def give_time(self, plot=False):

        self.radius_arr = np.geomspace(1*cgs.pc, 200*cgs.pc, num=self.integration_points)
        integration_constant = shock.give_time_radius_integration(self.radius_arr[0],
                                                         self.wind_radius,
                                                         self.bubble_radius)

        if self.model_shell:
            self.speed_arr = csm.speed_profile(self.radius_arr,
                                               m_ej=self.ejected_mass,
                                               mass_loss=self.stellar_mass_loss,
                                               wind_speed=self.wind_speed,
                                               rw=self.wind_radius,
                                               rb=self.bubble_radius,
                                               rho_bubble=self.bubble_density,
                                               rho_shell=self.shell_density,
                                               rho_ISM=self.ism_density,
                                               r_shell=self.shell_width,
                                               E_SN=self.supernova_energy,
                                               weaver=self.weaver)
            integration_constant = inte.quad(lambda x:1/csm.speed_profile(r=np.array([x]),
                                                                          m_ej=self.ejected_mass,
                                                                          mass_loss=self.stellar_mass_loss,
                                                                          wind_speed=self.wind_speed,
                                                                          rw=self.wind_radius,
                                                                          rb=self.bubble_radius,
                                                                          rho_bubble=self.bubble_density,
                                                                          rho_shell=self.shell_density,
                                                                          rho_ISM=self.ism_density,
                                                                          r_shell=self.shell_width,
                                                                          E_SN=self.supernova_energy,
                                                                          weaver=self.weaver),
                                             0.001, self.radius_arr[0])[0]
        elif not self.model_shell:
            self.speed_arr = shock.give_speed_radius_analytical(self.radius_arr,
                                                                self.wind_radius,
                                                                self.bubble_radius,
                                                                self.supernova_energy,
                                                                self.ejected_mass)
            integration_constant = inte.quad(lambda x:1/shock.give_speed_radius_analytical(r=x,
                                                                                           r_w=self.wind_radius,
                                                                                           r_b=self.bubble_radius,
                                                                                           E=self.supernova_energy,
                                                                                           M=self.ejected_mass),
                                             0.001, self.radius_arr[0])[0]
                

        self.time_arr = np.array([inte.simpson(1/self.speed_arr[:i], self.radius_arr[:i])
                                  for i in range(2, self.integration_points)]) + integration_constant
        
        self.radius_arr = self.radius_arr[1:-1]
        self.speed_arr = self.speed_arr[1:-1]

        if plot:
            fig = plt.figure()

            plt.plot(self.time_arr/cgs.kyr, self.radius_arr/cgs.pc)
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Time [kyr]")
            plt.ylabel("Radius [pc]")
            plt.grid()
            fig.tight_layout()
            plt.show()


    def merger(self, alpha = 2, plot:bool = False):

        if plot:
            fig = plt.figure()

            plt.plot(self.time_arr/cgs.kyr, self.speed_arr)
            plt.plot(self.time_arr/cgs.kyr, self.soundspeed*alpha)
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Time [kyr]")
            plt.ylabel("Radius [pc]")
            plt.grid()
            fig.tight_layout()
            plt.show()

        # Merger with the medium because the shock becomes subsonic
        for i in range(len(self.radius_arr)):
            if self.soundspeed[i]*alpha > self.speed_arr[i]:
                break

        self.merger_time = self.time_arr[i]
        self.merger_radius = self.radius_arr[i]
        self.radius_arr[i:] = 0


    def radiative_phase(self):

        i = 0

        while self.time_arr[i] > self.coolingtime[i] and \
              self.radius_arr[i] > self.wind_radius:
            i += 1

        self.first_radiative_radius = self.radius_arr[i]
        self.first_radiative_time = self.time_arr[i]
        self.first_radiative_speed = self.speed_arr[i]

        if self.first_radiative_radius < self.shell_radius:
            self.radiative_phase_inside()
        else:
            self.radiative_phase_outside()
        


    def radiative_phase_inside(self):

        i = 0

        # Condition of going outside the shell
        while self.radius_arr[i+1] < self.shell_radius:
            i += 1

        self.radius_arr[i:] = self.shell_radius
        self.speed_arr[i:] = 0


    def radiative_phase_outside(self):

        i = 0

        # Condition of going outside the shell
        while self.radius_arr[i+1] < self.shell_radius:
            i += 1

        shell_index = i

        # Condition of having the same powerlaw as the radiative radius
        # after the shell
        powerlaw_index = np.array([(np.log10(self.radius_arr[j+1]) - np.log10(self.radius_arr[j]))/\
                         (np.log10(self.time_arr[j+1]) - np.log10(self.time_arr[j])) for j in range(shell_index, len(self.radius_arr)-1)])

        i = 0

        while powerlaw_index[i] < 0.3:
            i+=1

        i += shell_index

        self.first_radiative_radius = self.radius_arr[i]
        self.first_radiative_time = self.time_arr[i]
        self.first_radiative_speed = self.speed_arr[i]

        for j in range(i, len(self.radius_arr)):
            self.radius_arr[j] = self.first_radiative_radius * (self.time_arr[j] / self.first_radiative_time)**(0.3)
            self.speed_arr[j] = self.first_radiative_speed * (self.time_arr[j] / self.first_radiative_time)**(-0.7)


    def evolve(self):

        start = time.time()
        # self.reinitialize()
        self.give_time()
        self.associate_values()
        self.radiative_phase()
        self.merger()
        end = time.time()
        # print(f"Computation takes {(end-start)} s.")

    def reinitialize(self):
        
        self.radius_arr = np.geomspace(1*cgs.pc, 200*cgs.pc, num=self.integration_points)
        del self.speed_arr, self.time_arr, self.first_radiative_radius,
        self.first_radiative_speed, self.first_radiative_time,
        self.merger_radius, self.merger_time

    def initialize_pulsar(self):

        self.kick_velocity = pulsar.give_kick_velocity()

        self.radius_pulsar = self.time_arr*self.kick_velocity

    def compare_pulsar(self):

        self.initialize_pulsar()
        self.escape_time = pulsar.find_exact_crossing_point(self.radius_arr,
                                                            self.radius_pulsar,
                                                            self.time_arr)
        
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


def evaluate_one_system(M=8, t=100e3*cgs.year):

    star = bubble.Star(M)

    system = PSR_SNR_System(mass_loss=star.mass_loss,
                            wind_speed=star.wind_speed,
                            wind_radius=star.wind_radius,
                            bubble_radius=star.bubble_radius,
                            bubble_density=star.bubble_density
                            )
    
    system.evolve()
    inside = system.give_pulsar_population_inside(t=t, n_pulsars=1)/100
    escape_time = system.escape_time

    return inside, escape_time


def evaluate_several_systems(n=1000, t=100e3*cgs.year):

    proportion_arr = np.array([])
    escape_times = np.array([])

    for _ in range(n):
        M = bubble.give_random_value(bubble.pick_IMF, 8, 40)
        result = evaluate_one_system(M, t)
        proportion_arr = np.append(proportion_arr, result[0])
        escape_times = np.append(escape_times, result[1])

    proportion = np.count_nonzero(proportion_arr)/n*100

    return proportion, escape_times



def test_integration_number_points(plot_evolution:bool=True):

    n_arr = 2**np.arange(2, 15, 1)

    difference = []

    system = PSR_SNR_System(n=1000)
    system.model_shell = False  
    system.give_time()
    system.associate_values()
    r_arr = system.radius_arr

    time_array = np.array([shock.give_time_radius_integration(r_, 1.5*cgs.pc, 25*cgs.pc)
                      for r_ in r_arr])
    
    integral_t_arr_reference = inte.simpson(time_array, r_arr)

    fig = plt.figure()

    if plot_evolution:
        plt.plot(time_array/cgs.kyr, r_arr/cgs.pc, color="black", linewidth=2, label = "Integration with QUADPACK")

    for n_ in tqdm(n_arr):
        system = PSR_SNR_System(n=n_)
        system.give_time()

        integral_t_arr = inte.simpson(system.time_arr, system.radius_arr)
        diff = np.abs(integral_t_arr - integral_t_arr_reference)/integral_t_arr_reference
        difference.append(diff)

        if plot_evolution:
            plt.plot(system.time_arr/cgs.kyr, system.radius_arr/cgs.pc)

    difference = np.array(difference)

    # print(integral_t_arr)

    if not plot_evolution:
        plt.plot(n_arr, difference*100)
    plt.xscale("log")
    plt.yscale("log")
    if plot_evolution:
        plt.xlabel("Time [kyr]")
        plt.ylabel("Radius [pc]")
    else:
        plt.xlabel("Number of array points")
        plt.ylabel("Integral difference [%]")
    plt.grid()
    if plot_evolution:
        plt.legend()
    fig.tight_layout()
    plt.show()


def final_system_evolution(n=100):

    system = PSR_SNR_System(n=n)
    system.associate_values()
    system.give_time()


    fig = plt.figure()

    time_array = np.array([shock.give_time_radius_integration(r_, 1.5*cgs.pc, 25*cgs.pc)
                            for r_ in system.radius_arr])

    plt.plot(time_array/cgs.kyr, system.radius_arr/cgs.pc, color="black", linewidth=2, label = "Integration with QUADPACK")

    plt.plot(system.time_arr/cgs.kyr, system.radius_arr/cgs.pc, label="Without radiative")

    system.radiative_phase()

    plt.plot(system.time_arr/cgs.kyr, system.radius_arr/cgs.pc, label="With radiative")

    system = PSR_SNR_System(n=n)
    system.evolve()

    plt.axhline(y=1.5, linestyle="--", alpha=0.5, linewidth=1,
                color="black", label="Wind radius")
    plt.axhline(y=25, linestyle="-.", alpha=0.5, linewidth=1,
                color="black", label="Bubble radius")


    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Time [kyr]")
    plt.ylabel("Radius [pc]")
    plt.grid()
    plt.legend(fontsize=12)
    fig.tight_layout()
    plt.show()


def plot_comparison_different_models(n=500):

    m_ej = 5*cgs.sun_mass

    system = PSR_SNR_System(n=n, m_ej=m_ej)
    system.model_shell = False
    system.give_time()
    system.associate_values()

    fig = plt.figure()

    time_array = np.array([shock.give_time_radius_integration(r_, 1.5*cgs.pc, 25*cgs.pc, M=m_ej)
                            for r_ in system.radius_arr])

    plt.plot(time_array/cgs.kyr, system.radius_arr/cgs.pc, color="black", linewidth=3, label = "QUADPACK CSM")

    time_array_no_CSM = np.array([shock.give_time_radius_integration_constant_density(r_, M=m_ej)
                            for r_ in system.radius_arr])

    plt.plot(time_array_no_CSM/cgs.kyr, system.radius_arr/cgs.pc, color="grey", linewidth=2, label = "QUADPACK ISM")

    system.radiative_phase()
    print(system.first_radiative_time/cgs.kyr)
    system.merger()
    plt.plot(system.time_arr/cgs.kyr, system.radius_arr/cgs.pc, label="No shell")

    for boolean in [False, True]:
        system.model_shell = True
        system.weaver = boolean
        system.give_time()
        system.radiative_phase()
        print(system.first_radiative_time/cgs.kyr)
        system.merger()


        plt.plot(system.time_arr/cgs.kyr, system.radius_arr/cgs.pc, label=f"Weaver {boolean}")

    system.reinitialize()
    system.give_time()

    radius_arr_simple = np.array([bubble.give_SN_radius(t_, E=1e51, n=1, T=8000) for t_ in system.time_arr/cgs.year]) # pc

    plt.plot(system.time_arr/cgs.kyr, radius_arr_simple, linestyle="--", label="Cioffi+1988")

    system.radiative_phase()
    system.evolve()


    plt.axhline(y=1.5, linestyle="--", alpha=0.5, linewidth=1,
                color="black", label="Wind radius")
    plt.axhline(y=25, linestyle="-.", alpha=0.5, linewidth=1,
                color="black", label="Bubble radius")
    plt.axvline(x=bubble.give_SN_PDS_time(E=1e51, n=1)/1e3, alpha=0.5,
                linewidth=1, color="red", linestyle=":", label="Radiative")
    
    # Characteristic pulsar

    v_kick = 300 * cgs.km / cgs.sec
    plt.plot(system.time_arr/cgs.kyr, system.time_arr*v_kick/cgs.pc,
             linestyle="-", linewidth=3, label="Characteristic pulsar")


    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Time [kyr]")
    plt.ylabel("Radius [pc]")
    plt.grid()
    plt.legend(fontsize=10)
    fig.tight_layout()
    plt.savefig("Project Summary/CSM_plots/radius(time)_comnparisons.pdf")
    plt.show()



def plot_escape_time_distribution():

    systems_number = 10000
    escape_times = np.array([])

    system = PSR_SNR_System()
    system.evolve()

    for _ in tqdm(range(systems_number)):
        escape_times = np.append(escape_times, system.give_escape_time()/cgs.kyr)

    fig = plt.figure()

    _, bins = np.histogram(escape_times, bins=100)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))

    plt.hist(escape_times, bins=logbins, histtype="step")
    plt.xscale("log")
    plt.xlabel(r"$t_\mathrm{BS}$ [kyr]")
    plt.ylabel(r"Pulsars")
    plt.grid()
    fig.tight_layout()
    plt.show()


def plot_is_pulsar_inside():

    times_arr = np.geomspace(1*cgs.kyr, 10*cgs.Myr, 20)
    systems_number = 100000
    proportion_arr = np.array([])

    system = PSR_SNR_System()
    system.evolve()

    for time in tqdm(times_arr):
        proportion_arr = np.append(proportion_arr,
                                   system.give_pulsar_population_inside(time, systems_number))

    fig = plt.figure()

    plt.plot(times_arr/cgs.kyr, proportion_arr)
    plt.xscale("log")
    plt.xlabel("Pulsar Age [kyr]")
    plt.ylabel("Pulsars inside SNR [%]")
    plt.grid()
    fig.tight_layout()
    plt.show()




if __name__ == "__main__":

    # convergence_radius()
    # test_density_temperature_profile()
    # test_sound_speed()

    # final_system_evolution(n=1000)

    plot_comparison_different_models(n=500)
    
    # plot_escape_time_distribution()

    # plot_is_pulsar_inside()

    # test_integration_number_points()


    # print(evaluate_several_systems(t=100*cgs.kyr))

    1
