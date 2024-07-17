import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from tqdm import tqdm
import cgs

plt.rcParams.update({'font.size': 12})
plt.rcParams["font.family"] = "serif"


def density_profile(r:np.ndarray,
                    mass_loss:float = 1e-5*cgs.sun_mass/cgs.year,
                    wind_speed:float = 3e6*cgs.cm/cgs.second,
                    rho_bubble:float = 1e-2*cgs.proton_mass,
                    rho_shell:float = 17*cgs.proton_mass,
                    rho_ISM:float = 1*cgs.proton_mass,
                    rw:float = 1.5*cgs.pc,
                    rb:float = 25*cgs.pc,
                    r_shell:float = 0.3*cgs.pc) -> np.ndarray:
    global weaver

    rho = []

    for r_ in r:
        if r_ < rw:
            rho.append(mass_loss/(4*np.pi * wind_speed) * r_**-2)
        elif r_ < rb:
            if weaver:
                rho.append(rho_bubble*((1 - r_/rb) / (1 - rw/rb))**(-2/5))
            else:
                rho.append(rho_bubble)
        elif r_ < rb+r_shell:
            rho.append(rho_shell)
        else:
            rho.append(rho_ISM)

    return np.array(rho)


def mass_wind_region(r:np.ndarray,
                     m_ej:float = 5*cgs.sun_mass,
                     mass_loss:float = 1e-5*cgs.sun_mass/cgs.year,
                     wind_speed:float = 3e6*cgs.cm/cgs.second) -> np.ndarray:

    masss = m_ej + mass_loss/wind_speed * r

    return masss


def mass_bubble_region(r:np.ndarray,
                       m_ej:float = 5*cgs.sun_mass,
                       mass_loss:float = 1e-5*cgs.sun_mass/cgs.year,
                       wind_speed:float = 3e6*cgs.cm/cgs.second,
                       rw:float = 1.5*cgs.pc,
                       rb:float = 25*cgs.pc,
                       rho_bubble:float = 1e-2*cgs.proton_mass) -> np.ndarray:
    global weaver
    
    masss = mass_wind_region(rw, m_ej, mass_loss, wind_speed)
    if weaver:
        masss += 4*np.pi * rho_bubble * (1 - rw/rb)**(2/5) * 5/156 * rb**(2/5) * \
                (-12*r**2*(-r+rb)**(3/5) - 15*r*rb*(-r+rb)**(3/5) + \
                25*rb**2*((rb-rw)**(3/5) - (rb-r)**(3/5)) + 15*rb*(rb-rw)**(3/5) \
                * rw + 12*(rb-rw)**(3/5)*rw**2)
    else:
        masss += 4*np.pi/3 * rho_bubble * (r**3 - rw**3)
    
    return masss

def mass_shell_region(r:np.ndarray,
                      m_ej:float = 5*cgs.sun_mass,
                      mass_loss:float = 1e-5*cgs.sun_mass/cgs.year,
                      wind_speed:float = 3e6*cgs.cm/cgs.second,
                      rw:float = 1.5*cgs.pc,
                      rb:float = 25*cgs.pc,
                      rho_bubble:float = 1e-2*cgs.proton_mass,
                      rho_shell:float = 17*cgs.proton_mass) -> np.ndarray:
    
    masss = mass_bubble_region(rb, 
                               m_ej, 
                               mass_loss, 
                               wind_speed, 
                               rw, 
                               rb, 
                               rho_bubble) \
            + 4*np.pi/3 * rho_shell * (r**3 - rb**3)
    
    return masss


def mass_ISM_region(r:np.ndarray,
                    m_ej:float = 5*cgs.sun_mass,
                    mass_loss:float = 1e-5*cgs.sun_mass/cgs.year,
                    wind_speed:float = 3e6*cgs.cm/cgs.second,
                    rw:float = 1.5*cgs.pc,
                    rb:float = 25*cgs.pc,
                    rho_bubble:float = 1e-2*cgs.proton_mass,
                    rho_shell:float = 17*cgs.proton_mass,
                    rho_ISM:float = 1*cgs.proton_mass,
                    r_shell:float = 0.3*cgs.pc) -> np.ndarray:
    
    masss = mass_shell_region(r_shell+rb, 
                              m_ej, 
                              mass_loss, 
                              wind_speed, 
                              rw, 
                              rb, 
                              rho_bubble,
                              rho_shell) \
            + 4*np.pi/3 * rho_ISM * (r**3 - (r_shell+rb)**3)
    
    return masss


def mass_profile(r:np.ndarray,
                 m_ej:float = 5*cgs.sun_mass,
                 mass_loss:float = 1e-5*cgs.sun_mass/cgs.year,
                 wind_speed:float = 3e6*cgs.cm/cgs.second,
                 rw:float = 1.5*cgs.pc,
                 rb:float = 25*cgs.pc,
                 rho_bubble:float = 1e-2*cgs.proton_mass,
                 rho_shell:float = 17*cgs.proton_mass,
                 rho_ISM:float = 1*cgs.proton_mass,
                 r_shell:float = 0.3*cgs.pc) -> np.ndarray:
    
    mass_array = []
    
    for r_ in r:
        if r_ < rw:
            mass_array.append(mass_wind_region(r_,
                                               m_ej,
                                               mass_loss,
                                               wind_speed))
            
        elif r_ < rb:
            mass_array.append(mass_bubble_region(r_,
                                                 m_ej,
                                                 mass_loss,
                                                 wind_speed,
                                                 rw,
                                                 rb,
                                                 rho_bubble))
            
        elif r_ < r_shell+rb:
            mass_array.append(mass_shell_region(r_,
                                                m_ej,
                                                mass_loss,
                                                wind_speed,
                                                rw,
                                                rb,
                                                rho_bubble,
                                                rho_shell))
            
        else:
            mass_array.append(mass_ISM_region(r_,
                                              m_ej,
                                              mass_loss,
                                              wind_speed,
                                              rw,
                                              rb,
                                              rho_bubble,
                                              rho_shell,
                                              rho_ISM,
                                              r_shell))
            
    return np.array(mass_array)


def integral_wind_region(r:np.ndarray,
                         m_ej:float = 5*cgs.sun_mass,
                         mass_loss:float = 1e-5*cgs.sun_mass/cgs.year,
                         wind_speed:float = 3e6*cgs.cm/cgs.second
                         ) -> np.ndarray:
    
    I = m_ej * r**alpha/alpha + mass_loss/wind_speed*r**(alpha+1)/(alpha+1)
    
    return I


def beta_func(x, a, b):
    return sp.betainc(a, b, x)*sp.beta(a, b)


def integral_bubble_region(r:np.ndarray,
                           m_ej:float = 5*cgs.sun_mass,
                           mass_loss:float = 1e-5*cgs.sun_mass/cgs.year,
                           wind_speed:float = 3e6*cgs.cm/cgs.second,
                           rw:float = 1.5*cgs.pc,
                           rb:float = 25*cgs.pc,
                           rho_bubble:float = 1e-2*cgs.proton_mass,
                           weaver:bool=False
                           ) -> np.ndarray:

    
    if weaver:
        I = integral_wind_region(rw,
                             m_ej,
                             mass_loss,
                             wind_speed)
        
        I += (m_ej + mass_loss/wind_speed*rw)*(r**alpha - rw**alpha)/alpha
        
        I += 4*np.pi * rho_bubble * (1 - rw/rb)**(2/5) / (156*alpha) * (5*rb*(r**alpha-rw**alpha) * (25*rb**(7/5) * (rb - rw)**(3/5) + 15*rb**(2/5) * (rb - rw)**(3/5) *rw + 12*rw**2 * (1 - rw/rb)**(3/5)) - 5*rb**(3+alpha) * alpha * (25*beta_func(r/rb, alpha, 8/5) + 15*beta_func(r/rb, 1+alpha, 8/5) + 12*beta_func(r/rb, 2+alpha, 8/5) - 25*beta_func(rw/rb, alpha, 8/5) - 15*beta_func(rw/rb, 1+alpha, 8/5) - 12*beta_func(rw/rb, 2+alpha, 8/5)))
        
    else:
        I = integral_wind_region(rw,
                             m_ej,
                             mass_loss,
                             wind_speed)
    
        I += (mass_bubble_region(r,
                             m_ej,
                             mass_loss,
                             wind_speed,
                             rw,
                             rb,
                             rho_bubble) - 4*np.pi/3*rho_bubble*r**3) \
            * (r**alpha - rw**alpha)/alpha
        I += 4*np.pi/3 * rho_bubble * ((r**(alpha+3) - rw**(alpha+3))/(alpha+3))
    
    return I


def integral_shell_region(r:np.ndarray,
                          m_ej:float = 5*cgs.sun_mass,
                          mass_loss:float = 1e-5*cgs.sun_mass/cgs.year,
                          wind_speed:float = 3e6*cgs.cm/cgs.second,
                          rw:float = 1.5*cgs.pc,
                          rb:float = 25*cgs.pc,
                          rho_bubble:float = 1e-2*cgs.proton_mass,
                          rho_shell:float = 17*cgs.proton_mass,
                          weaver:bool=False) -> np.ndarray:
    
    I = integral_bubble_region(rb, 
                               m_ej, 
                               mass_loss, 
                               wind_speed, 
                               rw, 
                               rb, 
                               rho_bubble,
                               weaver)
    
    I += (mass_shell_region(r,
                            m_ej,
                            mass_loss,
                            wind_speed,
                            rw,
                            rb,
                            rho_bubble,
                            rho_shell) - 4*np.pi/3*rho_shell*r**3) \
            * (r**alpha - rb**alpha)/alpha
    
    I += 4*np.pi/3 * rho_shell * ((r**(alpha+3) - rb**(alpha+3))/(alpha+3))
    
    return I


def integral_ISM_region(r:np.ndarray,
                        m_ej:float = 5*cgs.sun_mass,
                        mass_loss:float = 1e-5*cgs.sun_mass/cgs.year,
                        wind_speed:float = 3e6*cgs.cm/cgs.second,
                        rw:float = 1.5*cgs.pc,
                        rb:float = 25*cgs.pc,
                        rho_bubble:float = 1e-2*cgs.proton_mass,
                        rho_shell:float = 17*cgs.proton_mass,
                        rho_ISM:float = 1*cgs.proton_mass,
                        r_shell:float = 0.3*cgs.pc) -> np.ndarray:
    
    rr = rb+r_shell
    
    I = integral_shell_region(rr, 
                              m_ej, 
                              mass_loss, 
                              wind_speed, 
                              rw, 
                              rb, 
                              rho_bubble,
                              rho_shell)
    
    I += (mass_ISM_region(r,
                          m_ej,
                          mass_loss,
                          wind_speed,
                          rw,
                          rb,
                          rho_bubble,
                          rho_shell,
                          rho_ISM,
                          r_shell) - 4*np.pi/3*rho_ISM*r**3) \
            * (r**alpha - rr**alpha)/alpha
    
    I += 4*np.pi/3 * rho_ISM * ((r**(alpha+3) - rr**(alpha+3))/(alpha+3))
    
    return I


def speed_profile(r:np.ndarray,
                  m_ej:float = 5*cgs.sun_mass,
                  mass_loss:float = 1e-5*cgs.sun_mass/cgs.year,
                  wind_speed:float = 3e6*cgs.cm/cgs.second,
                  rw:float = 1.5*cgs.pc,
                  rb:float = 25*cgs.pc,
                  rho_bubble:float = 1e-2*cgs.proton_mass,
                  rho_shell:float = 17*cgs.proton_mass,
                  rho_ISM:float = 1*cgs.proton_mass,
                  r_shell:float = 0.3*cgs.pc,
                  E_SN:float = 1e51*cgs.erg,
                  weaver:bool=False) -> np.ndarray:
    
    integrall = []
    mass_array = mass_profile(r,
                              m_ej,
                              mass_loss,
                              wind_speed,
                              rw,
                              rb,
                              rho_bubble,
                              rho_shell,
                              rho_ISM,
                              r_shell)
    
    factor = (gamma+1)/2 * np.sqrt(2*alpha*E_SN/(mass_array**2 * r**alpha))
    
    for r_ in r:
        if r_ < rw:
            integrall.append(integral_wind_region(r_,
                                                  m_ej,
                                                  mass_loss,
                                                  wind_speed))
            
        elif r_ < rb:
            integrall.append(integral_bubble_region(r_,
                                                    m_ej,
                                                    mass_loss,
                                                    wind_speed,
                                                    rw,
                                                    rb,
                                                    rho_bubble,
                                                    weaver))
            
        elif r_ < r_shell+rb:
            integrall.append(integral_shell_region(r_,
                                                   m_ej,
                                                   mass_loss,
                                                   wind_speed,
                                                   rw,
                                                   rb,
                                                   rho_bubble,
                                                   rho_shell))
            
        else:
            integrall.append(integral_ISM_region(r_,
                                                 m_ej,
                                                mass_loss,
                                                wind_speed,
                                                rw,
                                                rb,
                                                rho_bubble,
                                                rho_shell,
                                                rho_ISM,
                                                r_shell))
            
    speedd = factor * np.sqrt(integrall)
            
    return speedd


def test_density_profile():
    global weaver
    
    r_arr = np.geomspace(0.1*cgs.pc, 100*cgs.pc, 10000)

    fig = plt.figure()

    for boolean in [True, False]:
        weaver = boolean
        rho_arr = density_profile(r_arr)
        plt.plot(r_arr/cgs.pc, rho_arr/cgs.proton_mass, label=r"$\rho(r)$, "f"Weaver {weaver}")

    plt.axvline(x=1.5, color='black', linestyle="--",
                label=r"$r_\mathrm{w}$")
    plt.axvline(x=25, color='black', linestyle="-.",
                label=r"$r_\mathrm{b}$")
    plt.axvline(x=25.3, color='black', linestyle=":",
                label=r"$r_\mathrm{shell}$")

    plt.xlabel(r"Radius [pc]")
    plt.ylabel(r"Density [cm$^{-3}$]")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()
    plt.legend()
    fig.tight_layout()
    #plt.savefig("Project Summary/CSM_plots/density_structure.pdf")
    plt.show()
    


def test_mass_profile():
    global weaver
    
    r_arr = np.geomspace(0.1*cgs.pc, 100*cgs.pc, 10000)

    fig = plt.figure()

    for boolean in [True, False]:
        weaver = boolean
        m_arr = mass_profile(r_arr, m_ej=15*cgs.sun_mass)
        plt.plot(r_arr/cgs.pc, m_arr/cgs.sun_mass, label=r"M(r), "f"Weaver {weaver}")

    plt.axvline(x=1.5, color='black', linestyle="--",
                label=r"$r_\mathrm{w}$")
    plt.axvline(x=25, color='black', linestyle="-.",
                label=r"$r_\mathrm{b}$")
    plt.axvline(x=25.3, color='black', linestyle=":",
                label=r"$r_\mathrm{shell}$")

    plt.xlabel(r"Radius [pc]")
    plt.ylabel(r"Mass [M$_\odot$]")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()
    plt.legend()
    fig.tight_layout()
    #plt.savefig("Project Summary/CSM_plots/accurate_csm mass structure.pdf")
    plt.show()


def test_speed_profile():
    global weaver
    
    r_arr = np.geomspace(0.1*cgs.pc, 100*cgs.pc, 10000)
    

    fig = plt.figure()

    for boolean in [True, False]:
        weaver = boolean
        s_arr = speed_profile(r_arr, m_ej=15*cgs.sun_mass)
        plt.plot(r_arr/cgs.pc, s_arr, label=r"$u_\mathrm{s}(r)$, "f"Weaver {weaver}")

    plt.axvline(x=1.5, color='black', linestyle="--",
                label=r"$r_\mathrm{w}$")
    plt.axvline(x=25, color='black', linestyle="-.",
                label=r"$r_\mathrm{b}$")
    plt.axvline(x=25.3, color='black', linestyle=":",
                label=r"$r_\mathrm{shell}$")

    plt.xlabel(r"Radius [pc]")
    plt.ylabel(r"Speed [cm/s]")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()
    plt.legend()
    fig.tight_layout()
    # plt.savefig("Project Summary/CSM_plots/accurate_csm speed structure.pdf")
    plt.show()

gamma = 5/3
alpha = 6*(gamma-1)/(gamma+1)
weaver = False

if __name__ == "__main__":

    # test_density_profile()
    # test_mass_profile()
    # test_speed_profile()

    1
