import math

# CGS UNITS
second = 1
centimeter = 1
gram = 1
erg = gram * (centimeter / second)**2
kelvin = 1
sr = 1
gauss = 1
esu = 1
statvolt = erg / esu
steradian = 1

# TIME UNITS
msec = 1e-3 * second
year = 3.154e7 * second
kiloyear = 1e3 * year
Megayear = 1e6 * year
Gigayear = 1e9 * year

# LENGTH UNITS
meter = 1e2 * centimeter
kilometer = 1e3 * meter
parsec = 3.086e16 * meter
kiloparsec = 1e3 * parsec
fm = 1e-13 * centimeter

# MASS UNITS
mgram = 1e-3 * gram
kilogram = 1e3 * gram

# ENERGY UNITS
joule = 1e7 * erg
electronvolt = 1.60217657e-19 * joule
kiloelectronvolt = 1e3 * electronvolt
megaelectronvolt = 1e6 * electronvolt
gigaelectronvolt = 1e9 * electronvolt
teraelectronvolt = 1e12 * electronvolt
petaelectronvolt = 1e15 * electronvolt

# em derived units
microgauss = 1e-6 * gauss
milligauss = 1e-3 * gauss
nanogauss = 1e-9 * gauss
volt = statvolt / 299.792458

# ABBREVIATION
sec = second
km = kilometer
kyr = kiloyear
Myr = Megayear
Gyr = Gigayear
pc = parsec
kpc = kiloparsec
eV = electronvolt
keV = kiloelectronvolt
MeV = megaelectronvolt
GeV = gigaelectronvolt
TeV = teraelectronvolt
PeV = petaelectronvolt
cm = centimeter
cm2 = centimeter**2
cm3 = centimeter**3
m2 = meter**2
kpc2 = kiloparsec**2
K = kelvin

# PHYSICAL CONSTANTS
c_light = 2.99792458e10 * centimeter / second
c_2 = c_light**2
c_3 = c_light**3
proton_mass = 1.67262158e-24 * gram
proton_mass_c2 = proton_mass * c_2
neutron_mass = 1.67492735e-24 * gram
neutron_mass_c2 = neutron_mass * c_2
electron_mass = 9.10938291e-28 * gram
electron_mass_c2 = electron_mass * c_2
sun_mass = 1.989e33 * gram
h_planck = 6.62607015e-34 * joule * second
k_boltzmann = 1.3806488e-23 * joule / kelvin
electron_radius = 2.8179403227e-15 * meter
elementary_charge = 4.80320427e-10
sigma_th = 6.6524e-25 * cm2
barn = 1e-24 * cm2
mbarn = 1e-3 * barn

# MODEL CONSTANTS
E_SN = 1e51 * erg
gas_density = 3 / cm3
gas_mass_density = proton_mass * gas_density
mass_ejected = sun_mass
R_ST = (3 * mass_ejected / 4 / math.pi / gas_mass_density)**(1 / 3)
u_ST = (2 * E_SN / mass_ejected)**(1 / 2)
t_ST = 0 # TO DO
pulsar_mass = 1.4 * sun_mass
pulsar_radius = 10 * km
