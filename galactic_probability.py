import numpy as np
import matplotlib.pyplot as plt
import cgs
import random
from tqdm import tqdm
import superbubble as SB
import leaving_the_cradle as CSM

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

def pick_IMF_exp(m: np.ndarray) -> np.ndarray:
    return 1*m**(-2.3)*np.exp(-0.35/m)


def associate_star_type(M=8*cgs.sun_mass):
    if M < 16*cgs.sun_mass:
        return "B"
    else:
        return "O"


def create_all_progenitor_stars(n_systems=100):

    masses = np.array([])
    star_types = np.array([])

    for _ in range(n_systems):
        mass = give_random_value(pick_IMF_exp, 8, 40)*cgs.sun_mass
        masses = np.append(masses, mass)
        star_types = np.append(star_types, associate_star_type(mass))
    return masses, star_types


def pick_model(proba_runaway):
    return random.choices(['CSM', 'SB'],
                          weights=[proba_runaway, 1-proba_runaway])[0]


def associate_model(n_systems=100):

    masses, star_types = create_all_progenitor_stars(n_systems=n_systems)
    models = np.array([])

    for i in range(n_systems):
        if star_types[i] == "O":
            proba_runaway = 0.25
        else:
            proba_runaway = 0.02
        models = np.append(models, pick_model(proba_runaway=proba_runaway))

    return masses, star_types, models


def plot_model_probability(n_systems=10000):

    models = associate_model(n_systems=n_systems)[2]
    weights = np.ones_like(models, dtype=float) / len(models)

    x = np.arange(2)
    width = 0.5

    fig = plt.figure()
    counts, edges, bars = plt.hist(models, weights=weights, bins=2)
    plt.bar_label(bars, label_type="center", fmt=".2f")
    plt.xlabel("Model")
    plt.ylabel("Probability")
    fig.tight_layout()
    # plt.savefig(r"Project Summary/Images/pick_arm.pdf")
    plt.show()


def create_galactic_escape_times(n_systems=10000):

    masses, star_types, models = associate_model(n_systems=n_systems)
    escape_times = np.array([])

    indices_runaway = models == "CSM"

    file = open(f"Escape Times/Galactic population.csv", "w")

    for i in tqdm(range(n_systems)):
        mass = masses[i]
        if indices_runaway[i]:
            escape_time = CSM.evaluate_one_system(mass/cgs.sun_mass)[1]/cgs.kyr
        else:
            escape_time = SB.evaluate_one_system(star_mass=mass)/cgs.kyr
        file.write(f"{escape_time}\n")
        escape_times = np.append(escape_times, escape_time)
    
    print(escape_times)

    return escape_times


    



if __name__ == "__main__":

    create_galactic_escape_times()
