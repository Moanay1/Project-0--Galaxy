import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.optimize as opt
import scipy.interpolate as interpol

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


def pick_kick_velocity_component(v: np.ndarray) -> np.ndarray:
    """Parametrization from Faucher-Giguère et al. 2006 eq 7"""
    w, sigma_v1, sigma_v2 = 0.9, 160*1e5, 780*1e5
    norm = 10000000  # optimum normalization parameter
    # for the value selection process
    return (w*make_gaussian(v, mu=0, sigma=sigma_v1)
            + (1-w)*make_gaussian(v, mu=0, sigma=sigma_v2)) * norm # cm/s


def give_kick_velocity() -> np.ndarray:
    v_x = give_random_value(pick_kick_velocity_component, -2000e5, 2000e5)
    v_y = give_random_value(pick_kick_velocity_component, -2000e5, 2000e5)
    v_z = give_random_value(pick_kick_velocity_component, -2000e5, 2000e5)
    return np.linalg.norm([v_x, v_y, v_z])


def test_kick_velocity() -> None:
    arr = []
    for _ in tqdm(range(10000)):
        arr.append(give_kick_velocity())

    fig = plt.figure()
    plt.hist(arr, histtype="step", density=True, bins=100)
    plt.xlabel(r"$v_\mathrm{k}$ [cm$\cdot$s$^{-1}$]")
    plt.ylabel("PDF")
    plt.xlim(np.min(arr), np.max(arr))
    plt.grid()
    fig.tight_layout()
    plt.show()





def find_exact_crossing_point(A, B, R):
    # Ensure input arrays are numpy arrays
    A = np.array(A)
    B = np.array(B)
    R = np.array(R)
    
    # Validate that the lengths of the arrays are the same
    if not (len(A) == len(B) == len(R)):
        raise ValueError("A, B, and R must have the same length")
    
    # Create interpolating functions for A and B
    interp_A = interpol.interp1d(R, A, kind='linear', fill_value="extrapolate")
    interp_B = interpol.interp1d(R, B, kind='linear', fill_value="extrapolate")
    
    # Define the function for which we want to find the root (crossing point)
    def func(r):
        return interp_A(r) - interp_B(r)
    
    # Find the interval where the crossing occurs
    for i in range(len(R) - 1):
        if (A[i] <= B[i] and A[i+1] > B[i+1]) or (A[i] >= B[i] and A[i+1] < B[i+1]):
            # Use brentq to find the root of the function in the interval [R[i], R[i+1]]
            crossing_point = opt.brentq(func, R[i], R[i+1])
            return crossing_point
    
    # If no crossing is found
    return None




if __name__ == "__main__":

    #test_kick_velocity()


    A = np.array([1, 2, 3, 4, 5])
    B = np.array([1, 2, 3, 4, 5])*2-1.5
    R = A

    crossing_point = find_exact_crossing_point(A, B, R)
    print("Exact crossing point in reference array R:", crossing_point)


    1
