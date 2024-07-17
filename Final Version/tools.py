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
    return 1/np.sqrt(2*np.pi*normal_std**2)\
            * np.exp(-(np.log(zz)-normal_mean)**2/(2*normal_std**2))


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

