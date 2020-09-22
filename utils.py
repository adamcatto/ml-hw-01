"""
Author: Adam Catto
09-13-2020
Machine Learning Problem Set 01
CUNY Graduate Center, C SC 74020, Fall 2020
"""

import math
from typing import Tuple

import numpy as np


def pairwise_distance_numpy(v1, v2):
    """

    :param v1: 1d numpy array
    :param v2: 1d numpy array
    :return: float representing pairwise Euclidean distance between v1 and v2.
             This is done by summing the squares of the lengths of v1 and v2,
             and subtracting twice their dot product, followed by the square root
             of the whole thing. Note that their is no need for transposition, since
             they are both 1d-arrays. (If they were 2d arrays with only one row, then
             v2 would need to be transposed.)

    """
    return np.sqrt((np.linalg.norm(v1)**2 + np.linalg.norm(v2)**2 - 2*np.dot(v1, v2)))


def pairwise_distance_loop(v1, v2):
    diff = v1 - v2
    payload = 0
    for d in diff:
        payload += d**2
    return math.sqrt(payload)


def covar(x, position: Tuple[int, int]):
    num_vectors = x.shape[0]
    # feature_dim = x.shape[1]
    """sample_mean = np.zeros(num_vectors)
    for i in range(num_vectors):
        sample_mean += x[:, i]
    sample_mean /= num_vectors"""
    sample_mean = 0
    for i in range(num_vectors):
        sample_mean += x[i][position[1]]
    sample_mean /= num_vectors
    s = 0
    for i in range(num_vectors):
        s += (x[i][position[0]] - sample_mean) * (x[i][position[1]] - sample_mean)
    return s/(num_vectors-1)


def sd(x, idx):
    return math.sqrt(covar(x, (idx, idx)))
