"""
Author: Adam Catto
09-13-2020
Machine Learning Problem Set 01
CUNY Graduate Center, C SC 74020, Fall 2020
"""

from timeit import timeit

import numpy as np
from numpy.random import rand
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import pairwise_distance_loop
from utils import pairwise_distance_numpy

"""
Problem Statement
================
We're given a matrix `X`, of dimension NxD, representing a set of N
data consisting of D features, e.g.
================
[x_00, x_01, ... , x0(D-1)]
[x_10, x_11, ... , x1(D-1)]
. . . . . . . . . . . . . 
. . . . . . . . . . . . . 
. . . . . . . . . . . . .
[x_N0, x_N1, ... , x(N-1)(D-1)]
================x

The first problem is to compute an NxN matrix whose entries are the
pairwise Euclidean distance between any two data, i.e. any two rows
in X.

There are two sub-problems: 
(1) compute Z using a two-level nested loop, iterating through all 
    (i,j)-pairs for i, j in v1, v2

(2) compute Z without any loop – only use numpy matrix operations.
    This is vectorization. 
"""


def compute_pairwise_distance_loop(x):
    """

    :param x: 2d numpy array of dimension num_vectors x feature_dim
    :return: num_vectors x num_vectors 2d numpy array in which each
             (i, j)-entry is the pairwise euclidean distance between
             rows indexed as `i` and `j`.

    """
    num_vectors = x.shape[0]     # number of vectors in dataset
    # feature_dim = x.shape[1]     # how many entries in vector
    z = np.zeros((num_vectors, num_vectors))
    for i in range(num_vectors):
        for j in range(num_vectors):
            # Note: pairwise_distance(x[i, :], x[j, :]) will give either
            # a small, nonzero float or NaN when taking pairwise_distance on
            # i = j, so best to just impute it to zero manually.
            z[i][j] = 0.0 if i == j else pairwise_distance_loop(x[i], x[j])
    return z

"""
def compute_pairwise_distance_numpy(x):
    """

    :param x: 2d numpy array of dimension num_vectors x feature_dim
    :return: num_vectors x num_vectors 2d numpy array in which each
             (i, j)-entry is the pairwise euclidean distance between
             rows indexed as `i` and `j`.

    """
    num_vectors = x.shape[0]     # number of vectors in dataset
    # feature_dim = x.shape[1]     # how many entries in vector
    z = np.zeros((num_vectors, num_vectors))
    for i in range(num_vectors):
        for j in range(num_vectors):
            # Note: pairwise_distance(x[i, :], x[j, :]) will give either
            # a small, nonzero float or NaN when taking pairwise_distance on
            # i = j, so best to just impute it to zero manually.
            z[i][j] = 0.0 if i == j else pairwise_distance_numpy(x[i], x[j])
    return z
"""


def compute_pairwise_distance_ops(x):
    num_vectors = x.shape[0]  # number of vectors in dataset
    feature_dim = x.shape[1]  # how many entries in vector
    d1 = np.matmul(np.diag(np.matmul(x, np.transpose(x))).reshape(-1, 1), np.ones(num_vectors).reshape(1, -1))
    d2 = np.transpose(d1)
    d3 = np.matmul(x, np.transpose(x))
    return np.sqrt(np.subtract(np.add(d1, d2), np.add(d3, d3)))


def comparison(size=None):
    if size is None:
        size = int(input('Up to what size matrix (integer greater than 1) would you like to compare? '))

    assert size > 1
    sizes = [i for i in range(2, size + 1)]
    loop_times = []
    mat_ops_times = []

    for i in tqdm(range(2, size + 1)):
        r = rand(i, i)

        loop_time = timeit(lambda: compute_pairwise_distance_loop(r), number=1)
        mat_ops_time = timeit(lambda: compute_pairwise_distance_ops(r), number=1)

        loop_times.append(loop_time)
        mat_ops_times.append(mat_ops_time)

    plt.xlabel('Matrix Size')
    plt.ylabel('Running Times')
    plt.plot(sizes, loop_times, label='Loop')
    plt.plot(sizes, mat_ops_times, label='Matrix Operations')
    plt.legend()
    plt.savefig('comparison-problem-01.png')
    plt.show()


if __name__ == '__main__':
    comparison()
