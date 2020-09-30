from timeit import timeit

import numpy as np
from numpy.random import rand
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import variance_, sd


def compute_correlation_matrix_loop(x):
    feature_dim = x.shape[1]
    z = np.zeros((feature_dim, feature_dim))
    for i in range(feature_dim):
        for j in range(feature_dim):
            if (sd_i := sd(x, i) != 0) and (sd_j := sd(x, j) != 0):
                z[i][j] = variance_(x, (i, j)) / (sd_i * sd_j)
            else:
                z[i][j] = 0
    return z


def compute_correlation_matrix_ops(x):
    num_vectors = x.shape[0]
    y = np.subtract(x, np.divide(np.matmul(np.ones((num_vectors, num_vectors)), x), num_vectors))
    vcv = np.divide(np.matmul(np.transpose(y), y), num_vectors)
    try:
        d = np.linalg.inv(np.sqrt(np.diag(np.diag(vcv))))
    except:
        d = np.zeros(vcv.shape)
    corr_matrix = np.matmul(np.matmul(d, vcv), d)
    return corr_matrix


def comparison(size=None):
    if size is None:
        size = int(input('Up to what size matrix (integer greater than 1) would you like to compare? '))

    assert size > 1
    sizes = [i for i in range(2, size + 1)]
    loop_times = []
    mat_ops_times = []

    for i in tqdm(range(2, size + 1)):
        r = rand(i, i)

        loop_time = timeit(lambda: compute_correlation_matrix_loop(r), number=1)
        mat_ops_time = timeit(lambda: compute_correlation_matrix_ops(r), number=1)

        loop_times.append(loop_time)
        mat_ops_times.append(mat_ops_time)

    plt.xlabel('Matrix Size')
    plt.ylabel('Running Times')
    plt.plot(sizes, loop_times, label='Loop')
    plt.plot(sizes, mat_ops_times, label='Matrix Operations')
    plt.legend()
    plt.savefig('comparison-problem-02.png')
    plt.show()


if __name__ == '__main__':
    comparison()
