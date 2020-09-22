import numpy as np

from utils import covar, sd


def compute_correlation_matrix_loop(x):
    feature_dim = x.shape[1]
    z = np.zeros((feature_dim, feature_dim))
    for i in range(feature_dim):
        for j in range(feature_dim):
            z[i][j] = covar(x, (i, j)) / (sd(x, i) * sd(x, j))
    return z


def compute_correlation_matrix_ops(x):
    num_vectors = x.shape[0]
    feature_dim = x.shape[1]
    
