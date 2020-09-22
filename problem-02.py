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
    y = x - np.matmul(np.ones((num_vectors, num_vectors)), x)/num_vectors
    vcv = (np.matmul(np.transpose(y), y)/num_vectors)
    d = np.linalg.inv(np.sqrt(np.diag(np.diag(vcv))))
    cov_matrix = np.matmul(np.matmul(d, vcv), d)
    return cov_matrix


r = np.random.rand(10, 9)

print(compute_correlation_matrix_loop(r))
print(compute_correlation_matrix_ops(r))
