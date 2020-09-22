from timeit import timeit

from sklearn.datasets import load_iris, load_breast_cancer, load_digits
import numpy as np

from problem01 import compute_pairwise_distance_loop, compute_pairwise_distance_ops
from problem02 import compute_correlation_matrix_loop, compute_correlation_matrix_ops


iris_data = load_iris(as_frame=True).data.to_numpy()
breast_cancer_data = load_breast_cancer(as_frame=True).data.to_numpy()
digits_data = load_digits(as_frame=True).data.to_numpy()

iris_pairwise_distance_loop_time = timeit(lambda: compute_pairwise_distance_loop(iris_data), number=1)
iris_pairwise_distance_ops_time = timeit(lambda: compute_pairwise_distance_ops(iris_data), number=1)
iris_correlation_matrix_loop_time = timeit(lambda: compute_correlation_matrix_loop(iris_data), number=1)
iris_correlation_matrix_ops_time = timeit(lambda: compute_correlation_matrix_ops(iris_data), number=1)

breast_cancer_pairwise_distance_loop_time = timeit(lambda: compute_pairwise_distance_loop(breast_cancer_data), number=1)
breast_cancer_pairwise_distance_ops_time = timeit(lambda: compute_pairwise_distance_ops(breast_cancer_data), number=1)
breast_cancer_correlation_matrix_loop_time = timeit(lambda: compute_correlation_matrix_loop(breast_cancer_data), number=1)
breast_cancer_correlation_matrix_ops_time = timeit(lambda: compute_correlation_matrix_ops(breast_cancer_data), number=1)

digits_pairwise_distance_loop_time = timeit(lambda: compute_pairwise_distance_loop(digits_data), number=1)
digits_pairwise_distance_ops_time = timeit(lambda: compute_pairwise_distance_ops(digits_data), number=1)
digits_correlation_matrix_loop_time = timeit(lambda: compute_correlation_matrix_loop(digits_data), number=1)
digits_correlation_matrix_ops_time = timeit(lambda: compute_correlation_matrix_ops(digits_data), number=1)

distance_matrix = [
    [iris_pairwise_distance_loop_time, iris_pairwise_distance_ops_time],
    [breast_cancer_pairwise_distance_loop_time, breast_cancer_pairwise_distance_ops_time],
    [digits_pairwise_distance_loop_time, digits_pairwise_distance_ops_time]
]

correlation_matrix = [
    [iris_correlation_matrix_loop_time, iris_correlation_matrix_ops_time],
    [breast_cancer_correlation_matrix_loop_time, breast_cancer_correlation_matrix_ops_time],
    [digits_correlation_matrix_loop_time, digits_correlation_matrix_ops_time]
]

print(distance_matrix)
print('\n\n')
print(correlation_matrix)