# distutils: language = c++
# cython: language_level=3
# cython: boundscheck=False, wraparound=False, initializedcheck=False

import numpy as np
from libcpp.vector cimport vector

cdef extern from "qubo_bnb.hpp":
    cdef cppclass QuboResult:
        double value
        vector[int] assignment
        long long explored_nodes

    QuboResult solve_qubo_bnb_parallel(
        const double* Q,
        int n,
        int split_depth,
        int num_threads
    ) except +


def solve_qubo_omp(object Q,
                   int split_depth=4,
                   int num_threads=4):
    cdef object Q_arr
    cdef double[:, ::1] Q_view
    cdef int n
    cdef QuboResult res
    cdef Py_ssize_t i
    cdef list assignment

    Q_arr = np.ascontiguousarray(Q, dtype=np.float64)

    if Q_arr.ndim != 2 or Q_arr.shape[0] != Q_arr.shape[1]:
        raise ValueError("Q must be a square matrix")

    n = Q_arr.shape[0]
    Q_view = Q_arr

    res = solve_qubo_bnb_parallel(&Q_view[0, 0], n, split_depth, num_threads)

    assignment = [0] * n
    for i in range(n):
        assignment[i] = res.assignment[i]

    return {
        "objective_value": res.value,
        "assignment": np.array(assignment, dtype=np.int8),
        "explored_nodes": res.explored_nodes,
    }
