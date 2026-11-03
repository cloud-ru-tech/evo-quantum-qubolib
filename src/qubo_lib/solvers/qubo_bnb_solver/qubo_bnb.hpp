#pragma once
#include <vector>

struct QuboResult {
    double value;
    std::vector<int> assignment;
    long long explored_nodes;
};

QuboResult solve_qubo_bnb_parallel(
    const double* Q,
    int n,
    int split_depth,
    int num_threads
);
