#include "qubo_bnb.hpp"

#include <vector>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <queue>
#include <functional>

namespace {

struct Node {
    std::vector<int> assign;
};

struct Solver {
    const double* Q;
    int n;
    int split_depth;

    // Shared state
    std::atomic<double> global_best;
    std::vector<int> global_assign;
    std::mutex best_mutex;
    std::atomic<long long> global_nodes;

    Solver(const double* Q_, int n_, int split_depth_)
        : Q(Q_), n(n_), split_depth(split_depth_),
          global_best(std::numeric_limits<double>::infinity()),
          global_assign(n_, 0),
          global_nodes(0) {}

    inline double q(int i, int j) const {
        return Q[(size_t)i * (size_t)n + (size_t)j];
    }

    double qubo_value(const std::vector<int>& x) const {
        double val = 0.0;
        for (int i = 0; i < n; ++i) {
            if (!x[i]) continue;
            for (int j = 0; j < n; ++j) {
                if (!x[j]) continue;
                val += q(i, j);
            }
        }
        return val;
    }

    double lower_bound(const std::vector<int>& assign) const {
        double assigned_obj = 0.0;
        for (int i = 0; i < n; ++i) {
            if (assign[i] != 1) continue;
            for (int j = 0; j < n; ++j) {
                if (assign[j] != 1) continue;
                assigned_obj += q(i, j);
            }
        }

        double single_lb = 0.0;
        for (int i = 0; i < n; ++i) {
            if (assign[i] != -1) continue;
            double c = q(i, i);
            for (int j = 0; j < n; ++j) {
                if (assign[j] == 1) c += 2.0 * q(i, j);
            }
            if (c < 0.0) single_lb += c;
        }

        double pair_lb = 0.0;
        for (int i = 0; i < n; ++i) {
            if (assign[i] != -1) continue;
            for (int j = i + 1; j < n; ++j) {
                if (assign[j] != -1) continue;
                double w = 2.0 * q(i, j);
                if (w < 0.0) pair_lb += w;
            }
        }

        return assigned_obj + single_lb + pair_lb;
    }

    // Greedy completion modifies assign in-place
    void greedy_completion(std::vector<int>& assign) const {
        while (true) {
            int best_v = -1;
            int best_side = 0;
            double best_gain = 0.0;
            bool found = false;

            for (int i = 0; i < n; ++i) {
                if (assign[i] != -1) continue;

                double c = q(i, i);
                for (int j = 0; j < n; ++j) {
                    if (assign[j] == 1) c += 2.0 * q(i, j);
                }

                int side = (c < 0.0) ? 1 : 0;
                double gain = std::min(c, 0.0);

                if (!found || gain < best_gain) {
                    found = true;
                    best_gain = gain;
                    best_v = i;
                    best_side = side;
                }
            }

            if (best_v == -1) break;
            assign[best_v] = best_side;
        }
    }

    int choose_branch_var(const std::vector<int>& assign) const {
        int best_v = -1;
        double best_assigned = -1.0;
        double best_total = -1.0;

        for (int i = 0; i < n; ++i) {
            if (assign[i] != -1) continue;

            double assigned_coupling = 0.0;
            double total_coupling = 0.0;

            for (int j = 0; j < n; ++j) {
                double a = std::abs(q(i, j));
                total_coupling += a;
                if (assign[j] != -1) assigned_coupling += a;
            }

            if (assigned_coupling > best_assigned ||
                (assigned_coupling == best_assigned && total_coupling > best_total)) {
                best_assigned = assigned_coupling;
                best_total = total_coupling;
                best_v = i;
            }
        }

        return best_v;
    }

    void update_global(double value, const std::vector<int>& assign) {
        // Fast path: atomic read without lock
        double current_best = global_best.load(std::memory_order_relaxed);
        if (value >= current_best) return;

        // Slow path: lock and double-check
        std::lock_guard<std::mutex> lock(best_mutex);
        current_best = global_best.load(std::memory_order_relaxed);
        if (value < global_best) {
            global_best.store(value, std::memory_order_relaxed);
            global_assign = assign;
        }
    }

    double read_global_best() const {
        return global_best.load(std::memory_order_relaxed);
    }

    // DFS с backtracking вместо копирования векторов
    void dfs(std::vector<int>& assign, long long& local_nodes) {
        local_nodes += 1;

        double lb = lower_bound(assign);
        double incumbent = read_global_best();
        if (lb >= incumbent - 1e-12) return;

        int v = -1;
        for (int i = 0; i < n; ++i) {
            if (assign[i] == -1) {
                v = i;
                break;
            }
        }

        if (v == -1) {
            // Leaf node
            double val = qubo_value(assign);
            if (val < incumbent) update_global(val, assign);
            return;
        }

        // Branching variable selection (can be optimized)
        v = choose_branch_var(assign);
        if (v < 0) {
            double val = qubo_value(assign);
            if (val < incumbent) update_global(val, assign);
            return;
        }

        // Greedy completion for both children to update global_best early
        std::vector<int> a0 = assign;
        a0[v] = 0;
        greedy_completion(a0);
        update_global(qubo_value(a0), a0);

        std::vector<int> a1 = assign;
        a1[v] = 1;
        greedy_completion(a1);
        update_global(qubo_value(a1), a1);

        // Order branches by promising first
        if (a1[0] < a0[0]) { // Simplified comparison
            dfs_branch(assign, v, 1, local_nodes);
            dfs_branch(assign, v, 0, local_nodes);
        } else {
            dfs_branch(assign, v, 0, local_nodes);
            dfs_branch(assign, v, 1, local_nodes);
        }
    }

    void dfs_branch(std::vector<int>& assign, int v, int value, long long& local_nodes) {
        assign[v] = value;
        dfs(assign, local_nodes);
        assign[v] = -1; // Backtrack
    }

    void build_frontier(std::vector<int>& assign, int depth, std::vector<Node>& frontier) {
        bool full = true;
        for (int i = 0; i < n; ++i) {
            if (assign[i] == -1) {
                full = false;
                break;
            }
        }

        if (depth == split_depth || full) {
            frontier.push_back({assign});
            return;
        }

        int v = choose_branch_var(assign);
        if (v < 0) {
            frontier.push_back({assign});
            return;
        }

        assign[v] = 0;
        build_frontier(assign, depth + 1, frontier);

        assign[v] = 1;
        build_frontier(assign, depth + 1, frontier);

        assign[v] = -1; // Backtrack
    }

    QuboResult solve_parallel(int num_threads) {
        std::vector<int> root(n, -1);

        // Initial greedy solution
        greedy_completion(root);
        double init_val = qubo_value(root);
        global_best.store(init_val, std::memory_order_relaxed);
        global_assign = root;
        for (int i = 0; i < n; ++i) root[i] = -1; // Reset for search

        // Build frontier
        std::vector<Node> frontier;
        build_frontier(root, 0, frontier);

        // Thread-local node counts
        std::vector<long long> thread_nodes(num_threads, 0);
        std::vector<std::thread> workers;

        // Simple work queue
        std::atomic<int> task_index(0);

        // Worker function
        auto worker_func = [&](int thread_id) {
            while (true) {
                int idx = task_index.fetch_add(1, std::memory_order_relaxed);
                if (idx >= (int)frontier.size()) break;

                std::vector<int> assign = frontier[idx].assign;
                long long local_nodes = 0;
                
                // Quick greedy completion before DFS
                greedy_completion(assign);
                update_global(qubo_value(assign), assign);
                
                // Reset for DFS
                for (int i = 0; i < n; ++i) {
                    if (frontier[idx].assign[i] != -1) {
                        assign[i] = frontier[idx].assign[i];
                    } else {
                        assign[i] = -1;
                    }
                }
                
                dfs(assign, local_nodes);
                thread_nodes[thread_id] = local_nodes;
            }
        };

        // Launch threads
        int actual_threads = std::min(num_threads, (int)frontier.size());
        for (int i = 0; i < actual_threads; ++i) {
            workers.emplace_back(worker_func, i);
        }

        // Wait for completion
        for (auto& t : workers) {
            t.join();
        }

        // Sum up nodes
        long long total_nodes = 0;
        for (auto count : thread_nodes) {
            total_nodes += count;
        }

        return {global_best.load(), global_assign, total_nodes};
    }
};

} // namespace

QuboResult solve_qubo_bnb_parallel(
    const double* Q_in,
    int n,
    int split_depth,
    int num_threads
) {
    std::vector<double> Qsym((size_t)n * (size_t)n, 0.0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double a = Q_in[(size_t)i * (size_t)n + (size_t)j];
            double b = Q_in[(size_t)j * (size_t)n + (size_t)i];
            Qsym[(size_t)i * (size_t)n + (size_t)j] = 0.5 * (a + b);
        }
    }

    Solver solver(Qsym.data(), n, split_depth);
    return solver.solve_parallel(num_threads);
}