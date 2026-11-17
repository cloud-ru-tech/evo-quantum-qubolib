import jsonpickle
import logging
import numpy as np
import os

from ..abc import ProblemABC, SolverABC
from ..solution import Solution
from ..state_vectors import StateVectors
from ..utils import PerformanceTimer
from .qubo_bnb_solver.pyqubo_bnb import solve_qubo_omp

logger = logging.getLogger(__name__)

ALL_METRICS = {'explored_nodes'}


class BnB(SolverABC):
    """
    Exact QUBO solver based on a branch-and-bound algorithm with OpenMP
    parallelization (CPU only).

    Parameters
    ----------
    split_depth : int, optional
        Tree depth used to create parallel tasks (default: 10).
    num_threads : int, optional
        Number of OpenMP threads (default: number of CPU cores).
    use_metrics : str or set, optional
        Collect metrics. Supported: {'explored_nodes'}, or 'all'.
    """

    def __init__(self):
        super().__init__()

    def solve(self, problem: ProblemABC, **kwargs) -> Solution:
        """
        Solve the given QUBO problem using branch and bound.

        Args:
            problem: Problem instance to solve.
            **kwargs: Solver parameters:
                - split_depth: tree depth for parallel splitting (default 10).
                - num_threads: number of OpenMP threads (default os.cpu_count()).
                - use_metrics: 'all' or a set of metric names.

        Returns:
            Solution object.
        """
        self.kwargs = kwargs
        self.json_str = jsonpickle.encode(kwargs, indent=4)

        # CPU-only solver
        self.parse_device_arg(**kwargs)
        if self.use_gpu:
            raise ValueError("bnb solver supports CPU only")
        self.device_name = 'cpu'

        self.timer = PerformanceTimer(self.use_gpu)

        self.N = problem.N
        self.Q = np.asarray(problem.build_Q_matrix(use_gpu=False))
        self.timer.pick_time('build_Q')

        best_energy, best_state = self._run_algorithm(**kwargs)
        self.timer.pick_time('solver_run')

        solution = self._pack_solution(problem, best_energy, best_state)
        self._cleanup()
        return solution

    def _run_algorithm(self, **kwargs):
        """Run the branch-and-bound solver."""
        split_depth = kwargs.get('split_depth', 10)
        num_threads = kwargs.get('num_threads', os.cpu_count() or 1)

        logger.debug(
            f'bnb.solve: N={self.N}, split_depth={split_depth}, num_threads={num_threads}'
        )

        result = solve_qubo_omp(
            self.Q,
            split_depth=int(split_depth),
            num_threads=int(num_threads),
        )

        best_energy = float(result['objective_value'])
        best_state = np.asarray(result['assignment'], dtype=np.int32)
        self.explored_nodes = int(result['explored_nodes'])

        return best_energy, best_state

    def _pack_solution(self, problem: ProblemABC, best_energy, best_state):
        """Convert the solver output into a Solution instance."""
        best_state = np.asarray(best_state, dtype=np.int32).reshape(self.N)

        epochs_cnt = self.kwargs.get('epochs_cnt', 1)
        solution = Solution(
            N=self.N,
            epochs_cnt=epochs_cnt,
            samples_cnt=1,
            problem_instance_json=problem.to_json(),
            solver_params_json=self.json_str,
        )
        solution.state_vectors = StateVectors(
            wide_vector_matrix=best_state.reshape(1, self.N)
        )
        solution.energies = np.array([best_energy], dtype=self.Q.dtype)

        use_metrics = self.kwargs.get('use_metrics', set())
        if use_metrics == 'all':
            use_metrics = ALL_METRICS
        metric_dict = {}
        if 'explored_nodes' in use_metrics:
            metric_dict['explored_nodes'] = self.explored_nodes
        solution.metrics = [metric_dict]

        solution.solver_class = self.__class__.__name__
        solution.best_E = best_energy
        solution.best_X = best_state
        solution.ising_offset = float(
            problem.get_Q_Ising_offset(self.gimme_lib(), self.Q)
        )
        if hasattr(problem, 'prefix_E'):
            solution.prefix_E = problem.prefix_E
        solution.time_to_solve_ms = self.timer.get_time_from_start()
        solution.timer = [(t, l) for t, l in self.timer.times if l != 'epoch']

        return solution

    def _cleanup(self):
        """No GPU cleanup needed for CPU-only solver."""
        pass
