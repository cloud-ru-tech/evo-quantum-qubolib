from ..abc import ProblemABC, SolverABC
from ..solution import Solution
from ..solvers import MSSA, QBF, QB_F1, BnB
from .scheduler import devices_scheduler
from ..state_vectors import StateVectors
from ..problems import ReducedProblem
import jsonpickle
import numpy as np
from numpy.random import Generator, PCG64
from typing import Type, List


# Solver parallelization approaches
SAMPLER_SOLVERS = dict(MSSA = MSSA)  # Parallelize by splitting samples 
BRUTEFORCE_SOLVERS = dict(QBF = QBF, QB_F1 = QB_F1, BnB = BnB)  # Parallelize by bit-fixing decomposition aka Embarrassingly Parallel Enumeration 

SUPPORTED_SOLVERS = dict(**SAMPLER_SOLVERS, **BRUTEFORCE_SOLVERS)


class Parallel(SolverABC):
    """
    This class solving the QUBO problem across multiple devices in parallel in a map-reduce manner, while retaining the standard solver interface.
    Each device runs its own dedicated worker process. The total workload is divided into splits_cnt tasks and distributed among the workers.
    Two parallelization approaches are supported:
    1. Sample splitting (for samplers like MSSA): Total number of samples is equally divided into tasks.
    2. Bit-fixing decomposition (for bruteforces like QBF, QB_F1): k bits are fixed to reduce the problem to splits_cnt = 2^k subproblems of size N-k.
        Cause BnB is exact solver, there is no sense to parallel it by samples. So it is included to 'brutforcers' set

    Parameters:
    -----------

    Examples:
    ---------
    >>> # CPU-based parallel processing
    >>> solver = Parallel(base_solver=QBF, splits_cnt=64, devices=['cpu']*8)
    >>> solution = solver.solve(problem)
    >>> 
    >>> # GPU-based parallel processing 
    >>> solver = Parallel(base_solver=MSSA, splits_cnt=15, devices=['cuda:0', 'cuda:3'])
    >>> solution = solver.solve(problem, total_samples_cnt=3000)
    >>> 
    >>> # Automatic allocation of all available GPUs
    >>> solver = Parallel(base_solver=QB_F1, splits_cnt=4, devices=['cuda']*4)
    >>> solution = solver.solve(problem)
    """


    def __init__(self, base_solver: Type[SolverABC], splits_cnt: int, devices: List[str]):
        """
        Initialize parallel solver with base solver type, split count, and devices.
        
        Creates a parallel execution environment where the problem is divided into splits_cnt
        subproblems, distributed across the specified devices. Each device runs one or more
        instances of the base_solver to process its assigned subproblems.
        
        For sampler solvers (MSSA): Work is divided by splitting samples among devices.
        
        Args:
            solver_type : type, required. Supported values: MSSA, QBF, QB_F1, BnB.

            splits_cnt : int, required
                Number of splits to divide total workload.
                For bruteforce solvers (QBF, QB_F1) and BnB must be a power of 2.

            devices : List of devices to distribute work across (e.g., ['cuda']*2, ['cuda:0', 'cuda:1'], or ['cpu']*8), required          
        """
        self._base_solver = base_solver
        self._splits_cnt = splits_cnt
        self._devices = devices
        self._validate_config()

    def _validate_config(self):
        SUPPORTED_SOLVERS_REVERSE = {v: k for k, v in SUPPORTED_SOLVERS.items()}        
        BRUTEFORCE_SOLVERS_REVERSE = {v: k for k, v in BRUTEFORCE_SOLVERS.items()}
	
        if self._base_solver not in SUPPORTED_SOLVERS_REVERSE:
            raise ValueError(f"Unsupported solver class: {self._base_solver}. "
                            f"Supported solvers: {', '.join(SUPPORTED_SOLVERS_REVERSE)}")
        self._solver_class_name = SUPPORTED_SOLVERS_REVERSE[self._base_solver]
        

        if not isinstance(self._splits_cnt, int) or self._splits_cnt <= 0:
            raise ValueError("splits_cnt must be a positive integer")
        
        if self._base_solver in BRUTEFORCE_SOLVERS_REVERSE:
            is_power_of_two = self._splits_cnt & (self._splits_cnt - 1) == 0
            if not is_power_of_two:
                raise ValueError(f"For bruteforce solvers with Embarrassingly Parallel Enumeration / "
                                f"Bit-Fixing Decomposition, "
                                f"splits_cnt must be a power of 2, got {self._splits_cnt}")
        
        if not self._devices:
            raise ValueError("Devices list cannot be empty")
        
        has_cpu = any('cpu' in d for d in self._devices)
        has_gpu = any('cuda' in d for d in self._devices)
        
        if has_cpu and has_gpu:
            raise ValueError("Cannot mix CPU and GPU devices")
        
        if all(d == 'cuda' for d in self._devices):
            self._devices = [f'cuda:{i}' for i in range(len(self._devices))]
        else:
            self._devices = self._devices.copy()

    def solve(self, problem: ProblemABC, timeout=None, **solver_type_specific_kwargs) -> Solution:
        internal_solver = _ParallelInternal()
        return internal_solver.solve(
            problem,
            solver_type=self._solver_class_name,
            splits_cnt=self._splits_cnt,
            devices=self._devices,
            timeout=timeout,
            **solver_type_specific_kwargs
        )
    

class _ParallelInternal(SolverABC):

    def __init__(self):
        # Do not ever use args and kwargs in solver __init__, cause it leads to change proto grpc
        pass
        
      
    def solve(self, problem: ProblemABC, solver_type=None, splits_cnt=None, devices=None, timeout=None, **solver_type_specific_kwargs) -> Solution:
        # Do not ever use any args in solve(), only kwargs please
        kw = dict(solver_type=solver_type, splits_cnt=splits_cnt, devices=devices, timeout=timeout, **solver_type_specific_kwargs)
        self.json_str = jsonpickle.encode(kw, indent=4)

        self.validate_args(solver_type, splits_cnt, devices)
        # Тут нужно просто задиспетчерить по какой схеме параллелим, по семплам, или по фиксированным битам.
        if self.solver_type in SAMPLER_SOLVERS:
            self.task_seeds = self.prepare_rand_seeds(**solver_type_specific_kwargs)
            tasks = dict()
            if 'total_samples_cnt' in solver_type_specific_kwargs:
                total_samples_cnt = solver_type_specific_kwargs['total_samples_cnt']
                samples_cnts = [total_samples_cnt // splits_cnt + (1 if i < total_samples_cnt % splits_cnt else 0) for i in range(splits_cnt)]
            for task_id in range(splits_cnt):
                kw = dict(**solver_type_specific_kwargs)
                if 'total_samples_cnt' in solver_type_specific_kwargs:
                    kw['samples_cnt'] = samples_cnts[task_id]
                kw.update(rand_seed = self.task_seeds[task_id])
                tasks[task_id] = ((solver_type, problem), kw)
            results = devices_scheduler(self.devices, tasks, do_solve, timeout=timeout)   # Here results[task_id] = (worker_id, partial_solution)
            solution = self.merge_sampler_partial_solutions(results)
        if self.solver_type in BRUTEFORCE_SOLVERS:
            tasks = dict()
            reduced_problems = []
            A = splits_cnt.bit_length() - 1
            for task_id in range(splits_cnt):
                kw = dict(**solver_type_specific_kwargs)
                bit_str = format(task_id, f'0{A}b')
                prefix = np.array(list(map(int, bit_str)))
                reduced_problem = ReducedProblem(problem, prefix)
                reduced_problems += [reduced_problem]
                tasks[task_id] = ((solver_type, reduced_problem), kw)
            results = devices_scheduler(self.devices, tasks, do_solve, timeout=timeout)   # Here results[task_id] = (worker_id, partial_solution)
            solution = self.merge_bruteforcer_partial_solutions(results, problem)
        return solution
    

    def prepare_rand_seeds(self, **solver_type_specific_kwargs):
        rand_seed = solver_type_specific_kwargs.get('rand_seed', None)        

        if isinstance(rand_seed, int):
            task_seeds = [rand_seed]
        elif rand_seed is None:
            rng = Generator(PCG64(42))
            task_seeds = rng.integers(0, 2**32, size=self.splits_cnt).tolist()
        else: 
            task_seeds = rand_seed

        correct = hasattr(task_seeds, '__len__') and len(task_seeds) == self.splits_cnt
        if not correct:
            raise ValueError(f'Incorrect rand_seed arg, None or List len {self.splits_cnt} expected, but {rand_seed} got')
        return task_seeds            


    def validate_args(self, solver_type, splits_cnt, devices):
        # Проверка solver (обязательный параметр)
        if solver_type not in SUPPORTED_SOLVERS:
            raise ValueError(f"Unsupported solver class: {solver_type}. "
                            f"Supported solvers: {', '.join(SUPPORTED_SOLVERS)}")
        self.solver_type = solver_type

        # Проверка splits_cnt с учетом стратегии параллелизации
        if not isinstance(splits_cnt, int) or splits_cnt <= 0:
            raise ValueError("splits_cnt must be a positive integer")
        
        # Для брутфорсеров проверяем что число батчей - степень двойки
        if self.solver_type in BRUTEFORCE_SOLVERS:
            is_power_of_two = splits_cnt & (splits_cnt - 1) == 0
            if not is_power_of_two:
                raise ValueError(f"For bruteforce solvers with Embarrassingly Parallel Enumeration / "
                                f"Bit-Fixing Decomposition, "
                                f"splits_cnt must be a power of 2, got {splits_cnt}")
        
        self.splits_cnt = splits_cnt        

        if not devices:
            raise ValueError("Devices list cannot be empty")
        
        # Проверяем что не смешаны cpu и gpu
        has_cpu = any('cpu' in d for d in devices)
        has_gpu = any('cuda' in d for d in devices)
        
        if has_cpu and has_gpu:
            raise ValueError("Cannot mix CPU and GPU devices")
        
        # Если все cuda, преобразуем в cuda:0, cuda:1, ...
        if all(d == 'cuda' for d in devices):
            self.devices = [f'cuda:{i}' for i in range(len(devices))]
        else:
            self.devices = devices.copy()
        self.num_workers = len(self.devices)


    def merge_partial_solutions(self, results, problems):
        if self.solver_type in BRUTEFORCE_SOLVERS:
            return self.merge_bruteforcer_partial_solutions(results, problems)
        elif self.solver_type in SAMPLER_SOLVERS:
            return self.merge_sampler_partial_solutions(results)
        else:
            raise ValueError('Something wrong')

    def merge_bruteforcer_partial_solutions(self, results, problem):
        # results: dict, where results[task_id] = (worker_id, partial_solution)
        partial_solutions = [result[1] for result in results.values()]
      
        # Create new solution with aggregated parameters
        merged_solution = Solution(
            N=problem.N,
            epochs_cnt=self.splits_cnt,
            samples_cnt=1,
            problem_instance_json=problem.to_json(),
            solver_params_json=self.json_str
        )
        # Copy other attributes
        merged_solution.ising_offset = None # problem.get_Q_Ising_offset(np)
        merged_solution.solver_class = partial_solutions[0].solver_class
        
        # Find best solution among all partial solutions
        best_E = float('inf')
        best_X = None
        best_task_id = None
        
        # Collect metrics data
        energy_arr = []
        t_arr = []

        for task_id, (worker_id, partial_solution) in results.items():
            partial_prefix_E = partial_solution.prefix_E
            
            # Get the best energy from this partial solution
            partial_metrics = partial_solution.metrics[0]
            
            if 'energy_arr' in partial_metrics:
                partial_energy_arr = partial_metrics['energy_arr']
                adjusted_energy_arr = partial_energy_arr + partial_prefix_E
                energy_arr.extend(adjusted_energy_arr)
            
            if 'time_arr' in partial_metrics:
                t_arr.extend(partial_metrics['time_arr'])

        # Set metrics
        merged_solution.metrics = [{}]

        if t_arr:  # Only add t_arr if we collected any data
            merged_solution.metrics[0]['time_arr'] = np.array(t_arr)

        if energy_arr:  # Only add energy_arr if we collected any data
            merged_solution.metrics[0]['energy_arr'] = np.array(energy_arr)

        for task_id, (worker_id, partial_solution) in results.items():
            partial_prefix_E = partial_solution.prefix_E # Здесь используется неявный контракт - ожидаем что солвер положил в инстанс поле prefix_E, если оно было у проблемы.
            # А у ReducedProblem поле prefix_E обязательно есть на момент когда решние найдено и солвер собирает солюшн

            partial_best_E = partial_solution.best_E
            adjusted_best_E = partial_best_E + partial_prefix_E
            
            # Update global best if this is better
            if adjusted_best_E < best_E:
                best_E = adjusted_best_E
                best_X = partial_solution.best_X.copy()
                best_task_id = task_id


        # Set the best solution fields
        merged_solution.best_E = best_E
        merged_solution.energies = np.array([best_E])
        
        # Reconstruct full solution vector
        if best_X is not None:
            # Get the prefix bits for the best task
            A = self.splits_cnt.bit_length() - 1
            bit_str = format(best_task_id, f'0{A}b')
            prefix = np.array(list(map(int, bit_str)))
            
            # Concatenate prefix with the suffix from the partial solution
            full_best_X = np.hstack([prefix, best_X])
            merged_solution.best_X = full_best_X
            
            # Create state vectors
            wide_matrix = full_best_X.reshape(1, -1)
            merged_solution.state_vectors = StateVectors(wide_vector_matrix=wide_matrix)
        
                
        # Calculate total time
        worker_times = {}
        for task_id, (worker_id, partial_solution) in results.items():
            worker_times[worker_id] = worker_times.get(worker_id, 0) + partial_solution.time_to_solve_ms
        merged_solution.time_to_solve_ms = max(worker_times.values()) if worker_times else 0
        
        # Set worker logs
        merged_solution.worker_logs = []
        for task_id, (worker_id, sol) in results.items():
            wl = dict(task_id=task_id, worker_id=worker_id, device=self.devices[worker_id])
            wl.update(time_to_solve_ms=sol.time_to_solve_ms)
            merged_solution.worker_logs += [wl]
        
        return merged_solution    

    def merge_sampler_partial_solutions(self, results):
        # results: dict, where results[task_id] = (worker_id, partial_solution)
        # Extract solutions from results
        partial_solutions = [result[1] for result in results.values()]
        
        # Get the first solution as template for merged solution
        template_solution = partial_solutions[0]
        
        # Create new solution with aggregated parameters
        total_samples_cnt = sum(sol.samples_cnt for sol in partial_solutions)
        
        merged_solution = Solution(
            N=template_solution.N,
            epochs_cnt=template_solution.epochs_cnt,
            samples_cnt=total_samples_cnt,
            problem_instance_json=template_solution.problem_instance_json,
            solver_params_json=self.json_str
        )
        
        # Merge metrics
        merged_solution.metrics = []
        for sol in partial_solutions:
            merged_solution.metrics.extend(sol.metrics)
        
        # Merge state vectors
        if all(sol.state_vectors is not None for sol in partial_solutions):
            wide_vectors_list = [sol.state_vectors.get_wide_vectors() for sol in partial_solutions]
            merged_wide_vectors = np.vstack(wide_vectors_list)
            merged_solution.state_vectors = StateVectors(wide_vector_matrix=merged_wide_vectors)
        
        # Merge energies
        if all(sol.energies is not None for sol in partial_solutions):
            merged_solution.energies = np.hstack([sol.energies for sol in partial_solutions])
        
        # Copy other attributes
        merged_solution.ising_offset = template_solution.ising_offset
        merged_solution.solver_class = template_solution.solver_class

        worker_times = {}
        for task_id, (worker_id, partial_solution) in results.items():
            worker_times[worker_id] = worker_times.get(worker_id, 0) + partial_solution.time_to_solve_ms
        merged_solution.time_to_solve_ms = max(worker_times.values()) if worker_times else 0

        merged_solution.worker_logs = []
        for task_id, (worker_id, sol) in results.items():
            wl = dict(task_id=task_id, worker_id=worker_id, device=self.devices[worker_id])
            wl.update(rand_seed=self.task_seeds[task_id], time_to_solve_ms=sol.time_to_solve_ms)
            merged_solution.worker_logs += [wl]

        # Find best solution among all samples
        if merged_solution.energies is not None and len(merged_solution.energies) > 0:
            best_idx = np.argmin(merged_solution.energies)
            merged_solution.best_E = merged_solution.energies[best_idx]
            if merged_solution.state_vectors is not None:
                merged_solution.best_X = merged_solution.state_vectors.get_wide_vectors()[best_idx]
        
        return merged_solution        


def do_solve(device, solver_type, problem, **solver_type_specific_kwargs):
    solver = SUPPORTED_SOLVERS[solver_type]()
    kw = dict(**solver_type_specific_kwargs, device=device)
    solution = solver.solve(problem, **kw)
    return solution
