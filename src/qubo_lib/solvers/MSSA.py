import jsonpickle
from ..utils import PerformanceTimer
from ..abc import ProblemABC, SolverABC
from ..solution import Solution
from ..state_vectors import StateVectors
from collections import namedtuple
import numpy as np
from numpy.random import Generator, PCG64
from typing import Type
import logging

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[logging.StreamHandler()]
# )
logger = logging.getLogger(__name__)

ALL_METRICS = set(['t_arr', 'energy_arr', 'flip_arr', 'time_arr'])


def run_MSSA(Q, states_cnt=200, epochs_cnt=10, iters_per_epoch=10 ** 6, T=25, rand_seed=42, need_intrmd_results=False):
    logger.warning('run_MSSA() is obsolete. Better use\n    solver = MSSA()\n    solution = solver.solve(problem)')
    return MSSA.run_MSSA(Q, states_cnt, epochs_cnt, iters_per_epoch, T, rand_seed, need_intrmd_results)

def defineMaxAvailableSharedMem(block_size=128, size_of_item=4):
    return MSSA.defineMaxAvailableSharedMem(block_size, size_of_item)

def dump_kernel_args_to_dict(N, iters_per_epoch, d_X, d_best_X, d_diag_Q, d_QQT, d_x_dot_QQT, T, d_thread_randseed, d_E, d_best_E, time_ms = -1, epoch=-1):
    return MSSA.dump_kernel_args_to_dict(N, iters_per_epoch, d_X, d_best_X, d_diag_Q, d_QQT, d_x_dot_QQT, T, d_thread_randseed, d_E, d_best_E, time_ms, epoch)

def run_MSSA_multiple_epoch(Q, states_cnt = 200, epochs_cnt = 10, iters_per_epoch = 10**6, T = 25, rand_seed = 42, block_size = 128, need_intrmd_results = True):
    return MSSA.run_MSSA_multiple_epoch(Q, states_cnt, epochs_cnt, iters_per_epoch, T, rand_seed, block_size, need_intrmd_results)

def run_MSSA_multiple_epoch_slow_and_large(Q, states_cnt = 200, epochs_cnt = 10, iters_per_epoch = 10**6, T = 25, rand_seed = 42, block_size = 128, need_intrmd_results = True):
    return MSSA.run_MSSA_multiple_epoch_slow_and_large(Q, states_cnt, epochs_cnt, iters_per_epoch, T, rand_seed, block_size, need_intrmd_results)

class MSSA(SolverABC):
    """
    Simulated Annealing solver for QUBO problems.

    This solver implements a parallelized Simulated Annealing algorithm.
    Operates exclusively on GPU, supports only float32 data type. Handles dense matrices sizes
    up to 12k variables (up to 20k on A100). Processes multiple independent samples in parallel. 
    Uses controlled randomization. Optionally returns intermediate states and energies across epochs    
    """


    def __init__(self):
        # Do not ever use args and kwargs in solver __init__, cause it leads to change proto grpc
        super(MSSA, self).__init__()
        self.timer = PerformanceTimer()

    def parse_device_arg(self, **kwargs):
        if 'devices' in kwargs:
            raise ValueError('Use device argument')

        device = kwargs.get('device', 'cuda')
        if device == 'cpu':
            raise ValueError(f"MSSA solver is GPU only")
        elif device[:4] == 'cuda':
            self.use_gpu = True
            if len(device) > 5:
                gpu_id = int(device[5:])
            else:
                gpu_id = 0
            self.init_gpu(gpu_id)
        else:
            raise ValueError(f"Unknown device name: {device}")
        self.device_name = device

    @staticmethod
    def get_temperature_range(Q, a, b, epochs_cnt = 2):
        """
        Compute initial and final temperatures for simulated annealing
        based on the given QUBO matrix.

        Temperatures are derived from the Boltzmann formula so that:
        - At T_init the probability of accepting the maximum possible ΔE is *a*.
        - At T_final the probability of accepting the minimum non-zero ΔE is *b*.

        Parameters
        ----------
        a : float, optional
            Acceptance probability for max ΔE at T_init, must be in (0, 1).
        b : float, optional
            Acceptance probability for min ΔE at T_final, must be in (0, 1).

        Returns
        -------
        T_init : float
            Initial temperature.
        T_final : float
            Final temperature.
        """
        if not (0 < a < 1):
            raise ValueError(f"Parameter 'a' must be in (0, 1), got {a}")

        if not (0 < b < 1):
            raise ValueError(f"Parameter 'b' must be in (0, 1), got {b}")

        Q_sym = (Q + Q.T) / 2.0

        delta_E_max = float(abs(Q_sym).sum(axis=1).max())

        nonzero = abs(Q_sym)[abs(Q_sym) > 0]
        delta_E_min = float(nonzero.min() if len(nonzero) > 0 else 1e-6)

        T_init = -delta_E_max / np.log(a)
        T_final = -delta_E_min / np.log(b)

        result = np.geomspace(T_init, T_final, num=epochs_cnt)

        return result

    def solve(self, problem: ProblemABC, samples_cnt=200, epochs_cnt=10, iters_per_epoch=10 ** 6, T=None, rand_seed=42, **kwargs) -> Solution:
        """
        Solve the given QUBO problem using Simulated Annealing approach.
    
        Args:
            problem: QUBO problem to solve
            samples_cnt: Number of parallel annealing chains (default: 200)
            epochs_cnt: Number of temperature steps (default: 10)
            iters_per_epoch: Iterations per temperature step (default: 10^6)
                Thus total SA iterations is = epochs_cnt * iters_per_epoch
            T: Temperature schedule — scalar or array of length epochs_cnt.
                If None, temperatures are derived automatically from the
                QUBO matrix via :meth:`get_temperature_range`. The parameters
                *a* and *b* can be passed through **kwargs to control the
                automatic range.
            rand_seed: Random seed for chain initialization and reproducibility (default: 42)
            **kwargs: Additional parameters:
                device: Device specification ('cpu', 'cuda', 'cuda:0', etc.)
                need_intrmd_results: Whether to collect all chains intermediate states and energies after every epoch
                use_metrics: Set of metrics to include in solution ('all' or subset of ['t_arr', 'energy_arr', 'flip_arr', 'time_arr'])
                a: acceptance probability for max ΔE at T_init (default 0.5).
                b: acceptance probability for min ΔE at T_final (default 0.001).
                
        Returns:
            Solution object containing the optimal energy, all samples state vectors and energies, metrics, and metadata about the solving process.
        """        
        # Do not ever use any args in solve(), only kwargs please

        import numpy as np

        self.parse_device_arg(**kwargs)
        # logger.debug(f'MSSA.solve(), T={T} 9283652095')
        super_kwargs = kwargs.copy()
        super_kwargs.update(samples_cnt=samples_cnt, epochs_cnt=epochs_cnt, iters_per_epoch=iters_per_epoch, T=T, rand_seed=rand_seed)
        empty_solution = super(MSSA, self).solve(problem, **super_kwargs)

        if T is None:
            a = kwargs.pop('a', 0.5)
            b = kwargs.pop('b', 0.001)
            T = MSSA.get_temperature_range(self.Q, a=a, b=b, epochs_cnt=epochs_cnt)

        self.timer.pick_time('start')
        mssa_rez_list = run_MSSA(self.Q, samples_cnt, epochs_cnt, iters_per_epoch, T, rand_seed, need_intrmd_results=kwargs.get('need_intrmd_results', False))
        # logger.debug(f'MSSA.solve(), {mssa_rez_list} 078950247')
        self.mssa_rez_list = mssa_rez_list
        self.timer.pick_time('finish')
        solution = self.pack_results_to_solution_instance(empty_solution, problem, mssa_rez_list, **kwargs)
        # logger.debug(f'MSSA.solve(), {solution.__str__()} 08235628356')
        return solution

    def pack_results_to_solution_instance(self, empty_solution, problem, mssa_rez_list, **kwargs):
        try:
            import cupy as cp
        except:
            raise

        use_metrics = kwargs.get('use_metrics', set())
        if use_metrics == 'all':
            use_metrics = ALL_METRICS

        solution = empty_solution
        solution.metrics = MSSA.parse_MSSA_results(mssa_rez_list)
        
        solution.energies = np.zeros(solution.samples_cnt)
        wide_matrix = np.zeros(shape = (solution.samples_cnt, self.N), dtype=np.int32)
        for sample_idx in range(solution.samples_cnt):
            wide_matrix[sample_idx] = solution.metrics[sample_idx].pop('final_state')
            solution.energies[sample_idx] = solution.metrics[sample_idx]['energy_arr'][-1]

        # Filter metrics based on use_metrics
        for sample_idx in range(solution.samples_cnt):
            metrics_to_remove = set(solution.metrics[sample_idx].keys()) - use_metrics
            for metric_key in metrics_to_remove:
                solution.metrics[sample_idx].pop(metric_key)

        solution.state_vectors = StateVectors(wide_vector_matrix=wide_matrix)
        solution.solver_class = 'MSSA'
        best_idx = np.argmin(solution.energies)
        solution.best_X = wide_matrix[best_idx]
        solution.best_E = solution.energies[best_idx]
        solution.ising_offset = problem.get_Q_Ising_offset(cp, self.Q)   
        if hasattr(problem, 'prefix_E'):
            solution.prefix_E = problem.prefix_E
        # solution.time_to_solve_ms = mssa_rez_list[-1]['total_time_ms']
        solution.time_to_solve_ms = self.timer.get_time_from_start()
        return solution

    @staticmethod
    def parse_MSSA_results(rez_list):
        result = []
        epochs_cnt = len(rez_list)
        samples_cnt = rez_list[-1]['d_E'].shape[0]
        iters_per_epoch = rez_list[-1]['iters_per_epoch']
        E_matrix = np.zeros((epochs_cnt, samples_cnt))
        T_matrix = np.zeros((epochs_cnt, samples_cnt))

        if epochs_cnt == 1:
            E_matrix = rez_list[-1]['d_E']
            T_matrix = rez_list[-1]['time_ms']
            temp_arr = rez_list[-1]['T'][0:1]
            for smp in range(samples_cnt):
                rez_dct = dict(flip_arr=np.array([iters_per_epoch]), time_arr=np.array([T_matrix]),
                               energy_arr=np.array([E_matrix[smp]]))
                rez_dct.update(t_arr=temp_arr)
                rez_dct.update(final_state=rez_list[-1]['d_X'][smp])
                result.append(rez_dct)

            return result

        E_matrix = np.zeros((epochs_cnt - 1, samples_cnt))
        T_matrix = np.zeros((epochs_cnt - 1, samples_cnt))
        temp_arr = np.zeros(epochs_cnt - 1)

        for ep, epoch_rez in enumerate(rez_list[1:]):  # Отбрасываем стартовое состояние
            E_matrix[ep] = epoch_rez['d_E']
            T_matrix[ep] = epoch_rez['time_ms']
            temp_arr[ep] = epoch_rez['T'][0]

        flip_arr = np.ones(epochs_cnt - 1) * iters_per_epoch
        for smp in range(samples_cnt):
            rez_dct = dict(flip_arr=flip_arr.copy(), time_arr=T_matrix[:, smp], energy_arr=E_matrix[:, smp])
            rez_dct.update(t_arr=temp_arr)
            rez_dct.update(final_state=rez_list[-1]['d_X'][smp])
            result.append(rez_dct)

        return result

    @staticmethod
    def run_MSSA(Q, states_cnt = 200, epochs_cnt = 10, iters_per_epoch = 10**6, T = 25, rand_seed = 42, need_intrmd_results = False):
        """
        This is MultiStart Simulated Annealing. It runs several consequently epochs due temperature schedule
        Between epoch it corrects energies, cause accumulated error have to be eliminated.

        This kernel process dense Q matrix. Problem size in this version is limited 12k spins on Nvidia V100, RTX 4070/80/90 GPU,
        and 20k spins under Nvidia A100.
        Here run two different kernels - fast for problem size < 8k, and slow for problem size in (8k..12k]  / (16k..20k] for A100

        :param Q: QUBO matrix, np.ndarray, square
        :param states_cnt: Number of simultaneous running SA states. Multistarts
        :param epochs_cnt: Number of temperature epochs, aka temperature schedule size
        :param iters_per_epoch: SA iterations per one schedule epoch
               so total_iters = epochs_cnt * iters_per_epoch
        :param T: Temperature schedule. Can be scalar, or np.ndarray epochs_cnt size
        :param rand_seed: random seed for generate initial states
        :param need_intrmd_results: this flag ask to return intermediate results
                If it is True - states and energies before and after all epochs will be saved
                Else only states and energies after last epoch returns
        :return: list of dictionaries. It contains epochs_cnt+1 if need_intrmd_results == True, and 1 item if need_intrmd_results == False
                0th list item contain state before kernel start, -1th item is final state

                Every dictionary in list contains these keys:
                    N - problem size, number of spins, Q.shape[0]
                    iters_per_epoch - equal iters_per_epoch argument
                    d_E - states_cnt shape np.array, contains energy for corresponding X-vector. d_E[i] == d_X[i] @ Q @ d_X[i].T
                    d_X - 2d-np.array, d_X.shape == (states_cnt, N). It stores X-vector for each state at the moment after epoch * iters_per_epoch.
                          In other words - at the end of actual epoch.
                    d_best_E - (states_cnt,) shaped np.array, contains best founded energy for each state at the end of actual epoch
                    d_best_X - It likes d_X but stores states having best founded energy for each state. d_best_E[i] == d_best_X[i] @ Q @ d_best_X[i].T
                    T - np.array of state temperatures due epoch. T.shape == (states_cnt,). Can be None for state before start
                    d_thread_randseed - please do not use
                    time_ms - kernel execution time in milliseconds due epoch. Can be -1 for state before start
                    epoch - Number of passed epoch, Can be -1 for state before start, 0 for dict after first epoch, epoch_cnt-1 for final dict
                    total_time_ms - sum of kernel execution time (time_ms key) over all epochs
        """

        N = Q.shape[0]
        available_shared_mem = MSSA.defineMaxAvailableSharedMem()
        need_shared_mem = 3 * N * 4 #sizeof(np.float32)
        if available_shared_mem >= need_shared_mem:
            return MSSA.run_MSSA_multiple_epoch(Q, states_cnt, epochs_cnt, iters_per_epoch, T, rand_seed, 128, need_intrmd_results)
        else:
            return MSSA.run_MSSA_multiple_epoch_slow_and_large(Q, states_cnt, epochs_cnt, iters_per_epoch, T, rand_seed, 128, need_intrmd_results)


    @staticmethod
    def defineMaxAvailableSharedMem(block_size = 128, size_of_item = 4):
        try:
            import cupy as cp
        except:
            raise

        kernel_static_shared_mem_reserve = (2 * block_size + 10) * size_of_item
        device = cp.cuda.Device()
        max_shared_mem = device.attributes['MaxSharedMemoryPerBlockOptin'] - kernel_static_shared_mem_reserve
        return max_shared_mem

    @staticmethod
    def dump_kernel_args_to_dict(N, iters_per_epoch, d_X, d_best_X, d_diag_Q, d_QQT, d_x_dot_QQT, T, d_thread_randseed, d_E, d_best_E, time_ms = -1, epoch=-1):
        rez = dict()
        rez['N'] = N
        rez['iters_per_epoch'] = iters_per_epoch
        rez['d_X'] = d_X.get()
        rez['d_best_X'] = d_best_X.get()
        rez['T'] = T if T is None else T.get()
        rez['d_thread_randseed'] = d_thread_randseed.get()
        rez['d_E'] = d_E.get()
        rez['d_best_E'] = d_best_E.get()
        rez['time_ms'] = time_ms
        rez['epoch'] = epoch
        return rez

    @staticmethod
    def run_MSSA_multiple_epoch(Q, states_cnt = 200, epochs_cnt = 10, iters_per_epoch = 10**6, T = 25, rand_seed = 42, block_size = 128, need_intrmd_results = True):
        # logger.debug(f"states_cnt={states_cnt}, epochs_cnt={epochs_cnt}, iters_per_epoch={iters_per_epoch}, T={T}, rand_seed={rand_seed}, block_size={block_size}, need_intrmd_results={need_intrmd_results} 890275257")

        try:
            import cupy as cp
        except:
            raise

        N = Q.shape[0]
        M = states_cnt
        Ts = (T * np.ones(epochs_cnt)).astype(np.float32)
        rng = Generator(PCG64(rand_seed))

        h_X = rng.random((M, N)).round().astype(np.float32)
        h_diag_Q = np.diag(Q).astype(np.float32)
        h_QQT = (Q + Q.T).astype(np.float32)
        h_thread_randseed = rand_seed + np.arange(states_cnt * block_size, dtype = np.uint32)

        d_X = cp.asarray(h_X)
        d_best_X = cp.asarray(h_X)
        d_diag_Q = cp.asarray(h_diag_Q)
        d_QQT = cp.asarray(h_QQT)
        d_thread_randseed = cp.asarray(h_thread_randseed)

        d_x_dot_QQT = (d_X @ d_QQT)
        d_E = 0.5 * cp.einsum('ij,ij->i', d_x_dot_QQT, d_X)
        d_best_E = cp.array(d_E, copy = True)

        grid_size = M
        shared_mem_bytes = 3 * N * 4 #sizeof(np.float32)

        MSSA_kernel_code = f'#define MSSA_kernel_BlockSize {block_size}\n' + MSSA_resumable_kernel_code_wo_BlockSize
        # MSSA_kernel = cp.RawKernel(MSSA_kernel_code, 'resumable_MSSA_kernel')
        # MSSA_kernel.max_dynamic_shared_size_bytes = defineMaxAvailableSharedMem()

        module = cp.RawModule(code=MSSA_kernel_code)
        module.compile()
        MSSA_kernel = module.get_function('resumable_MSSA_kernel')
        MSSA_kernel.max_dynamic_shared_size_bytes = defineMaxAvailableSharedMem()

        # logger.debug(f"run_MSSA_multiple_epoch() max_dynamic_shared_size_bytes={MSSA_kernel.max_dynamic_shared_size_bytes} 22464554246")
        # logger.debug(f"run_MSSA_multiple_epoch() MSSA_kernel_code={MSSA_kernel_code} 135786453")

        full_results = []
        if need_intrmd_results:
            kernel_args = (N, iters_per_epoch, d_X, d_best_X, d_diag_Q, d_QQT, d_x_dot_QQT, None, d_thread_randseed, d_E, d_best_E)
            full_results += [dump_kernel_args_to_dict(*kernel_args)]

        total_time_ms = 0
        for ep in range(epochs_cnt):
            d_T = cp.asarray(Ts[ep] * np.ones(M, dtype = np.float32))
            kernel_args = (N, iters_per_epoch, d_X, d_best_X, d_diag_Q, d_QQT, d_x_dot_QQT, d_T, d_thread_randseed, d_E, d_best_E)
            # if ep == 0:
            #     logger.debug(f"Kernel args: N: {N}\n iters_per_epoch: {iters_per_epoch}\n d_X: {d_X.dtype}  {d_X.shape}\n d_best_X: {d_best_X.dtype}  {d_best_X.shape}\n d_diag_Q: {d_diag_Q.dtype}  {d_diag_Q.shape}\n")
            #     logger.debug(f"d_QQT: {d_QQT.dtype}  {d_QQT.shape}\n d_x_dot_QQT: {d_x_dot_QQT.dtype}  {d_x_dot_QQT.shape}\n d_T: {d_T.dtype}  {d_T.shape}\n d_thread_randseed: {d_thread_randseed.dtype}  {d_thread_randseed.shape}\n")
            #     logger.debug(f"d_E: {d_E.dtype}  {d_E.shape}\n d_best_E: {d_best_E.dtype}  {d_best_E.shape}")
            #     logger.debug(f"T: {T.dtype}  {T.shape}\n Ts: {Ts.dtype}  {Ts.shape}")

            start_event = cp.cuda.Event()
            end_event = cp.cuda.Event()
            start_event.record()

            MSSA_kernel((grid_size,), (block_size,), kernel_args, shared_mem = shared_mem_bytes)

            end_event.record()
            end_event.synchronize()
            time_ms = cp.cuda.get_elapsed_time(start_event, end_event)
            total_time_ms += time_ms

            d_x_dot_QQT = (d_X @ d_QQT)
            d_E = 0.5 * cp.einsum('ij,ij->i', d_x_dot_QQT, d_X)
            d_best_E = 0.5 * cp.einsum('ij,jk,ik->i', d_best_X, d_QQT, d_best_X)

            if need_intrmd_results or ep == epochs_cnt-1:
                kernel_args = (N, iters_per_epoch, d_X, d_best_X, d_diag_Q, d_QQT, d_x_dot_QQT, d_T, d_thread_randseed, d_E, d_best_E)
                full_results += [MSSA.dump_kernel_args_to_dict(*kernel_args, epoch=ep, time_ms = time_ms)]

        full_results[-1]['total_time_ms'] = total_time_ms

        logger.debug(f"run_MSSA_multiple_epoch() {full_results[:3]} 62354928365")

        return full_results



    @staticmethod
    def run_MSSA_multiple_epoch_slow_and_large(Q, states_cnt = 200, epochs_cnt = 10, iters_per_epoch = 10**6, T = 25, rand_seed = 42, block_size = 128, need_intrmd_results = True):
        try:
            import cupy as cp
        except:
            raise ImportError

        N = Q.shape[0]
        M = states_cnt
        Ts = T * np.ones(epochs_cnt, dtype = np.float32)
        rng = Generator(PCG64(rand_seed))

        h_X = rng.random((M, N)).round().astype(np.float32)
        h_diag_Q = np.diag(Q).astype(np.float32)
        h_QQT = (Q + Q.T).astype(np.float32)
        h_thread_randseed = rand_seed + np.arange(states_cnt * block_size, dtype = np.uint32)

        d_X = cp.asarray(h_X)
        d_best_X = cp.asarray(h_X)
        d_QQT = cp.asarray(h_QQT)
        d_thread_randseed = cp.asarray(h_thread_randseed)

        d_x_dot_QQT = (d_X @ d_QQT)
        d_E = 0.5 * cp.einsum('ij,ij->i', d_x_dot_QQT, d_X)
        d_best_E = cp.array(d_E, copy = True)

        grid_size = M
        MSSA_module_code = f'#define MSSA_kernel_BlockSize {block_size}\n' + MSSA_resumable_kernel_code_slow_but_large

        MSSA_module = cp.RawModule(code=MSSA_module_code)
        MSSA_kernel = MSSA_module.get_function('resumable_MSSA_kernel')
        MSSA_kernel.max_dynamic_shared_size_bytes = MSSA.defineMaxAvailableSharedMem()

        shared_mem_bytes = 2 * N * 4 #sizeof(np.float32)
        MSSA_module_code = f'#define MSSA_kernel_BlockSize {block_size}\n' + MSSA_resumable_kernel_code_slow_but_large
        const_mem_ptr_diag_Q = MSSA_module.get_global('const_diag_Q')
        const_diag_Q = cp.ndarray((N,), cp.float32, const_mem_ptr_diag_Q)
        const_diag_Q = h_diag_Q

        full_results = []
        if need_intrmd_results:
            kernel_args = (N, iters_per_epoch, d_X, d_best_X, None, d_QQT, d_x_dot_QQT, None, d_thread_randseed, d_E, d_best_E)
            full_results += [MSSA.dump_kernel_args_to_dict(*kernel_args)]

        total_time_ms = 0
        for ep in range(epochs_cnt):
            d_T = cp.asarray(Ts[ep] * np.ones(M, dtype = np.float32))
            kernel_args = (N, iters_per_epoch, d_X, d_best_X, d_QQT, d_x_dot_QQT, d_T, d_thread_randseed, d_E, d_best_E)

            start_event = cp.cuda.Event()
            end_event = cp.cuda.Event()
            start_event.record()

            MSSA_kernel((grid_size,), (block_size,), kernel_args, shared_mem = shared_mem_bytes)

            end_event.record()
            end_event.synchronize()
            time_ms = cp.cuda.get_elapsed_time(start_event, end_event)
            total_time_ms += time_ms

            d_x_dot_QQT = (d_X @ d_QQT)
            d_E = 0.5 * cp.einsum('ij,ij->i', d_x_dot_QQT, d_X)
            d_best_E = 0.5 * cp.einsum('ij,jk,ik->i', d_best_X, d_QQT, d_best_X)

            if need_intrmd_results or ep == epochs_cnt-1:
                kernel_args = (N, iters_per_epoch, d_X, d_best_X, None, d_QQT, d_x_dot_QQT, d_T, d_thread_randseed, d_E, d_best_E)
                full_results += [MSSA.dump_kernel_args_to_dict(*kernel_args, epoch=ep, time_ms = time_ms)]

        full_results[-1]['total_time_ms'] = total_time_ms
        return full_results

MSSA_resumable_kernel_code_wo_BlockSize = """
extern "C" __global__ void resumable_MSSA_kernel( // This kernel is stateful. It can run, stop, and after stop - resume and run again
                   const int N,                         // N - spins count, state size
                   const unsigned long long it_cnt,     // iterations count
                   float* d_X,                          // M*N size matrix, rows are states. Initial states in, actual states out, it makes kernel resumable
                   float* d_best_X,                     // M*N size matrix, rows are states. Best states in not used, best found states out. Write only
                   const float* d_diag_Q,               // N size vector
                   const float* d_QQT,                  // N*N square matrix Q + Q.T
                   const float* d_x_dot_QQT,            // M*N matrix. Row i is X[i] @ QQT, X[i] is initial i-th state. X@QQT for initial d_X in
                   const float* d_T,                    // M size vector
                   unsigned int* d_thread_randseed,     // M * block_size == grid_size vector, randseed for each thread. In and out
                   const float* d_E,                    // M size vector, only E for initial states d_X in. Writing energies out is useless, cause error accumulates
                   const float* d_best_E)               // M size vector, E for best states d_best_X in. Writing energies out is useless, cause error accumulates
{
    const int state_idx = blockIdx.x;

    const int K = 1 + (N - 1) / MSSA_kernel_BlockSize; // Spins per one thread. Count of block runs per state
    // N in [0                         .. 1 * MSSA_kernel_BlockSize], K=1
    // N in [MSSA_kernel_BlockSize + 1 .. 2 * MSSA_kernel_BlockSize], K=2

    extern __shared__ float sharedMemory[];
    float* x = sharedMemory;               // local N size state, inited as X[state_idx] matrix row
    float* xQQT = sharedMemory + N;        // local N size x@QQT vector
    float* diag_Q = sharedMemory + 2*N;    // N size diag(Q) vector

    // read from d_X to x, K-times, BlockSize items, K * MSSA_kernel_BlockSize == N
    // also read d_xQQT, d_diag_Q
    int device_idx = state_idx * N + threadIdx.x;
    int i = threadIdx.x; // i always is spin idx in state
    for (int chunk_idx = 0; chunk_idx != K; ++chunk_idx) {
        if (i < N) {
            x[i] = d_X[device_idx];
            xQQT[i] = d_x_dot_QQT[device_idx];
            diag_Q[i] = d_diag_Q[i];
            device_idx += MSSA_kernel_BlockSize;
            i += MSSA_kernel_BlockSize;
        }
    }

    const float T = d_T[state_idx]; // T, best_E, E is one per block, but T is frequently used, so put it into registries
    __shared__ float best_E; // Only 0-th thread works process this variable
    __shared__ float E;      // Only 0-th thread works process this variable
    if (threadIdx.x == 0) {
        best_E = d_best_E[state_idx];
        E = d_E[state_idx];
    }

    unsigned int seed_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int seed = d_thread_randseed[seed_idx];
    // TODO: Better use CuRAND in release
    float dice;
    const unsigned int a = 1664525;
    const unsigned int c = 1013904223;
    const unsigned int m = 1u << 31;  // 2^32

    __shared__ int chosen_ones[MSSA_kernel_BlockSize];
    __shared__ float acc_Ws[MSSA_kernel_BlockSize];

    // Main loop
    for (int it_num = 0; it_num != it_cnt; ++it_num) {
        float acc_W = 0;
        int chosen_one = 0;

        int i = threadIdx.x;
        for (int chunk_idx = 0; chunk_idx != K; ++chunk_idx) {
            if (i < N) {
                float dE = diag_Q[i] + xQQT[i] * (1 - 2 * x[i]);
                float W = (dE > 0) ? expf(-dE / T) : 1.0F;
                acc_W += W;
                seed = (a * seed + c) % m;
                dice = 1.0F * seed / m;
                if (acc_W * dice < W)
                    chosen_one = i;
                i += MSSA_kernel_BlockSize;
            }
        }
        chosen_ones[threadIdx.x] = chosen_one;
        acc_Ws[threadIdx.x] = acc_W;
        __syncthreads();

        for (unsigned int shift = 1; shift < MSSA_kernel_BlockSize; shift *= 2) {
            if (((threadIdx.x + shift) < N) && (threadIdx.x % (2 * shift) == 0)) {
                seed = (a * seed + c) % m;
                dice = 1.0F * seed / m;
                acc_Ws[threadIdx.x] += acc_Ws[threadIdx.x + shift];
                if (acc_Ws[threadIdx.x] * dice < acc_Ws[threadIdx.x + shift])
                    chosen_ones[threadIdx.x] = chosen_ones[threadIdx.x + shift];
            }
            __syncthreads();
        }


        __shared__ int flip_idx;
        __shared__ float flip;
        if (threadIdx.x == 0) {
            flip_idx = chosen_ones[0]; // Yeah, we have choosen spin to flip
//            float prev_E = E;
            float delta_E = diag_Q[flip_idx] + xQQT[flip_idx] * (1 - 2 * x[flip_idx]);
            E += delta_E;
            x[flip_idx] = (x[flip_idx] == 0) ? 1 : 0;
            flip = 1 - 2 * x[flip_idx];
        }
        __syncthreads();

        const float* QQT_row_for_flip = d_QQT + N * flip_idx;
        i = threadIdx.x;
        for (int chunk_idx = 0; chunk_idx != K; ++chunk_idx) {
            if (i < N) {
                xQQT[i] -= flip * QQT_row_for_flip[i];
                i += MSSA_kernel_BlockSize;
            }
        }

        __shared__ bool need_update_d_X;
        if (threadIdx.x == 0) {
            need_update_d_X = (E < best_E);
            if (E < best_E)
                best_E = E;
        }
        __syncthreads();

        if (need_update_d_X) {
            int device_idx = state_idx * N + threadIdx.x;
            int i = threadIdx.x; // i always is spin idx in state
            for (int chunk_idx = 0; chunk_idx != K; ++chunk_idx) {
                if (i < N) {
                    d_best_X[device_idx] = x[i];
                    device_idx += MSSA_kernel_BlockSize;
                    i += MSSA_kernel_BlockSize;
                }
            }
        }

        __syncthreads();
    }

// Main loop is finished, now lets load state out of kernel

//     Write out enregies is useles, cause accumulated error
//    if (threadIdx.x == 0) {
//        d_best_E[state_idx] = best_E;
//        d_E[state_idx] = E;
//    }
    device_idx = state_idx * N + threadIdx.x;
    i = threadIdx.x; // i is always spin idx in state
    for (int chunk_idx = 0; chunk_idx != K; ++chunk_idx) {
        if (i < N) {
            d_X[device_idx] = x[i];
//            d_x_dot_QQT[device_idx] = xQQT[i];
            device_idx += MSSA_kernel_BlockSize;
            i += MSSA_kernel_BlockSize;
        }
    }

    d_thread_randseed[seed_idx] = seed;
}
"""


# This kernel version uses diag_Q array in constant memory. This is anti-pattern, so it is very slow
# But it expand problem size 1.5 times, up to 8k/16k (V100/A100)
MSSA_resumable_kernel_code_slow_but_large = """
#define MAX_SPINS_COUNT 3 * 4096
__constant__ float const_diag_Q[MAX_SPINS_COUNT];

extern "C" __global__ void resumable_MSSA_kernel( // This kernel is stateful. It can run, stop, and after stop - resume and run again
                   const int N,                         // N - spins count, state size
                   const unsigned long long it_cnt,     // iterations count
                   float* d_X,                          // M*N size matrix, rows are states. Initial states in, actual states out, it makes kernel resumable
                   float* d_best_X,                     // M*N size matrix, rows are states. Best states in not used, best found states out. Write only
                   const float* d_QQT,                  // N*N square matrix Q + Q.T
                   const float* d_x_dot_QQT,            // M*N matrix. Row i is X[i] @ QQT, X[i] is initial i-th state. X@QQT for initial d_X in
                   const float* d_T,                    // M size vector
                   unsigned int* d_thread_randseed,     // M * block_size == grid_size vector, randseed for each thread. In and out
                   const float* d_E,                    // M size vector, only E for initial states d_X in. Writing energies out is useless, cause error accumulates
                   const float* d_best_E)               // M size vector, E for best states d_best_X in. Writing energies out is useless, cause error accumulates
{
    const int state_idx = blockIdx.x;

    const int K = 1 + (N - 1) / MSSA_kernel_BlockSize; // Spins per one thread. Count of block runs per state
    // N in [0                         .. 1 * MSSA_kernel_BlockSize], K=1
    // N in [MSSA_kernel_BlockSize + 1 .. 2 * MSSA_kernel_BlockSize], K=2

    extern __shared__ float sharedMemory[];
    float* x = sharedMemory;               // local N size state, inited as X[state_idx] matrix row
    float* xQQT = sharedMemory + N;        // local N size x@QQT vector

    // read from d_X to x, K-times, BlockSize items, K * MSSA_kernel_BlockSize == N
    // also read d_xQQT, d_diag_Q
    int device_idx = state_idx * N + threadIdx.x;
    int i = threadIdx.x; // i always is spin idx in state
    for (int chunk_idx = 0; chunk_idx != K; ++chunk_idx) {
        if (i < N) {
            x[i] = d_X[device_idx];
            xQQT[i] = d_x_dot_QQT[device_idx];
            device_idx += MSSA_kernel_BlockSize;
            i += MSSA_kernel_BlockSize;
        }
    }

    const float T = d_T[state_idx]; // T, best_E, E is one per block, but T is frequently used, so put it into registries
    __shared__ float best_E; // Only 0-th thread works process this variable
    __shared__ float E;      // Only 0-th thread works process this variable
    if (threadIdx.x == 0) {
        best_E = d_best_E[state_idx];
        E = d_E[state_idx];
    }

    unsigned int seed_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int seed = d_thread_randseed[seed_idx];
    // TODO: Better use CuRAND in release
    float dice;
    const unsigned int a = 1664525;
    const unsigned int c = 1013904223;
    const unsigned int m = 1u << 31;  // 2^32

    __shared__ int chosen_ones[MSSA_kernel_BlockSize];
    __shared__ float acc_Ws[MSSA_kernel_BlockSize];

    // Main loop
    for (int it_num = 0; it_num != it_cnt; ++it_num) {
        float acc_W = 0;
        int chosen_one = 0;

        int i = threadIdx.x;
        for (int chunk_idx = 0; chunk_idx != K; ++chunk_idx) {
            if (i < N) {
                float dE = const_diag_Q[i] + xQQT[i] * (1 - 2 * x[i]);
                float W = (dE > 0) ? expf(-dE / T) : 1.0F;
                acc_W += W;
                seed = (a * seed + c) % m;
                dice = 1.0F * seed / m;
                if (acc_W * dice < W)
                    chosen_one = i;
                i += MSSA_kernel_BlockSize;
            }
        }
        chosen_ones[threadIdx.x] = chosen_one;
        acc_Ws[threadIdx.x] = acc_W;
        __syncthreads();

        for (unsigned int shift = 1; shift < MSSA_kernel_BlockSize; shift *= 2) {
            if (((threadIdx.x + shift) < N) && (threadIdx.x % (2 * shift) == 0)) {
                seed = (a * seed + c) % m;
                dice = 1.0F * seed / m;
                acc_Ws[threadIdx.x] += acc_Ws[threadIdx.x + shift];
                if (acc_Ws[threadIdx.x] * dice < acc_Ws[threadIdx.x + shift])
                    chosen_ones[threadIdx.x] = chosen_ones[threadIdx.x + shift];
            }
            __syncthreads();
        }


        __shared__ int flip_idx;
        __shared__ float flip;
        if (threadIdx.x == 0) {
            flip_idx = chosen_ones[0]; // Yeah, we have choosen spin to flip
//            float prev_E = E;
            float delta_E = const_diag_Q[flip_idx] + xQQT[flip_idx] * (1 - 2 * x[flip_idx]);
            E += delta_E;
            x[flip_idx] = (x[flip_idx] == 0) ? 1 : 0;
            flip = 1 - 2 * x[flip_idx];
        }
        __syncthreads();

        const float* QQT_row_for_flip = d_QQT + N * flip_idx;
        i = threadIdx.x;
        for (int chunk_idx = 0; chunk_idx != K; ++chunk_idx) {
            if (i < N) {
                xQQT[i] -= flip * QQT_row_for_flip[i];
                i += MSSA_kernel_BlockSize;
            }
        }

        __shared__ bool need_update_d_X;
        if (threadIdx.x == 0) {
            need_update_d_X = (E < best_E);
            if (E < best_E)
                best_E = E;
        }
        __syncthreads();

        if (need_update_d_X) {
            int device_idx = state_idx * N + threadIdx.x;
            int i = threadIdx.x; // i always is spin idx in state
            for (int chunk_idx = 0; chunk_idx != K; ++chunk_idx) {
                if (i < N) {
                    d_best_X[device_idx] = x[i];
                    device_idx += MSSA_kernel_BlockSize;
                    i += MSSA_kernel_BlockSize;
                }
            }
        }

        __syncthreads();
    }

// Main loop is finished, now lets load state out of kernel

//     Write out enregies is useles, cause accumulated error
//    if (threadIdx.x == 0) {
//        d_best_E[state_idx] = best_E;
//        d_E[state_idx] = E;
//    }
    device_idx = state_idx * N + threadIdx.x;
    i = threadIdx.x; // i is always spin idx in state
    for (int chunk_idx = 0; chunk_idx != K; ++chunk_idx) {
        if (i < N) {
            d_X[device_idx] = x[i];
//            d_x_dot_QQT[device_idx] = xQQT[i];
            device_idx += MSSA_kernel_BlockSize;
            i += MSSA_kernel_BlockSize;
        }
    }

    d_thread_randseed[seed_idx] = seed;
}
"""
