import jsonpickle
from ..utils import PerformanceTimer
from ..abc import ProblemABC, SolverABC
from ..solution import Solution
from ..state_vectors import StateVectors
from ..solvers import QBF
from collections import namedtuple
import numpy as np
import logging
import time

# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[logging.StreamHandler()]
# )
logger = logging.getLogger(__name__)

type_dict = {np.int16: 'short', np.int32: 'int', np.uint16: 'unsigned short', np.uint32: 'unsigned long', np.int64: 'long long int'}

class QB_F1(SolverABC):
    """
    State-of-the-art bruteforce solver for QUBO problems.
    
    Implements an optimized brute force approach that uses Gray code iteration, prefix-suffix decomposition and cached precomputations.
    Only GPU, only np.int16 supported.
    
    Examples:
    ---------
    >>> solver = QB_F1()
    >>> solution = solver.solve(problem, device='cuda')
    """    

    def __init__(self, **kwargs):
        super().__init__()
        self.ABBA = None
        self.BB = None
        self.AA = None
        self.best_E_vec = None
        self.best_x_A_vec = None
        self.best_x_B_vec = None
        self.gridDim_x = 256
        self.blockDim_x = 64
        self.energy_type = np.int16
        self.index_type = np.int32
        self.state_type = np.int32
        self.A = None
        self.B = None

        self.device_name = None
        self.cuda_device = None
        self.use_gpu = None

    def solve(self, problem: ProblemABC, **kwargs) -> Solution:
        """
        Solve the given QUBO problem using brute force approach.
        
        Args:
            problem: Problem instance to solve
            **kwargs: Additional solver parameters including:
                - device: Device specification ('cuda', 'cuda:0', etc.)
                - use_metrics: Set of metrics to collect ('time_arr', 'energy_arr', or 'all')
            
        Returns:
            Solution object containing the optimal energy, state vector, metrics, and metadata about the solving process.
        """      

        self.prepare(problem, **kwargs)
        QB_F1_kernel_code = QB_F1.gimme_QB_F1_kernel_code(self.N, self.A, self.gridDim_x, self.blockDim_x, 64)
        cp = self.gimme_lib()
        QB_F1_module = cp.RawModule(code=QB_F1_kernel_code, )
        QB_F1_kernel = QB_F1_module.get_function('QB_F1_kernel')
        self.timer.pick_time('Compile')

        kernel_args = (0, self.ABBA, self.BB, self.AA, self.best_E_vec, self.best_x_A_vec, self.best_x_B_vec)
        QB_F1_kernel((self.gridDim_x,), (self.blockDim_x,), kernel_args)
        self.timer.pick_time('Kernel run time')
        # self.post_process_result()

        solution = self.pack_results_to_solution_instance(problem)
        del self.ABBA, self.BB, self.AA
        cp.get_default_memory_pool().free_all_blocks()
        return solution


    def prepare(self, problem: ProblemABC, **kwargs):
        self.check_obsolete_args(**kwargs)

        self.json_str = jsonpickle.encode(kwargs, indent=4)
        self.timer = PerformanceTimer(self.use_gpu)
        self.parse_device_arg(**kwargs)

        self.N = problem.N
        self.B = 18
        self.A = self.N - self.B
        threads_cnt = self.gridDim_x * self.blockDim_x

        self.Q = problem.build_Q_matrix(use_gpu=self.use_gpu)
        if self.Q.dtype != np.int16:
            raise TypeError('Q dtype only np.int16 supported yet')
        self.timer.pick_time('build_Q')
        self.split_matrix_to_blocks(self.A)

        cp = self.gimme_lib()
        self.BB = QBF.build_energy_vector(self.Q_1B, 10, self.B - 10, cp)
        self.timer.pick_time('BB')

        indices = cp.arange(2**self.A, dtype=cp.uint32)
        gray_indices = indices ^ (indices >> 1)
        tmp = QBF.build_energy_vector(self.Q_0A, 10, self.A - 10, cp)
        self.AA = tmp[gray_indices]
        self.timer.pick_time('AA')
        del tmp, indices, gray_indices

        X_B = cp.zeros((2 ** self.B, self.B), dtype=np.int16)
        QBF.fill_all_state_matrix(X_B)
        self.timer.pick_time('X_B')
        self.ABBA = (self.Q_0B + self.Q_1A.T) @ X_B.T
        self.timer.pick_time('ABBA')

        h_best_E = np.zeros(threads_cnt, dtype=self.energy_type)
        h_best_x_A = np.zeros(threads_cnt, dtype=self.state_type)
        h_best_x_B = np.zeros(threads_cnt, dtype=self.state_type)
        self.best_E_vec = cp.asarray(h_best_E)
        self.best_x_A_vec = cp.asarray(h_best_x_A)
        self.best_x_B_vec = cp.asarray(h_best_x_B)

        for arr in [self.best_x_A_vec, self.best_x_B_vec]:
            assert arr.dtype == self.state_type
        for arr in [self.ABBA, self.BB, self.AA, self.best_E_vec]:
            assert arr.dtype == self.energy_type
        self.timer.pick_time('Prepare')
        del X_B

    # def post_process_result(self):
    #     idx = np.argmin(self.best_E_vec)
    #     self.glob_E = int(self.best_E_vec[idx])
    #     pref = self.best_x_A_vec[idx]
    #     bin_str = format(pref, '032b')
    #     # cp = self.gimme_lib()
    #     x_A = np.array(list(bin_str[-self.A:]), dtype=np.uint8)
    #     # e_a = x_A @ self.Q_0A @ x_A.T

    #     tmp = self.BB.copy()
    #     for i in range(self.A):
    #         if x_A[i] == 1:
    #             tmp += self.ABBA[i]
    #     suff = np.argmin(tmp)
    #     bin_str2 = format(suff, '032b')
    #     x_B = np.array(list(bin_str2[-self.B:]), dtype=np.uint8)
    #     self.glob_pref = x_A
    #     self.glob_X = x_B
    #     self.timer.pick_time('Postprocess')

    @staticmethod
    def gimme_QB_F1_kernel_code(N=9, A=3, gridDim_x=2, blockDim_x=5, E_A_window_len=64, energy_type=np.int16, state_type=np.int32, index_type=np.int32):
        qbf1_constexprs_code = f"""
        using T_Energy = {type_dict[energy_type]};
        using T_State = {type_dict[state_type]};
        using T_Index = {type_dict[index_type]};

        constexpr T_Index N = {N};
        constexpr T_Index A = {A};
        constexpr T_Index B = N - A;

        constexpr T_State _2_power_A = 1LL << A;
        constexpr T_State _2_power_B = 1LL << B;

        constexpr T_Index gridDim_x = {gridDim_x};
        constexpr T_Index blockDim_x = {blockDim_x};
        constexpr T_Index threads_cnt = gridDim_x * blockDim_x;
        constexpr T_Index segs_cnt = threads_cnt;

        constexpr T_Index seg_len = (_2_power_B + segs_cnt - 1) / segs_cnt;
    //    constexpr T_Index E_A_window_len = {E_A_window_len};    

        """

        qbf1_main_kernel_code = """
        extern "C" __global__ void QB_F1_kernel(
                           const T_Energy cut_E,                 
                           const T_Energy* ABBA,                 // A x 2**B size matrix
                           const T_Energy* BB,                   // 2**B size vector
                           const T_Energy* Es_A,                 // 2**A size vector
                           T_Energy* best_E,                     // threads_count size vector, output for best energy founded by thread
                           T_State* best_x_A,                   // threads_count size vector, output for prefix of best energy state founded by thread
                           T_State* best_x_B)                   // threads_count size vector, output for suffix of best energy state founded by thread
        {
            T_Energy ABBA_seg[seg_len * A];
            T_Energy E_vec_seg[seg_len];
            const T_Index tid = blockIdx.x * blockDim.x + threadIdx.x;

            for (T_Index ABBA_row_idx = 0; ABBA_row_idx < A; ++ABBA_row_idx) {
                for (T_Index col = 0; col < seg_len; col++) {
                    T_Index global_col = blockIdx.x * blockDim.x * seg_len + blockDim.x * col + threadIdx.x;
                    T_Index global_idx = ABBA_row_idx * _2_power_B + global_col;
                    T_Index local_idx = ABBA_row_idx * seg_len + col;
                    ABBA_seg[local_idx] = ABBA[global_idx];
                }
            }
            for (T_Index col = 0; col < seg_len; col++) {
                T_Index global_col = blockIdx.x * blockDim.x * seg_len + blockDim.x * col + threadIdx.x;
                E_vec_seg[col] = BB[global_col];
            }

            T_State x_A_idx = 0;
            T_State x_A = 0;

            T_State prev_x_A = 0;
            T_Index flipped_bit_idx;

            T_State seg_best_x_A = 0;
            T_State seg_best_x_B = 0;
            T_Energy seg_best_E = min(min(min(E_vec_seg[0],E_vec_seg[1]), min(E_vec_seg[2],E_vec_seg[3])),min(min(E_vec_seg[4],E_vec_seg[5]), min(E_vec_seg[6],E_vec_seg[7])));
            seg_best_E = min(seg_best_E, min(min(seg_best_E, min(min(E_vec_seg[8],E_vec_seg[9]), min(E_vec_seg[10],E_vec_seg[11]))),min(min(E_vec_seg[12],E_vec_seg[13]), min(E_vec_seg[14],E_vec_seg[15]))));

            while (x_A_idx < _2_power_A-1) {
                x_A_idx++;
                x_A = x_A_idx ^ (x_A_idx >> 1);
                flipped_bit_idx = (__ffs(x_A_idx) - 1);
                T_Energy E_A = Es_A[x_A_idx];
    //            T_Energy E_A = Es_A[x_A];
                // self.A - gray_code.flipped_bit_idx - 1
                T_Index row = A - flipped_bit_idx - 1;

                if (x_A > prev_x_A) {
                    for (T_Index seg_col = 0; seg_col < seg_len; ++seg_col) {
                        E_vec_seg[seg_col] += ABBA_seg[seg_len * row + seg_col];
                    }
                }
                else {
                    for (T_Index seg_col = 0; seg_col < seg_len; ++seg_col) {
                        E_vec_seg[seg_col] -= ABBA_seg[seg_len * row + seg_col];
                    }
                }
                T_Energy min_E_seg = min(min(min(min(min(min(min(E_vec_seg[0],E_vec_seg[1]),E_vec_seg[2]),E_vec_seg[3]),E_vec_seg[4]),E_vec_seg[5]),E_vec_seg[6]),E_vec_seg[7]);
                min_E_seg = min(min_E_seg, min(min(min(min(min(min(min(E_vec_seg[8],E_vec_seg[9]),E_vec_seg[10]),E_vec_seg[11]),E_vec_seg[12]),E_vec_seg[13]),E_vec_seg[14]),E_vec_seg[15]));  

                T_Energy tested_E = E_A + min_E_seg;
                seg_best_x_A = (tested_E < seg_best_E) ? x_A : seg_best_x_A; 
                seg_best_E = min(seg_best_E, tested_E);
                prev_x_A = x_A;
            }

            best_E[tid] = seg_best_E;
            best_x_A[tid] = seg_best_x_A;
            best_x_B[tid] = seg_best_x_B;
        }
        """

        return qbf1_constexprs_code + qbf1_main_kernel_code

    def pack_results_to_solution_instance(self, problem):
        idx = np.argmin(self.best_E_vec)
        self.glob_E = int(self.best_E_vec[idx])
        result_prefix = QBF.int_to_numpy_binary_array(self.best_x_A_vec[idx], self.A)

        tmp = self.BB.copy()
        for i in range(self.A):
            if result_prefix[i] == 1:
                tmp += self.ABBA[i]

        idx2 = np.argmin(tmp)
        result_suffix = QBF.int_to_numpy_binary_array(idx2, self.B)
        result_state_vector = np.hstack((result_prefix, result_suffix))
        self.timer.pick_time('Finish')

        wide_matrix = np.zeros(shape = (1, self.N), dtype = result_state_vector.dtype)
        wide_matrix[0] = result_state_vector

        epochs_cnt = 1
        solution = Solution(self.N, epochs_cnt, 1, problem.to_json(), self.json_str)
        solution.state_vectors = StateVectors(wide_vector_matrix=wide_matrix)
        solution.energies = np.array([self.glob_E])
        metric_dict = {}
        # metric_dict['t_arr'] = self.parse_epoch_times()
        # metric_dict['energy_arr'] = np.array([self.glob_E] * epochs_cnt)
        # metric_dict['energy_arr'] = np.array(self.best_E_by_epochs)
        solution.metrics = [metric_dict]
        solution.solver_class = 'QB_F1'
        solution.best_E = self.glob_E
        solution.best_X = result_state_vector
        if hasattr(problem, 'prefix_E'):
            solution.prefix_E = problem.prefix_E

        solution.ising_offset = problem.get_Q_Ising_offset(self.gimme_lib(), self.Q)  
        solution.timer = [(t, l) for t, l in self.timer.times if l != 'epoch']
        solution.time_to_solve_ms = self.timer.get_time_from_start()
        return solution


    def split_matrix_to_blocks(self, A):
        self.Q_0A = self.Q[:A, :A]
        self.Q_0B = self.Q[:A, A:]
        self.Q_1A = self.Q[A:, :A]
        self.Q_1B = self.Q[A:, A:]

    def check_obsolete_args(self, **kwargs):
        if 'use_gpu' in kwargs:
            raise ValueError('use_gpu param is obsolete!')

        if 'gpu_id' in kwargs:
            raise ValueError('gpu_id param is obsolete!')
        
        if 'devices' in kwargs:
            raise ValueError('Use device argument')

    def parse_device_arg(self, **kwargs):
        device = kwargs.get('device', 'cuda')
        if device[:4] == 'cuda':
            self.use_gpu = True
            if len(device) > 5:
                gpu_id = int(device[5:])
            else:
                gpu_id = 0
            self.init_gpu(gpu_id)
        else:
            raise ValueError(f"QB_F1 solver is GPU only: {device}")
        self.device_name = device