import jsonpickle
from ..utils import PerformanceTimer
from ..abc import ProblemABC, SolverABC
from ..solution import Solution
from ..state_vectors import StateVectors
from collections import namedtuple
import numpy as np
import gc
import logging

# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[logging.StreamHandler()]
# )
logger = logging.getLogger(__name__)

ALL_METRICS = set(['time_arr', 'energy_arr'])

class QBF(SolverABC):
    """
    O(1) per state bruteforce solver for QUBO problems.
    
    Implements an optimized brute force approach that uses Gray code iteration, prefix-suffix decomposition and cached precomputations.
    Supports both CPU (numpy) and GPU (cupy) computation backends.
    
    Examples:
    ---------
    >>> solver = QBF()
    >>> solution = solver.solve(problem, device='cuda')
    """

    def __init__(self):
        # Do not ever use args and kwargs in solver __init__, cause it leads to change proto grpc
        super().__init__()
        self.ABBA = None
        self.diag_BB = None
        self.glob_E = 0
        self.glob_pref = 0
        self.glob_X = 0
        self.best_E_by_epochs = []
        self.device_name = None
        self.cuda_device = None
        self.use_gpu = None
        self.correction_period = None


    def solve(self, problem: ProblemABC, **kwargs) -> Solution:
        """
        Solve the given QUBO problem using brute force approach.
        
        Args:
            problem: Problem instance to solve
            **kwargs: Additional solver parameters including:
                - device: Device specification ('cpu', 'cuda', 'cuda:0', etc.)
                - correction_period: number of iterations to correct accumulated computational error
                - A, B: Matrix block size parameters for decomposition
                - use_metrics: Set of metrics to collect ('time_arr', 'energy_arr', or 'all')
                - use_mem_GB: Memory limit for GPU allocation
            
        Returns:
            Solution object containing the optimal energy, state vector, metrics, and metadata about the solving process.
        """        
        # Do not ever use any args in solve(), only kwargs please
        self.prepare(problem, **kwargs)
        self.iterate_prefix_interval(fixed_prefix_bits=())
        solution = self.pack_results_to_solution_instance(problem, **kwargs)

        self.clear_mem()
        return solution

    def prepare(self, problem: ProblemABC, **kwargs):
        self.log_gpu_memory('prepare 0')
        self.check_obsolete_args(**kwargs)
        self.kwargs = kwargs
        self.json_str = jsonpickle.encode(kwargs, indent=4)
        self.parse_device_arg(**kwargs)
        self.timer = PerformanceTimer(self.use_gpu)
        self.N = problem.N
        self.Q = problem.build_Q_matrix(use_gpu=self.use_gpu)
        self.log_gpu_memory('prepare 1')
        sizeof = self.Q.dtype.itemsize
        self.timer.pick_time('build_Q')
        kwgs = dict(**kwargs)
        if self.use_gpu:
            free_mem_GB, total_mem_GB = self.cuda_device.mem_info[0] / 2 ** 30, self.cuda_device.mem_info[1] / 2 ** 30
            logger.debug(f'QBF.prepare, free_mem_GB = {free_mem_GB:.2f} 037502753')
            kwgs.update(use_mem_GB = free_mem_GB * 0.75) # Reserve 25% of free mem for GEMM tmp buffers
        self.A, self.B = QBF.define_chunk_size(N=self.N, size_of=sizeof, **kwgs)
        self.correction_period = kwargs.get('correction_period', 1024)
        self.split_matrix_to_blocks(self.A)
        self.log_gpu_memory('prepare 2')


        # logger.debug(f'QBF.prepare(), before build_cache self.use_gpu: {self.use_gpu}, self.device.id: {device_id} 037502753')
        self.ABBA, self.diag_BB = self.build_cache()
        # logger.debug(f'QBF.prepare(), after build_cache self.use_gpu: {self.use_gpu}, self.device.id: {device_id} 9836523095860')
        self.glob_E = 0
        self.glob_pref = 0
        self.glob_X = 0
        self.best_E_by_epochs = []
        self.log_gpu_memory('prepare 3')
        logger.debug(f'QBF.prepare(), self.use_gpu: {self.use_gpu} 8902375')

    def check_obsolete_args(self, **kwargs):
        if 'use_gpu' in kwargs:
            raise ValueError('use_gpu param is obsolete!')

        if 'gpu_id' in kwargs:
            raise ValueError('gpu_id param is obsolete!')
        
        if 'devices' in kwargs:
            raise ValueError('Use device argument')

    def clear_mem(self):
        del self.ABBA, self.diag_BB
        if self.use_gpu:
            lib = self.gimme_lib()
            lib.get_default_memory_pool().free_all_blocks()

    def log_gpu_memory(self, label):
        if not self.use_gpu:
            return
        cp = self.gimme_lib()

        # Получаем информацию о памяти напрямую с устройства
        device = self.cuda_device
        mempool = cp.get_default_memory_pool()
        
        # Память в пуле CuPy
        used_bytes = mempool.used_bytes()
        total_bytes = mempool.total_bytes()
        
        # Физическая память GPU
        free_bytes_device, total_bytes_device = device.mem_info
        used_bytes_device = total_bytes_device - free_bytes_device

        
        logger.debug(
            f"{label} -- GPU memory: "
            f"pool_used={used_bytes / 1024**2:.1f}MB, "
            f"pool_total={total_bytes / 1024**2:.1f}MB, "
            f"device_used={used_bytes_device / 1024**2:.1f}MB, "
            f"device_total={total_bytes_device / 1024**2:.1f}MB, "
            f"device_free={free_bytes_device / 1024**2:.1f}MB"            
        )
        self.log_large_gpu_arrays()

    def log_large_gpu_arrays(self, threshold_mb=100):
        if not self.use_gpu:
            return
        cp = self.gimme_lib()
        threshold_bytes = threshold_mb * 1024 * 1024
        
        try:
            # Получаем все объекты в памяти
            for obj in gc.get_objects():
                # Проверяем, является ли объект CuPy массивом
                if isinstance(obj, cp.ndarray):
                    array_size = obj.nbytes
                    if array_size >= threshold_bytes:
                        shape = str(obj.shape)
                        size_mb = array_size / (1024 * 1024)
                        dtype = str(obj.dtype)
                        
                        # Пытаемся найти имя переменной (опционально)
                        var_name = "unknown"
                        try:
                            # Это не всегда работает, но можно попробовать
                            for frame in reversed(gc.get_referrers(obj)):
                                if hasattr(frame, 'f_locals'):
                                    for name, value in frame.f_locals.items():
                                        if value is obj:
                                            var_name = name
                                            break
                                    if var_name != "unknown":
                                        break
                        except:
                            pass
                        
                        logger.debug(
                            f"Large GPU array [{var_name}]: "
                            f"shape={shape}, size={size_mb:.1f}MB, dtype={dtype}"
                        )
        except Exception as e:
            logger.debug(f"Failed to scan GPU arrays: {e}")        

    def iterate_prefix_interval(self, fixed_prefix_bits):
        logger.debug(f'QBF.iterate_prefix_interval, fixed_prefix_bits: {fixed_prefix_bits}, use_gpu: {self.use_gpu}')
        chunk_E_vec = self.diag_BB.copy()

        for i, bit in enumerate(fixed_prefix_bits):
            if bit == 1:
                chunk_E_vec += self.ABBA[i]

        # bit_array[:fixed_count] = fixed_prefix_bits
        # bit_array[actual_flipped_bit_idx] += sign
        # yield QBF.GrayCodeTuple(flipped_bit_idx, sign, current_gray, bit_array.copy())

        # best_X_suffix = x = chunk_E_vec.argmin()
        # best_chunk_E = chunk_E_vec[best_X_suffix]
        # self.glob_E, self.glob_pref, self.glob_X = best_chunk_E + prefix_E, gray_code.value, best_X_suffix

        # result_prefix = QBF.int_to_numpy_binary_array(num=self.glob_pref, bits_count=self.A)
        # result_suffix = QBF.int_to_numpy_binary_array(num=self.glob_X, bits_count=self.B)
        # result_state_vector = np.hstack((result_prefix, result_suffix))
        # solution.best_X = result_state_vector
        # solution.best_X == array([0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0], dtype=int32)
        # solution.best_X @ Q @ solution.best_X.T == np.float64(-4044.633762359619)


        for i, gray_code in enumerate(self.gray_code_iterator(bits_count = self.A, fixed_prefix_bits=fixed_prefix_bits, timer=self.timer, label=f'epoch')):
            ABBA_row_idx = self.A - gray_code.flipped_bit_idx - 1
            if gray_code.flip_sign == 1:
                chunk_E_vec += self.ABBA[ABBA_row_idx]
            elif gray_code.flip_sign == -1:
                chunk_E_vec -= self.ABBA[ABBA_row_idx]

            if (i+1) % self.correction_period == 0:
                self.log_gpu_memory(f'Check Mem before corection {i}')
                chunk_E_vec_before = chunk_E_vec.copy()
                logger.debug(f'QBF main loop, chunk_E_vec before correction: {chunk_E_vec_before}')
                chunk_E_vec = self.do_correction(gray_code)
                logger.debug(f'QBF main loop, chunk_E_vec after correction: {chunk_E_vec}')
                corr_norm = self.gimme_lib().linalg.norm(chunk_E_vec - chunk_E_vec_before) 
                logger.debug(f'Correction norm: {corr_norm}')
                self.log_gpu_memory(f'Check Mem after corection {i}')

            x = chunk_E_vec.argmin()
            self.update_global_minima(x, chunk_E_vec, gray_code)

    def do_correction(self, gray_code):
        # chunk_E_vec = self.diag_BB.copy()
        # logger.debug(f'QBF.do_correction, diag_BB: {self.diag_BB.copy()}')
        # logger.debug(f'QBF.do_correction, gray_code.bit_array: {gray_code.bit_array}')
        # for i, bit in enumerate(gray_code.bit_array):
        #     logger.debug(f'QBF.do_correction, bit_array[{i}]: bit, ABBA[{i}]: {self.ABBA[i]}')
        #     if bit == 1:
        #         chunk_E_vec += self.ABBA[i]
        x = self.gimme_lib().array(gray_code.bit_array)
        chunk_E_vec = self.diag_BB.copy() + x @ self.ABBA
        # logger.debug(f'QBF.do_correction, chunk_E_vec: {chunk_E_vec}')
        # logger.debug(f'QBF.do_correction, test_corr_E_vec: {test_corr_E_vec}')
        return chunk_E_vec

    def build_cache(self):
        self.log_gpu_memory('build_cache 0')
        tmp_A = 0
        if self.B > 10:
            tmp_A = 5
        diag_BB = QBF.build_energy_vector(self.Q_1B, tmp_A, self.B - tmp_A, self.gimme_lib())
        self.log_gpu_memory('build_cache 1')
        logger.debug(f'QBF.build_cache(), diag_BB shape: {diag_BB.shape}, dtype: {diag_BB.dtype}, mem:{diag_BB.nbytes//2**20} Mb,       903725023975')
        self.timer.pick_time('diag_BB')

        X_B = self.generate_X_B()
        logger.debug(f'QBF.build_cache(), X_B shape: {X_B.shape}, dtype: {X_B.dtype}, mem:{X_B.nbytes//2**20} Mb,       2368296720')

        assert self.Q_0B.dtype == self.Q_1A.dtype == X_B.dtype, f"Inconsistent dtypes: {self.Q_0B.dtype}, {self.Q_1A.dtype}, {X_B.dtype}"
        logger.debug(f"ABBA will be: shape=({self.Q_0B.shape[0]}, {X_B.shape[0]}), dtype={self.Q_0B.dtype}, memory={(self.Q_0B.shape[0] * X_B.shape[0] * self.Q_0B.dtype.itemsize / 1024**2):.1f}MB")

        self.log_gpu_memory('build_cache 2')
        self.timer.pick_time('X_B')
        Q_1A = self.Q_0B + self.Q_1A.T
        ABBA = self.compute_ABBA_chunked(Q_1A, X_B)
        # ABBA = Q_1A @ X_B.T
        self.log_gpu_memory('build_cache 3')
        logger.debug(f'QBF.build_cache(), ABBA shape: {ABBA.shape}, dtype: {ABBA.dtype}, mem:{ABBA.nbytes//2**20} Mb,       392865293856')
        self.timer.pick_time('ABBA')

        X_B = None
        self.log_gpu_memory('build_cache 4')
        return ABBA, diag_BB
    

    def compute_ABBA_chunked(self, Q_1A, X_B):
        A, B = self.A, self.B
        cnt = 16
        lil_size = (2**B) // cnt
        
        result = self.gimme_lib().empty((A, 2**B), dtype=Q_1A.dtype)
        
        for i in range(cnt):
            start = i * lil_size
            end = (i + 1) * lil_size
            X_chunk = X_B[start:end, :]
            result[:, start:end] = Q_1A @ X_chunk.T
            
        return result

    GrayCodeTuple = namedtuple('GrayCodeTuple', ['flipped_bit_idx', 'flip_sign', 'value', 'bit_array'])
    @staticmethod
    def gray_code_iterator(bits_count, fixed_prefix_bits=(), timer=None, label=''):
        fixed_count = len(fixed_prefix_bits)
        lower_bits_count = bits_count - fixed_count

        if fixed_count == 0:
            fixed_bits_val = 0
        else:
            fixed_bits_val = 0
            for b in fixed_prefix_bits:
                fixed_bits_val = (fixed_bits_val << 1) | int(b)

        flip_positions_dict = {2 ** i: i for i in range(lower_bits_count)}
        bit_array = np.zeros(bits_count)
        bit_array[:fixed_count] = fixed_prefix_bits
        # prev_gray = fixed_bits_val * 2**lower_bits_count
        prev_gray = 0

        current_gray = (fixed_bits_val << lower_bits_count) | 0

        # logger.debug(f'gray_code_iterator {current_gray}')
        yield QBF.GrayCodeTuple(0, 0, current_gray, bit_array.copy())
        for i in range(1, 1 << lower_bits_count):
            gray = i ^ (i >> 1)
            sign = 1 if gray > prev_gray else -1
            flipped_bit_idx = flip_positions_dict[gray ^ prev_gray]
            actual_flipped_bit_idx = bits_count - (flipped_bit_idx + 1)
            bit_array[actual_flipped_bit_idx] += sign
            prev_gray = gray
            current_gray = (fixed_bits_val << lower_bits_count) | gray
            if timer is not None:
                timer.pick_time(label)
            # logger.debug(f'gray_code_iterator {current_gray}')
            yield QBF.GrayCodeTuple(flipped_bit_idx, sign, current_gray, bit_array.copy())


    def pack_results_to_solution_instance(self, problem : ProblemABC, **kwargs):
        self.timer.pick_time('epoch')
        result_prefix = QBF.int_to_numpy_binary_array(num=self.glob_pref, bits_count=self.A)
        result_suffix = QBF.int_to_numpy_binary_array(num=self.glob_X, bits_count=self.B)
        result_state_vector = np.hstack((result_prefix, result_suffix))

        wide_matrix = np.zeros(shape = (1, self.N), dtype = result_state_vector.dtype)
        wide_matrix[0] = result_state_vector

        epochs_cnt = 2**self.A
        solution = Solution(self.N, epochs_cnt, 1, problem.to_json(), self.json_str)
        solution.state_vectors = StateVectors(wide_vector_matrix=wide_matrix)
        solution.energies = np.array([self.glob_E])

        use_metrics = kwargs.get('use_metrics', set())
        if use_metrics == 'all':
            use_metrics = ALL_METRICS
        metric_dict = {}
        if 'time_arr' in use_metrics:
            metric_dict['time_arr'] = self.parse_epoch_times()
        # metric_dict['energy_arr'] = np.array([self.glob_E] * epochs_cnt)
        if 'energy_arr' in use_metrics:
            metric_dict['energy_arr'] = np.array(self.best_E_by_epochs)
        solution.metrics = [metric_dict]
        solution.solver_class = 'QBF'
        solution.best_E = self.glob_E
        solution.best_X = result_state_vector
        solution.ising_offset = problem.get_Q_Ising_offset(self.gimme_lib(), self.Q)
        if hasattr(problem, 'prefix_E'):
            solution.prefix_E = problem.prefix_E
        solution.time_to_solve_ms = self.timer.get_time_from_start()
        solution.timer = [(t, l) for t, l in self.timer.times if l != 'epoch']
        return solution

    def parse_epoch_times(self):
        time_arr = np.array([t for t, l in self.timer.times if l == 'epoch'])
        # assert time_arr.shape[0] == 2**self.A
        return time_arr

    def generate_X_B(self):
        B = self.B
        result = self.gimme_lib().zeros((2 ** B, B), dtype=self.Q.dtype)
        self.fill_all_state_matrix(result)
        return result

    @staticmethod
    def fill_all_state_matrix(zeros_matrix):
        mx = zeros_matrix
        n = mx.shape[1]
        assert mx.shape == (2**n, n)
        # assert (mx == 0).all()
        mx[1, -1] = 1
        for i in range(1, n):
            col = n - i
            row = 2 ** i
            mx[row:2 * row, col:] = mx[:row, col:]
            mx[row:2 * row, col - 1] = 1


    @staticmethod
    def int_to_numpy_binary_array(num, bits_count, dtype=np.int32):
        if bits_count == 0:
            return np.array([], dtype=dtype)
        binary_string = f'{num:b}'.zfill(bits_count)
        binary_array = np.array([int(bit) for bit in binary_string], dtype=dtype)
        return binary_array

    def update_global_minima(self, best_X_suffix, energy_vector, gray_code):
        # self.timer.pick_time('host_epoch')
        if gray_code.value == 0:
            self.glob_E = 0
            self.glob_pref = 0
            self.glob_X = 0
            self.best_E_by_epochs = []

        if self.use_gpu:
            best_X_suffix = best_X_suffix.item()
            best_chunk_E = energy_vector[best_X_suffix].item()
        else:
            best_chunk_E = energy_vector[best_X_suffix]

        prefix_E = gray_code.bit_array @ self.np_Q_0A @ gray_code.bit_array.T
        self.best_E_by_epochs += [best_chunk_E + prefix_E]
        if best_chunk_E + prefix_E < self.glob_E:
            self.glob_E, self.glob_pref, self.glob_X = best_chunk_E + prefix_E, gray_code.value, best_X_suffix

        # logger.debug(f'gray_code {gray_code}, energy_vector {energy_vector}, prefix_E {prefix_E}')

    @staticmethod
    def gimme_arr_as_np(arr):
        if hasattr(arr, '__cuda_array_interface__'):
            return arr.get()
        return arr

    def split_matrix_to_blocks(self, A):
        self.Q_0A = self.Q[:A, :A]
        self.Q_0B = self.Q[:A, A:]
        self.Q_1A = self.Q[A:, :A]
        self.Q_1B = self.Q[A:, A:]
        self.np_Q_0A = QBF.gimme_arr_as_np(self.Q_0A)

    @staticmethod
    def define_chunk_size(N, **kwargs):
        logger.debug(f'QBF.define_chunk_size(), kwargs: {kwargs} 9275091751')
        if 'A' in kwargs:
            return kwargs['A'], N - kwargs['A']
        if 'B' in kwargs:
            return N - kwargs['B'], kwargs['B']
        if 'use_mem_GB' in kwargs:
            use_mem_GB = kwargs['use_mem_GB']
            sizeof = kwargs.get('size_of', 4)
            rez = QBF.adjust_AB_to_fit_memory(N, use_mem_GB, sizeof)
            logger.debug(f'QBF.define_chunk_size(), kwargs[use_mem_GB]: {use_mem_GB}, A, B:{rez}, sizeof {sizeof} 0897304917')
            return rez

        B = min(N, 20)
        rez = N - B, B
        logger.debug(f'QBF.define_chunk_size(), default branch, A, B: {rez} 2365293659')
        return rez

    @staticmethod
    def estimate_memory_consumption_bytes(A, B, sizeof):
        diag_BB_size = B * (2 ** B)
        ABBA_size = A * 2 ** B
        return (diag_BB_size + ABBA_size) * sizeof

    @staticmethod
    def adjust_AB_to_fit_memory(N, use_mem_GB, sizeof = 4):
        mem_size_bytes = use_mem_GB * 2**30
        bestB = N - 2
        for B in range(N - 2, 0, -1):
            A = N - B
            bestB = B
            mem_reqire = QBF.estimate_memory_consumption_bytes(A, B, sizeof)
            if mem_reqire <= mem_size_bytes:
                logger.debug(f'QBF.adjust_AB_to_fit_memory(), A, B, mem_reqire, mem_size_bytes: {A}, {B}, {mem_reqire}, {mem_size_bytes}         43098634')
                break
        logger.debug(f'QBF.adjust_AB_to_fit_memory(), N - bestB, bestB, mem_reqire, mem_size_bytes: {N - bestB}, {bestB}, {mem_reqire}, {mem_size_bytes}         892356856')
        return N - bestB, bestB

    def dummy_solve(self, problem: ProblemABC, **kwargs) -> Solution:
        self.kwargs = kwargs
        self.N = problem.N
        self.Q = problem.build_Q_matrix(use_gpu = self.use_gpu)
        lib = self.gimme_lib()
        X = lib.zeros((2 ** self.N, self.N), dtype=self.Q.dtype)
        self.fill_all_state_matrix(X)
        E = StateVectors.evaluate_states_energies_vector(lib, X, self.Q)
        if self.use_gpu:
            x_idx = E.argmin().item()
            best_e = E[x_idx].item()
            best_x = X[x_idx].item()
        else:
            x_idx = E.argmin()
            best_e = E[x_idx]
            best_x = X[x_idx]

        solver_json = jsonpickle.encode(kwargs, indent=4)
        solution = Solution(self.N, 1, 1, problem.to_json(), solver_json)
        solution.state_vectors = StateVectors(wide_vector_matrix=best_x.reshape(1, self.N))
        solution.energies = np.array([best_e])
        return solution

    def check_all_values(self, prefix, E_vec):
        x_prefix = QBF.int_to_numpy_binary_array(num=prefix, bits_count=self.A)
        prefix_E = x_prefix @ self.np_Q_0A @ x_prefix
        actual_E_vec = np.zeros(2**self.B)
        expected_E_vec = np.zeros(2**self.B)
        for i, E in enumerate(E_vec):
            x_suffix = QBF.int_to_numpy_binary_array(num=i, bits_count=self.B)
            x = np.hstack((x_prefix, x_suffix))
            expected_E = x @ self.Q @ x.T
            expected_E_vec[i] = expected_E
            actual_E = prefix_E + E_vec[i]
            actual_E_vec[i] = actual_E
            if not np.allclose(expected_E_vec, actual_E_vec, atol=1e-4):
                logger.error(f'Error in prefix={prefix} i={i} x={x},expected_E={expected_E} actual_E={actual_E}, prefix_E={prefix_E}, E_vec[i]={E_vec[i]}')
                raise AssertionError


    @staticmethod
    def build_energy_vector_dummy(Q, A, B, lib):
        N = Q.shape[0]
        X = lib.zeros((2 ** N, N), dtype=Q.dtype)
        QBF.fill_all_state_matrix(X)
        result = lib.einsum('ij,jk,ik->i', X, Q, X)
        return result

    @staticmethod
    def build_energy_vector(Q, A, B, lib):
        N = Q.shape[0]        
        result = lib.zeros(2**N, dtype=Q.dtype)
        for E_chunk, gray_code in QBF.yield_energy_vector_chunks(Q, A, B, lib):
            result[gray_code.value * 2**B: (gray_code.value + 1) * 2**B] = E_chunk 
        return result
    
    @staticmethod
    def yield_energy_vector_chunks(Q, A, B, lib):
        Q_0A = QBF.gimme_arr_as_np(Q[:A, :A])
        Q_0B = Q[:A, A:]
        Q_1A = Q[A:, :A]
        Q_1B = Q[A:, A:]

        X_B = lib.zeros((2 ** B, B), dtype=Q.dtype)
        QBF.fill_all_state_matrix(X_B)
        ABBA = (Q_0B + Q_1A.T) @ X_B.T
        if B <= 10:
            chunk_E_vec = lib.einsum('ij,jk,ik->i', X_B, Q_1B, X_B)
        else:
            chunk_E_vec = QBF.build_energy_vector(Q_1B, B - 10, 10, lib)
        X_B = None

        for gray_code in QBF.gray_code_iterator(bits_count = A):
            ABBA_row_idx = A - gray_code.flipped_bit_idx - 1
            if gray_code.flip_sign == 1:
                chunk_E_vec += ABBA[ABBA_row_idx]
            elif gray_code.flip_sign == -1:
                chunk_E_vec -= ABBA[ABBA_row_idx]

            prefix_E = gray_code.bit_array @ Q_0A @ gray_code.bit_array.T
            yield prefix_E + chunk_E_vec, gray_code
