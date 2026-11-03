from typing import Union
from ..abc import ProblemABC
import numpy as np
from numpy.random import Generator, PCG64
import jsonpickle

class RandomQUBO(ProblemABC):
    """
    Uniformly distributed random QUBO matrix problem generator.
    
    The randomness is controlled by seed and bounds for reproducibility.
    The coefficient matrix is built on-demand directly on the target device to reduce memory overhead throughout the pipeline for large-scale problems.
    """
    def __init__(self, N: int, dtype: np.dtype = np.float32):
        super().__init__(N, dtype)
        self.low = -100
        self.high = 100

    def build_Q_matrix(self, use_gpu: bool = False):
        shape = (self.N, self.N)
        rng = Generator(PCG64(self.seed))
        if np.issubdtype(self.dtype, np.integer):
            cpu_matrix = rng.integers(low=self.low, high=self.high, size=shape, dtype=self.dtype)
        else:
            cpu_matrix = self.low + rng.random(size=shape, dtype=self.dtype) * (self.high - self.low)
        if use_gpu:
            import cupy as cp
            gpu_matrix = cp.asarray(cpu_matrix)
            assert gpu_matrix.dtype == self.dtype
            return gpu_matrix
        else:
            assert cpu_matrix.dtype == self.dtype
            return cpu_matrix

    def to_json(self) -> str:
        return jsonpickle.encode(self, indent=4)

    @staticmethod
    def init_from_json(json_str: str) -> 'ProblemABC':
        return jsonpickle.decode(json_str)

    @staticmethod
    def init_random_instance(seed: int, N: int, low = -100, high = 100, dtype: np.dtype = np.float32) -> 'RandomQUBO':
        """
        Create a random QUBO problem instance.

        Args:
            seed: Random seed for reproducibility
            N: Number of variables (matrix size N x N)
            low: Lower bound for uniform distribution (default: -100)
            high: Upper bound for uniform distribution (default: 100)
            dtype: NumPy data type for matrix elements (default: np.float32)

        Returns:
            RandomQUBO: Random QUBO problem instance
        """        
        obj = RandomQUBO(N, dtype)
        obj.seed = seed
        obj.low = low
        obj.high = high
        return obj
