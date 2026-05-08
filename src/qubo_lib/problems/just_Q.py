from ..abc import ProblemABC
import numpy as np
import jsonpickle

class Just_Q(ProblemABC):
    """
    QUBO problem defined by an explicit, user-given Q matrix.

    This class is designed for custom QUBO problems beyond the scope of qubo_lib's covered problems.
    It simply stores the provided matrix and returns it when requested.
    """

    def __init__(self, Q):
        N = Q.shape[0]
        dtype = Q.dtype
        self.Q_matrix = Q
        super().__init__(N, dtype)

    def build_Q_matrix(self, use_gpu: bool = False):
        if use_gpu:
            import cupy as cp
            gpu_matrix = cp.asarray(self.Q_matrix)
            assert gpu_matrix.dtype == self.dtype
            return gpu_matrix
        else:
            assert self.Q_matrix.dtype == self.dtype
            return self.Q_matrix

    def to_json(self) -> str:
        return jsonpickle.encode(self, indent=4)

    @staticmethod
    def init_from_json(json_str: str) -> 'ProblemABC':
        return jsonpickle.decode(json_str)

    @staticmethod
    def init_random_instance(*args, **kwargs) -> 'ProblemABC':
        raise NotImplementedError("It's a user-defined matrix, for random instance better use RandomQUBO class")
