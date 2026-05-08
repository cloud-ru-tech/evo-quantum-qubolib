from ..abc import ProblemABC
import numpy as np
import jsonpickle

class ReducedProblem(ProblemABC):
    """
    Reduced QUBO problem with fixed prefix variables.

    This class represents a QUBO problem reduced by fixing prefix variables to specific values.
    The base problem is stored, and reduced matrix is computed on-demand to minimize memory usage. 
    Useful for decomposing large optimization problems into smaller subproblems.

    The prefix_E field contains constant energy offset from substituted prefix variables.
    """    

    def __init__(self, problem : ProblemABC, prefix : np.array):
        """Initialize reduced QUBO problem.
        
        Args:
            problem: Original QUBO problem to reduce
            prefix: numpy array, binary values to fix prefix variables, length is prefix size
        """        
        self.base_problem = problem
        self.A = prefix.shape[0]
        self.prefix = prefix
        self.prefix_E = None
        N = problem.N - self.A
        dtype = problem.dtype
        super().__init__(N, dtype)

    def build_Q_matrix(self, use_gpu: bool = False):
        Q = self.base_problem.build_Q_matrix(use_gpu=use_gpu)
        A = self.A

        if use_gpu:
            import cupy as cp
            lib = cp
        else:
            lib = np

        x = lib.array(self.prefix, dtype=self.dtype)
        self.prefix_E = (x @ Q[:A, :A] @ x.T).item()
        dgnl = x @ (Q[:A, A:] + Q[A:, :A].T)
        result = Q[A:,A:] + lib.diag(dgnl)
        assert result.dtype == self.dtype
        return result

    def to_json(self) -> str:
        return jsonpickle.encode(self, indent=4)

    @staticmethod
    def init_from_json(json_str: str) -> 'ReducedProblem':
        return jsonpickle.decode(json_str)

    @staticmethod
    def init_random_instance(*args, **kwargs) -> 'ProblemABC':
        raise NotImplementedError("It's a user-defined matrix, for random instance better use RandomQUBO class")
