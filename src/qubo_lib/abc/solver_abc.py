from abc import ABC, abstractmethod
import numpy as np

import jsonpickle
from ..utils import PerformanceTimer
from .problem_abc import ProblemABC
from ..solution import Solution

import logging
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[logging.StreamHandler()]
# )
logger = logging.getLogger(__name__)


class SolverABC(ABC):
    """
    Abstract base class for QUBO solvers.
    
    Defines the interface that all QUBO solver classes must implement.
    """

    def __init__(self):
        # Do not ever use args and kwargs in solver __init__, cause it leads to change proto grpc
        self.use_gpu = None
        self.kwargs = None
        self.json_str = None
        self.timer = None
        self.N = None
        self.Q = None
        self.epochs_cnt = None
        self.samples_cnt = None


    def solve(self, problem:ProblemABC, **kwargs) -> Solution:
        """
        Solve the given QUBO problem.
        
        Args:
            problem: Problem instance to solve
            **kwargs: Solver-specific parameters and device specification
            
        Returns:
            Solution object with results
        """
        # Do not ever use any args in solve(), only kwargs please
        self.kwargs = kwargs
        self.json_str = jsonpickle.encode(kwargs, indent=4)
        self.N = problem.N
        self.epochs_cnt = kwargs.get('epochs_cnt', 0)
        self.samples_cnt = kwargs.get('samples_cnt', 0)
        sol = Solution(self.N, self.epochs_cnt, self.samples_cnt, problem.to_json(), self.json_str)

        self.timer = PerformanceTimer(self.use_gpu)
        self.Q = problem.build_Q_matrix(use_gpu = self.use_gpu)
        self.timer.pick_time('build_Q')

        return sol
    
    def parse_device_arg(self, **kwargs):
        device = kwargs.get('device', 'cpu')
        if device == 'cpu':
            self.use_gpu = False
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

    def init_gpu(self, gpu_id):
        try:
            lib = self.gimme_lib()
            self.cuda_device = lib.cuda.Device(gpu_id)
            self.cuda_device.use()
            logger.debug(f'QBF.init_gpu use GPU gpu_id={gpu_id}, self.device.id: {self.cuda_device.id} 98623592867')
        except Exception as e:
            logger.error(f'QBF cannot init GPU gpu_id={gpu_id}', exc_info=True)


    def gimme_lib(self):
        if self.use_gpu:
            import cupy as cp
            return cp
        else:
            return np
