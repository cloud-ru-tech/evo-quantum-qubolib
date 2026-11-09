from abc import ABC, abstractmethod
from typing import Optional
from ..state_vectors import StateVectors
import numpy as np

class ProblemABC(ABC):
    """
    Abstract base class for QUBO problems.
    
    Defines the interface that all QUBO problem classes must implement.
    Provides common functionality for problem definition and evaluation.
    The QUBO matrix is not explicitly stored; it is constructed lazily, on-demand, directly on the target device.
    """
    def __init__(self, N: int, dtype: np.dtype = np.float32):
        self.N = N  # обязательное поле
        self.dtype = dtype # обязательное поле

        self.seed = None   # опциональное поле
        self.ground_E = None # опциональное поле
        self.ground_state = None # опциональное поле
        self.ground_utility = None # опциональное поле

    @abstractmethod
    def build_Q_matrix(self, use_gpu: bool = False):
        """
        Build and return the QUBO matrix for this problem.
        
        Constructs the QUBO matrix representation of the problem.
        Can return GPU cupy-array if requested.
        
        Args:
            use_gpu: Whether to return GPU cupy-array
            
        Returns:
            QUBO matrix as numpy or cupy array
        """
        #TODO сказать про ленивость
        pass

    @abstractmethod
    def to_json(self) -> str:
        pass

    @staticmethod
    @abstractmethod
    def init_from_json(json_str: str) -> 'ProblemABC':
        pass

    @staticmethod
    @abstractmethod
    def init_random_instance(seed: int, N: int, dtype: np.dtype = np.float32) -> 'ProblemABC':
        """
        Create random problem instance.
        
        Args:
            seed: Random seed
            N: Problem size
            dtype: Data type for matrix elements
            
        Returns:
            Random problem instance
        """
        pass

    def calculate_utility_metric(self, states: StateVectors) -> Optional[np.ndarray]:
        """
        Вычисляет метрику полезности для данных стейт-векторов
        :return: Массив значений полезности
        """
        pass

    def calculate_ground(self):
        """
        Метод оценивает граунд-стейт и его энергию лучшим доступным для данной проблемы способом.
        Результат запоминается во внутренних полях self.ground_E и self.ground_state
        Метод необязателен к имплементации в наследниках. Не в каждой проблеме можно построить хорошую оценку граунда

        :return: None
        """
        pass

    def get_Q_Ising_offset(self, lib):
        Q = self.build_Q_matrix()
        return ProblemABC.get_Q_Ising_offset(lib, Q)

    @staticmethod
    def get_Q_Ising_offset(lib, Q):
        offset_value = 1/4 * lib.sum(Q) + 1/4 * lib.sum(lib.diag(Q))
        return offset_value