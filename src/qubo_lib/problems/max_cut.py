from ..abc import ProblemABC
import numpy as np
import jsonpickle
import zipfile
from pathlib import Path
import csv
import io
from importlib.resources import files, as_file

import logging
logger = logging.getLogger(__name__)

class MaxCut(ProblemABC):
    """
    MaxCut problem on graphs.

    This class implements the MaxCut problem: partitioning graph vertices into two sets to maximize edges between them.
    It supports some popular benchmark MaxCut instances, including named graphs with known optimal solution
    """

# Возможные методы инициализации:
# по имени инстанса, в этом случае знаем граунд
# рандом

    def __init__(self, instance_name : str):
        """
        Initialize MaxCut problem instance.
        
        Args:
            instance_name (str): Name of the MaxCut graph instance
            
        Raises:
            ValueError: If instance_name is not in known instances
        """        
        known_instances = MaxCut.get_known_instances()
        if not instance_name in known_instances:
            raise ValueError('Unknown instance name')

        self.instance_name = instance_name
        N, best = known_instances[instance_name]
        super().__init__(N)
        self.ground_E = -best
        self.ground_utility = self.ground_E

    def build_Q_matrix(self, use_gpu: bool = False):
        ref = files("qubo_lib.problems.well_known_instances") / "MaxCut.zip"
        with as_file(ref) as zip_path:
            logger.debug(f'MaxCut zip_path: {zip_path}  239865239')
            instance_path = f'instance/{self.instance_name}/{self.instance_name}.txt'
            logger.debug(f'MaxCut instance_path: {instance_path} 209572260')

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                with zip_ref.open(instance_path) as file:
                    num_vertices, num_edges = map(int, file.readline().strip().split())
                    Q = np.zeros((num_vertices, num_vertices), dtype=self.dtype)
                    for _ in range(num_edges):
                        i, j, wij = map(int, file.readline().strip().split())
                        i -= 1
                        j -= 1
                        Q[i, i] -= wij
                        Q[j, j] -= wij
                        Q[i, j] += 2 * wij

        if use_gpu:
            import cupy as cp
            gpu_matrix = cp.asarray(Q)
            assert gpu_matrix.dtype == self.dtype
            return gpu_matrix
        else:
            assert Q.dtype == self.dtype
            return Q

    def to_json(self) -> str:
        return jsonpickle.encode(self, indent=4)

    @staticmethod
    def init_from_json(json_str: str) -> 'ProblemABC':
        return jsonpickle.decode(json_str)

    @staticmethod
    def init_random_instance(*args, **kwargs) -> 'ProblemABC':
        raise NotImplementedError("not implemented yet")

    @staticmethod
    def get_known_instances():
        zip_path = Path(__file__).parent / 'well_known_instances' / 'MaxCut.zip'

        result = {}

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # with zip_ref.open('best_solutions.csv') as csvfile:
            with zip_ref.open('best_solutions.csv') as binary_file:
                csvfile = io.TextIOWrapper(binary_file, encoding='ascii')
                reader = csv.DictReader(csvfile)
                for row in reader:
                    instance = row['instance'].strip()
                    best = int(row['best'].strip())
                    N = int(row['vertices'].strip())
                    result[instance] = (N, best)

        return result