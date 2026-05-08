# from .problems import *
# from .solvers import *
# from .abc import *
# from .solution import Solution
# from .state_vectors import StateVectors
from .remote_solver import RemoteSolver

# библиотека экспортирует наружу:
# from qubo_lib.problems import TSP
# from qubo_lib.solvers import MSSA
# from qubo_lib import RemoteSolver
# Сервер в норме запускается скриптом run-qubo-server после инсталляции
# Всё остальное нужно-то для внутреннего использования, в т.ч. в юнит-тестах


import logging
# Добавляем NullHandler для предотвращения предупреждений
logging.getLogger(__name__).addHandler(logging.NullHandler())