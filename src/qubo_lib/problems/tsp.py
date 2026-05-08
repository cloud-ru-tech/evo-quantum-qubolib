from typing import Union
from ..abc import ProblemABC
from ..state_vectors import StateVectors
import numpy as np
from numpy.random import Generator, PCG64
import math
import jsonpickle

import os
from contextlib import redirect_stdout
with open(os.devnull, 'w') as f, redirect_stdout(f):
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp


class TSP(ProblemABC):
    """
    Traveling Salesman Problem (TSP) formulated as QUBO.
    This class implements the TSP problem: find shortest possible route that visits each city exactly once and returns
    to the origin city. The class supports both random problem generation and explicit city coordinate specification.
    Google OR-Tools is utilized to find the shortest path and identify the ground state.
    """
    @staticmethod
    def init_from_cities(cities, dtype: np.dtype = np.float32) -> 'TSP':
        """
        Initialize TSP problem instance from cities coordinates.
        
        Args:
            cities: np.array of shape (n_cities, 2), where cities[:, 0] is X-coordinates, and cities[:, 1] is Y-coordinates.
            dtype: NumPy data type for matrix elements (default: np.float32).

        Returns: Initialized TSP instance.
        """            
        n_cities = cities.shape[0]
        N = n_cities ** 2
        obj = TSP(N, dtype=dtype)
        obj.n_cities = n_cities
        obj.cities = cities.astype(dtype)
        # xs, ys = cities[:, 0], cities[:, 1]
        return obj

    @staticmethod
    def init_random_instance(seed: int, n_cities: int, low = 0, high = 100, dtype: np.dtype = np.float32) -> 'TSP':
        """
        Initialize TSP problem instance from uniformly distributed random cities coordinates.
        
        Args:
            seed: Random seed for reproducibility.
            n_cities: Amount of cities.
            low: Lower bound for uniform distribution of cities coordinates (default: 0).
            high: Upper bound for uniform distribution of cities coordinates (default: 100).
            dtype: NumPy data type for matrix elements (default: np.float32).
        Returns: Initialized TSP instance.
        """            
        rng = Generator(PCG64(seed))
        cities = rng.uniform(low=low, high=high, size=(n_cities, 2)).astype(dtype)
        obj = TSP.init_from_cities(cities, dtype)
        obj.seed = seed
        return obj

    def __init__(self, N: int, dtype: np.dtype = np.float32):
        self.cities = None
        self.best_path = None
        super().__init__(N, dtype)


    def build_Q_matrix(self, use_gpu: bool = False):
        if use_gpu:
            import cupy as cp
            lib = cp
        else:
            lib = np

        xs, ys = self.cities[:, 0], self.cities[:, 1]
        return self.qubo_from_positions(xs, ys, lib=lib).astype(self.dtype)

    def to_json(self) -> str:
        return jsonpickle.encode(self, indent=4)

    @staticmethod
    def init_from_json(json_str: str) -> 'ProblemABC':
        return jsonpickle.decode(json_str)


    def calculate_utility_metric(self, states: StateVectors) -> np.array:
        """
        For TSP utility metric is route length.
        Calculate total route length for each provided state vector.

        Args:
            states: An object containing multiple candidate solutions encoded as bit vectors.

        Returns:
            np.ndarray: Array of route lengths corresponding to each input state.
        """
        xs = states.wide_vectors
        cnt = xs.shape[0]
        utilities_vec = np.zeros(cnt)
        for i in range(cnt):
            utilities_vec[i] = self.calculate_state_utility_metric(xs[i])
        return utilities_vec

    def calculate_state_utility_metric(self, x):
        """
        For TSP utility metric is route length
        Compute the total travel distance for a single decoded path represented by a state vector.

        Args:
            x: Binary array representing one candidate solution.

        Returns:
            float: Length of the tour defined by the path.
        """
        path = TSP.decode_x_to_path(x)
        xs, ys = self.cities[:, 0], self.cities[:, 1]
        path_length = TSP.get_path_length(path, xs, ys)
        return path_length

    @staticmethod
    def create_data_model(cities):
        """Stores the data for the problem."""
        data = {}
        # Locations in block units

        data["locations"] = [(c[0], c[1]) for c in cities]
        data["num_vehicles"] = 1
        data["depot"] = 0
        return data

    @staticmethod
    def compute_euclidean_distance_matrix(locations):
        """Creates callback to return distance between points."""
        distances = {}
        for from_counter, from_node in enumerate(locations):
            distances[from_counter] = {}
            for to_counter, to_node in enumerate(locations):
                if from_counter == to_counter:
                    distances[from_counter][to_counter] = 0
                else:
                    # Euclidean distance
                    distances[from_counter][to_counter] = int(
                        math.hypot((from_node[0] - to_node[0]), (from_node[1] - to_node[1]))
                    )
        return distances

    @staticmethod
    def make_path_from_solution(manager, routing, solution):
        index = routing.Start(0)
        path = []
        route_distance = 0
        while not routing.IsEnd(index):
            path.append(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
        return path

    @staticmethod
    def make_ORTools_path(cities):
        """Entry point of the program."""
        # Instantiate the data problem.
        data = TSP.create_data_model(cities)

        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(
            len(data["locations"]), data["num_vehicles"], data["depot"]
        )

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        distance_matrix = TSP.compute_euclidean_distance_matrix(data["locations"])

        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )

        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)
        if not solution:
            return

        path = TSP.make_path_from_solution(manager, routing, solution)
        return path

    @staticmethod
    def get_path_length(path, xs, ys):
        path_length = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            distance = ((xs[u] - xs[v]) ** 2 + (ys[u] - ys[v]) ** 2) ** 0.5
            path_length += distance
        return path_length

    @staticmethod
    def cut_longest_edge_from_path(path, xs, ys):
        n = xs.shape[0]
        cycle = path[:]
        if len(path) == n:
            cycle = path + [path[0]]
        assert len(cycle) == n + 1
        assert cycle[0] == cycle[-1]

        lengths = np.array(
            [((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2) ** 0.5 for i, j in zip(cycle[:-1], cycle[1:])])
        longest = np.argmax(lengths)
        twice_cycle = cycle[:-1] + cycle
        result = twice_cycle[longest + 1: longest + n + 1]
        return result

    @staticmethod
    def encode_path_to_x(path):
        n_cities = len(path)
        x = np.zeros(n_cities * n_cities)
        for i, c in enumerate(path):
            # time = i
            # city_num = c
            # x_idx = city_num * n_cities + time
            x[c * n_cities + i] = 1

        return x

    @staticmethod
    def decode_x_to_path(wide_x):
        N = wide_x.shape[0]
        n = int(N**0.5)
        assert n**2 == N
        assert wide_x.sum() == n
        path = np.zeros(n, dtype=np.int32)
        for x_idx in range(N):
            if wide_x[x_idx] != 1:
                continue
            # x_idx = city_num * n_cities + time
            # =>
            # time = x_idx % n_cities
            # city_num = x_idx // n_cities
            city_num, time = x_idx // n, x_idx % n
            path[time] = city_num
        return path

    def calculate_ground(self, use_gpu: bool = False):
        """
        Find an approximate optimal solution (ground state) for the TSP instance, using Google OR-Tools.
        The resulting path is stored internally along with its energy value and utility metric.        
        """
        xs, ys = self.cities[:, 0], self.cities[:, 1]
        raw_path = TSP.make_ORTools_path(self.cities)
        best_path = TSP.cut_longest_edge_from_path(raw_path, xs, ys)
        best_path_length = TSP.get_path_length(best_path, xs, ys)
        best_x = TSP.encode_path_to_x(best_path)
        Q = self.build_Q_matrix(use_gpu = use_gpu)
        ground_E = best_x @ Q @ best_x.T
        self.ground_E = ground_E
        self.ground_state = best_x
        self.ground_utility = best_path_length
        self.best_path = best_path

    @staticmethod
    def qubo_from_positions(x_pos, y_pos, lib = np):
        n = x_pos.shape[0]
        N = n ** 2
        distances = lib.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                distances[i][j] = np.sqrt((x_pos[i] - x_pos[j]) ** 2 + (y_pos[i] - y_pos[j]) ** 2)
                distances[j][i] = distances[i][j]
        Q = TSP.qubo_from_distances(distances, lib=lib)
        return Q


    @staticmethod
    def qubo_from_distances(w, lib = np):
        """
        Make qubo matrix from distaces of points
        """

        # make costs function
        n = w.shape[0]
        # super index
        super = lambda i, j: i * n + j
        Q = lib.zeros((n ** 2, n ** 2))

        # cost function

        for i in range(n):
            for j in range(n):
                for p in range(n - 1):
                    Q[super(i, p), super(j, p + 1)] += w[i, j]

        # constraints
        A = 2 * lib.max(w) + 1

        C = lib.zeros((n ** 2, n ** 2))
        # constraints
        for p in range(n):
            for i in range(n):
                for j in range(n):
                    C[super(i, p), super(j, p)] += -1 if (i == j) else 1

        for i in range(n):
            for p in range(n):
                for k in range(n):
                    C[super(i, p), super(i, k)] = -1 if (p == k) else 1

        Q += A * C

        return lib.triu(Q) + lib.tril(Q, k=-1).T
