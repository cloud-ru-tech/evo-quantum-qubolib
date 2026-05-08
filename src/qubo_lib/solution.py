import jsonpickle

class Solution:

    """
    Container for QUBO problem solution results.
    
    Stores the solution data including energies, state vectors, metrics,
    and metadata about the solving process. 

    Fields:
        N (int): Problem dimension (number of spins/variables)
        epochs_cnt (int): Number of temperature epochs/sweeps performed
        samples_cnt (int): Number of independent solution samples
        problem_instance_json (str): JSON representation of original problem
        solver_params_json (str): JSON representation of solver parameters
        metrics (list): List of metric dictionaries per sample, containing:
            - 'energy_arr': np.array of energies over epochs (shape: epochs_cnt,)
            - 'time_arr': np.array of timestamps per epoch (optional)
            - 't_arr': np.array of temperatures per epoch (optional)
            - 'flip_arr': np.array of flip counts per epoch (optional)
            - other solver-specific metrics
        state_vectors (StateVectors): Final solution states for all samples
        energies (np.array): Final energy values for each sample (shape: samples_cnt,)
        solver_class (str): Name of solver class used ('MSSA', 'QBF', etc.)
        best_E (float): Best energy found across all samples
        best_X (np.array): Best solution vector corresponding to best_E
        error (str): Error message if solver failed
        time_to_solve_ms (float): Total solving time in milliseconds
        prefix_E (float): Prefix energy for reduced problems (optional)
        timer (list): Timing logs [(timestamp, label), ...] (optional)
        worker_logs (list): Worker execution logs for parallel solvers (optional)
        
    Note:
        For MSSA solver, each metrics dictionary may contain additional keys:
        - 'd_E': Energy values at each epoch
        - 'd_X': State vectors at each epoch  
        - 'd_best_E': Best energies found so far
        - 'd_best_X': Best state vectors found so far
        - 'T': Temperature values
        - 'time_ms': Kernel execution times
        - 'epoch': Epoch numbers
        - 'total_time_ms': Cumulative execution times
    """    

    def __init__(self, N: int, epochs_cnt: int, samples_cnt: int, problem_instance_json: str, solver_params_json: str):
        self.N = N
        self.epochs_cnt = epochs_cnt
        self.samples_cnt = samples_cnt
        self.problem_instance_json: str = problem_instance_json
        self.solver_params_json: str = solver_params_json
        self.metrics = []  # Список словарей. Итемка списка - словарь метрик для каждого семпла
            # Словарь метрик включает ключи 'energy_arr', 't_arr' итд. По этим ключам лежат np.array, shape == (epochs_cnt, ) то есть стартовое состояние исключено
        self.state_vectors = None # экземпляр StateVectors, хранит найденные финальные решения для всех семплов
        self.energies = None # np.array, energies.shape == (samples_cnt, ) хранит энергии для self.state_vectors
        self.ising_offset = None
        self.solver_class = None
        self.best_E = None
        self.best_X = None
        self.error = ''
        self.time_to_solve_ms = None


    def to_json(self):
        return jsonpickle.encode(self, indent=4)

    @staticmethod
    def init_from_json(json_str: str) -> 'Solution':
        return jsonpickle.decode(json_str)

    def __str__(self):
        if self.error:
            return f"Error: {self.error}"
        best_x_str = ''.join(map(str, self.best_X)) if self.best_X is not None else None
        return (
            f"Solution(\n"
            f"  N={self.N},\n"
            f"  best_E={self.best_E:.3f},\n"
            # f"  best_E_Ising={self.best_E - self.ising_offset:.3f},\n"
            f"  time_to_solve_ms={self.time_to_solve_ms:.1f},\n"
            f"  samples_cnt={self.samples_cnt},\n"
            f"  solver_class='{self.solver_class}',\n"
            f"  best_X='{best_x_str}'\n"
            f")"
        )