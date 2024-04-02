import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import diags
from .solve import compute_solution_gpu
from .solve import compute_solution_cpu

class Solver:
    '''
    Solver object

    - tot_length (float): Total length of the spatial domain.
    - num_space_steps (int): Number of spatial steps.
    - tot_time (float): Total simulation time.
    - num_time_steps (int): Number of time steps.
    '''
    def __init__(self, tot_length, num_space_steps, tot_time, num_time_steps) -> None:
        self.total_length = tot_length
        self.number_space_steps = num_space_steps
        self.total_time = tot_time
        self.number_time_steps = num_time_steps
        self.space_step = self.total_length/self.number_space_steps
        self.time_step = self.total_time/self.number_time_steps


    def initialize_iteration_matrix(self, potential: np.ndarray):
        '''
        Calculates the iteration matrix from a given potential.

        Inputs:
        - potential (np.ndarray): An array representing the potential.

        Outputs:
        - iteration_matrix (np.ndarray): An array representing the iteration matrix.
        '''
        # First matrix
        constant_1 = -1j * self.time_step / self.space_step**2
        Diagonal_1 = (1 - constant_1) * np.ones(self.number_space_steps-2, dtype=np.complex128) + 1j * self.time_step * potential
        
        # Second matrix
        constant_2 = constant_1 / 2
        Diagonal_2 = constant_2 * np.ones(self.number_space_steps-3, dtype=np.complex128)
        
        # Iteration matrix
        iteration_matrix = diags([Diagonal_1, Diagonal_2, Diagonal_2], offsets=[0, -1, 1], shape=(self.number_space_steps-2, self.number_space_steps-2), format='csc')
        return iteration_matrix
    
    def solve(self, initial_condition: np.ndarray, potential: np.ndarray, gpu=False, verbose=False):
        '''
        Given an initial condition and a potential, computes the solution of the numerical scheme.

        Inputs:
        - initial_condition (np.ndarray): An array representing the initial condition.
        - potential (np.ndarray): An array representing the potential.
        - verbose (bool): A flag to print additional information.

        Outputs:
        - solution (np.ndarray): An array representing the solution.
        '''
        # Checking the used device
        if verbose and gpu:
            gpu_id = cp.cuda.runtime.getDevice()
            print(f"Using GPU: ID {gpu_id}")
        
        # Calculate the iteration matrix
        if verbose: print("Evaluating iteration matrix")
        iteration_matrix = self.initialize_iteration_matrix(potential)

        # Initialize the solution array (ensure it's normalized)
        solution = np.zeros((self.number_time_steps, self.number_space_steps-2), dtype=cp.complex128)
        normalization = np.sum(np.absolute(initial_condition)**2) * self.space_step
        solution[0] = initial_condition / normalization

        # Compute the solution (GPU)
        if verbose: print("Solving numerical scheme")
        if gpu:
            solution = compute_solution_gpu(solution, iteration_matrix, self.number_time_steps, self.space_step)
        else:
            solution = compute_solution_cpu(solution, iteration_matrix, self.number_time_steps, self.space_step)

        # Convert the final solution into an np.ndarray
        if verbose: print("Returning solution")
        return solution