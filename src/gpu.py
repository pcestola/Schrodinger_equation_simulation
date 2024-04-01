import scipy
import cupyx
import cupy as cp
import numpy as np

from cupyx.scipy.sparse.linalg import splu

def compute_solution_gpu(solution:np.ndarray, iteration_matrix:scipy.sparse._csr.csr_matrix, time_steps:int, dx:float):
    '''
    Computes the numerical scheme's solution on the GPU using LU decomposition.

    Inputs:
    - solution (np.ndarray): array containing the solution of the numerical method.
    - iteration_matrix (scipy.sparse.csr_matrix): The CSR sparse matrix used in the numerical scheme for iteration.
    - time_steps (int): The total number of time steps to evolve the solution.
    - dx (float): The spatial step size used in the discretization.

    Outputs:
    - A np.ndarray containing the evolved solution for all time steps.
    '''

    # Convert to cp.ndarray and cupyx.scipy.sparse.spmatrix for GPU computation
    solution = cp.asarray(solution)
    iteration_matrix = cupyx.scipy.sparse.csr_matrix(iteration_matrix)

    # Compute the LU factorization of the iteration matrix
    LU_method = splu(iteration_matrix)

    # Solve the numerical scheme
    for t in range(time_steps-1):
        # check status
        if t%100 == 0:
            print(f'{t*100/time_steps:.2f}',end='\r')
        # Solve the t-th linear system
        solution[t+1] = LU_method.solve(solution[t])
        # Normalize the solution
        norm = cp.sqrt(cp.sum(cp.abs(solution[t + 1]) ** 2) * dx)
        solution[t+1] /= norm

    # Convert back to np.ndarray and return
    return solution.get()

