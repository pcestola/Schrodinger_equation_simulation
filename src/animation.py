import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

def create_animation(x: np.ndarray, solution: np.ndarray, potential: np.ndarray):
    '''
    Creates an animation from the computed solution of a numerical scheme.

    Inputs:
    - x (np.ndarray): An array of spatial coordinates.
    - solution (np.ndarray): The computed solution where each row represents the solution at a time step.
    - potential (np.ndarray): An array representing the potential used in the computation.

    Outputs:
    - A GIF animation is saved in the '../output_gifs' directory. Max 1000 frames.
    '''
    def initalize():
        pass

    def update(frame):
        # Updates the data for each frame of the animation
        print(frame,end='\r')
        ln1.set_data(x, solution[10*frame])
    
    # Initialize the animation settings
    plt.rcParams["animation.html"] = "jshtml"
    plt.ioff()

    # Animation parameters
    y_min = np.min(solution)
    y_max = np.max(solution)
    x_min = x[0]
    x_max = x[-1]
    dx = x[1]-x[0]

    # Plot potential with scaling
    fig, ax = plt.subplots(1,1,figsize=(8,4))
    # TODO: to fix
    ax.plot(x, potential*np.max(solution)/np.max(potential), '--k', lw=1)
    ln1, = ax.plot([], [], '-r', lw=2, markersize=8, label="Finite_Difference")
    ax.set_ylim(y_min-1, y_max+1)
    ax.set_xlim(x_min-dx, x_max+dx)
    ax.set_ylabel('$|\psi(x)|^2$', fontsize=20)
    ax.set_xlabel('$x/L$', fontsize=20)
    ax.legend(loc='upper left')
    plt.tight_layout()

    # Save the animation as a GIF
    path = os.path.dirname(__file__)
    path = os.path.join(path, '..', 'output_gifs', 'schrodinger.gif')
    # TODO: fix the //10
    ani = FuncAnimation(fig, update, init_func=initalize, frames=min(solution.shape[0]//10,1000), interval=50)
    ani.save(path, writer='pillow', fps=50, dpi=100)