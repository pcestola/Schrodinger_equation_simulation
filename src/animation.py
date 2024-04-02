import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

def create_animation(x: np.ndarray, solution: np.ndarray, potential: np.ndarray, name:str = 'schrodinger.gif'):
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

    # Plot potential
    fig, ax1 = plt.subplots(1,1,figsize=(8,4))
    color = 'tab:red'
    ln1, = ax1.plot([], [], '-r', lw=2)
    ax1.tick_params(axis='y')
    ax1.set_ylim(y_min-1, y_max+1)
    ax1.set_xlim(x_min-dx, x_max+dx)
    ax1.set_ylabel('$|\psi(x)|^2$', color=color, fontsize=20)
    ax1.set_xlabel('$x$', fontsize=20)

    # Plot solution
    ax2 = ax1.twinx()
    ax2.set_ylabel('$V(x)$', fontsize=20)
    ax2.plot(x, potential, '--k')
    ax2.tick_params(axis='y')
    plt.tight_layout()

    # Save the animation as a GIF
    path = os.path.dirname(__file__)
    path = os.path.join(path, '..', 'output_gifs', name)
    # TODO: fix the //10
    ani = FuncAnimation(fig, update, init_func=initalize, frames=min(solution.shape[0]//10,1000), interval=50)
    ani.save(path, writer='pillow', fps=30)
    #ani.save(path, writer='pillow', fps=50, dpi=100)