"""
Random Walk Diffusion Module

This module implements a 2D random walk diffusion simulation, where molecules
move randomly in a 2D space. It can be used to visualize how molecules diffuse
through random movement.

Author: Agosh Saini - conact@agoshsaini.com
"""

import numpy as np
import matplotlib.pyplot as plt

def random_walk(runs_per_mol=100, total_runs=10000, delta_x=0.1, delta_y=0.1):
    """
    Simulates random walk of molecules in 2D space.
    
    Args:
        runs_per_mol (int): Number of movements per molecule
        total_runs (int): Number of molecules to simulate
        delta_x (float): Smallest movement possible in x direction
        delta_y (float): Smallest movement possible in y direction
        
    Returns:
        tuple: (x_coordinates, y_coordinates) as numpy arrays
    """
    # these are the arrays use to track the coordinates of molecules in a system
    x, y = np.zeros(total_runs, dtype=float), np.zeros(total_runs, dtype=float)
    
    #This for loop calculates the changes in X plane
    for i in range(total_runs):
        for j in range(runs_per_mol):
            #integers -2,2 result in movement, others result in no movement
            direction = np.random.randint(-2, 3) 
            if direction == -2:
                x[i] += -1*delta_x
            elif direction == 2:
                x[i] += 1*delta_x 
                
    #This for loop calculates the changes in Y plane            
    for i in range(total_runs):
        for j in range(runs_per_mol):
            direction = np.random.randint(-2, 3)
            if direction == -2:
                y[i] += -1*delta_y
            elif direction == 2:
                y[i] += 1*delta_y
    
    x_reshaped, y_reshaped = x.reshape(total_runs, 1), y.reshape(total_runs, 1)
    return x_reshaped, y_reshaped

def plot_random_walk(x, y, total_runs, runs_per_mol, save=True, output_file=None, show=False):
    """
    Plots the diffusion pattern of molecules from random walk simulation.
    
    Args:
        x (numpy.array): x coordinates of molecules
        y (numpy.array): y coordinates of molecules
        total_runs (int): Number of molecules simulated
        runs_per_mol (int): Number of movements per molecule
        save (bool): Whether to save the plot to a file
        output_file (str): Path to save the plot (if None, uses default name)
        show (bool): Whether to display the plot (default: False)
    """
    plt.figure()
    plt.scatter(x, y, s=2)
    plt.title(f"{total_runs} Gas Molecules After {runs_per_mol} Movements")
    plt.xlabel('x')
    plt.ylabel('y')
    
    #limiting the axis to -5 to 5 in X and Y planes to better see diffusion
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    
    if save:
        if output_file is None:
            output_file = f"{total_runs}_molecules_after_{runs_per_mol}_movements.png"
        plt.savefig(output_file)
    
    if show:
        plt.show()
    else:
        plt.close()

def run_random_walk_simulation(runs_per_mol=100, total_runs=10000, 
                             delta_x=0.1, delta_y=0.1, save_plot=True):
    """
    Run a complete random walk diffusion simulation.
    
    Args:
        runs_per_mol (int): Number of movements per molecule
        total_runs (int): Number of molecules to simulate
        delta_x (float): Smallest movement possible in x direction
        delta_y (float): Smallest movement possible in y direction
        save_plot (bool): Whether to save the plot
        
    Returns:
        tuple: (x_coordinates, y_coordinates) as numpy arrays
    """
    x, y = random_walk(runs_per_mol, total_runs, delta_x, delta_y)
    
    if save_plot:
        plot_random_walk(x, y, total_runs, runs_per_mol)
    
    return x, y

if __name__ == "__main__":
    # Run default simulation
    x, y = run_random_walk_simulation() 