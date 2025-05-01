"""
2D Diffusion Simulation using Fick's Law

This module implements a 2D diffusion simulation using finite volume method
and Fick's law of diffusion. It can simulate how a substance diffuses in a
2D space over time.

Author: Agosh Saini - conact@agoshsaini.com

Resources used:
    http://math.tifrbng.res.in/~praveen/notes/cm2013/heat_2d.pdf
    https://en.wikipedia.org/wiki/Finite_volume_method_for_two_dimensional_diffusion_problem#cite_note-3
    http://opencourses.emu.edu.tr/course/view.php?id=27&lang=en
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
import time
import os

def check_stability(dx, dy, dt, t_rng, coeff):
    """
    Check if the simulation parameters will result in a stable solution.
    
    Args:
        dx (float): Grid spacing in x direction
        dy (float): Grid spacing in y direction
        dt (float): Time step
        t_rng (numpy.ndarray): Time range [t_start, t_end]
        coeff (float): Diffusion coefficient
        
    Returns:
        int: Stable number of time steps if current dt is unstable
        
    Raises:
        Exception: If time step is too large for stability
    """
    # Modified stability condition to allow for larger time steps
    max_dt = (1 / (4*coeff)) * (dx*dy)**2 / (dx**2 + dy**2)
    
    if dt > max_dt: 
        nt_stable = (t_rng[1]-t_rng[0]) / max_dt + 1
        raise Exception('time step too large, system unstable. use nt of ' + str(nt_stable))

def grid_formation(nt, nx, ny):
    """
    Create the initial grid for the simulation with a clear concentration pattern.
    
    Args:
        nt (int): Number of time steps
        nx (int): Number of grid points in x direction
        ny (int): Number of grid points in y direction
        
    Returns:
        numpy.ndarray: 3D array with initial concentration pattern
    """
    # Create empty grid
    grid = np.zeros([nt, nx, ny])
    
    # Add initial concentration pattern (Gaussian blob in the center)
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    # Create a Gaussian blob
    sigma = 0.3  # Wider initial pattern
    initial_pattern = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    
    # Normalize to have max concentration of 1
    initial_pattern = initial_pattern / np.max(initial_pattern)
    
    # Set initial condition
    grid[0] = initial_pattern
    
    return grid

def boundary_cond(grid, bd_condition_x, bd_condition_y, nx, ny):
    """
    Apply boundary conditions to the grid.
    
    Parameters:
    -----------
    grid : numpy.ndarray
        The concentration grid
    bd_condition_x : str
        Boundary condition for x direction ('periodic', 'dirichlet', or 'neumann')
    bd_condition_y : str
        Boundary condition for y direction ('periodic', 'dirichlet', or 'neumann')
    nx : int
        Number of grid points in x direction
    ny : int
        Number of grid points in y direction
    
    Returns:
    --------
    numpy.ndarray
        Grid with boundary conditions applied
    """
    if bd_condition_x == 'periodic':
        # Periodic boundary conditions in x
        grid[:, 0, :] = grid[:, -2, :]
        grid[:, -1, :] = grid[:, 1, :]
    elif bd_condition_x == 'dirichlet':
        # Dirichlet boundary conditions in x (fixed concentration)
        grid[:, 0, :] = 0.0
        grid[:, -1, :] = 0.0
    elif bd_condition_x == 'neumann':
        # Neumann boundary conditions in x (zero flux)
        grid[:, 0, :] = grid[:, 1, :]
        grid[:, -1, :] = grid[:, -2, :]
    
    if bd_condition_y == 'periodic':
        # Periodic boundary conditions in y
        grid[:, :, 0] = grid[:, :, -2]
        grid[:, :, -1] = grid[:, :, 1]
    elif bd_condition_y == 'dirichlet':
        # Dirichlet boundary conditions in y (fixed concentration)
        grid[:, :, 0] = 0.0
        grid[:, :, -1] = 0.0
    elif bd_condition_y == 'neumann':
        # Neumann boundary conditions in y (zero flux)
        grid[:, :, 0] = grid[:, :, 1]
        grid[:, :, -1] = grid[:, :, -2]
    
    return grid

def diffusion_iteration(u, dt, dx, dy, nt, coeff, show_progress=True):
    """
    Perform the diffusion simulation using FTCS scheme.
    
    Args:
        u (numpy.ndarray): The simulation grid
        dt (float): Time step
        dx (float): Grid spacing in x direction
        dy (float): Grid spacing in y direction
        nt (int): Number of time steps
        coeff (float): Diffusion coefficient
        show_progress (bool): Whether to show progress bar
        
    Returns:
        numpy.ndarray: Updated grid after diffusion simulation
    """
    alpha, beta = coeff**2*dt/dx**2, coeff**2*dt/dy**2
    
    # Create progress bar if requested
    iterator = tqdm(range(nt - 1), desc="Simulating diffusion") if show_progress else range(nt - 1)
    
    for i in iterator:
        u_x = alpha*(u[i, 2:, 1:-1] - 2*u[i, 1:-1, 1:-1] + u[i, :-2, 1:-1])
        u_y = beta*(u[i, 1:-1, 2:] - 2*u[i, 1:-1, 1:-1] + u[i, 1:-1, :-2])
        u[i+1, 1:-1, 1:-1] = u[i, 1:-1, 1:-1] + u_x + u_y 
    
    return u

def plot_ficks_diffusion(grid, save=True, show=False, output_dir="data", output_file=None):
    """
    Plot a single frame of the diffusion simulation.
    
    Args:
        grid (numpy.ndarray): The concentration grid to plot
        save (bool): Whether to save the plot
        show (bool): Whether to display the plot
        output_dir (str): Directory to save the plot
        output_file (str): Specific file to save the plot to
    """
    plt.figure(figsize=(8, 6))
    im = plt.imshow(grid, cmap='hot', vmin=0, vmax=1)
    plt.colorbar(im, label='Concentration')
    plt.title('Diffusion State')
    
    if save:
        if output_file is None:
            output_file = os.path.join(output_dir, "diffusion_state.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()

def save_simulation_frames(grid, nt, output_dir="data", show_progress=True):
    """
    Save simulation frames as images.
    
    Args:
        grid (numpy.ndarray): The simulation grid
        nt (int): Number of time steps
        output_dir (str): Directory to save the images
        show_progress (bool): Whether to show progress bar
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create progress bar if requested
    iterator = tqdm(range(nt), desc="Saving frames") if show_progress else range(nt)
    
    for i in iterator:
        plot_ficks_diffusion(
            grid[i],
            save=True,
            show=False,
            output_dir=output_dir,
            output_file=os.path.join(output_dir, f"{i}.png")
        )

def run_simulation(nt=50, nx=30, ny=30, coeff=0.2, 
                  t_rng=None, x_rng=None, y_rng=None,
                  bd_condition_x='periodic', bd_condition_y='periodic',
                  save_frames=True, output_dir="data", show_progress=True):
    """
    Run a complete diffusion simulation.
    
    Args:
        nt (int): Number of time steps (default: 50)
        nx (int): Number of grid points in x direction (default: 30)
        ny (int): Number of grid points in y direction (default: 30)
        coeff (float): Diffusion coefficient (default: 0.2)
        t_rng (numpy.ndarray): Time range [t_start, t_end]
        x_rng (numpy.ndarray): X range [x_start, x_end]
        y_rng (numpy.ndarray): Y range [y_start, y_end]
        bd_condition_x (str): Boundary condition for x direction ('periodic', 'dirichlet', or 'neumann')
        bd_condition_y (str): Boundary condition for y direction ('periodic', 'dirichlet', or 'neumann')
        save_frames (bool): Whether to save simulation frames
        output_dir (str): Directory to save the images
        show_progress (bool): Whether to show progress updates
    """
    if show_progress:
        print(f"\nStarting simulation with parameters:")
        print(f"Grid size: {nx}x{ny}")
        print(f"Time steps: {nt}")
        print(f"Diffusion coefficient: {coeff}")
        print(f"Boundary conditions: x={bd_condition_x}, y={bd_condition_y}")
        print("\nInitializing...")
    
    start_time = time.time()
    
    if t_rng is None:
        t_rng = np.array([0, 0.5])  # Longer time range
    if x_rng is None:
        x_rng = np.array([0, 2])    # Larger spatial range
    if y_rng is None:
        y_rng = np.array([0, 2])    # Larger spatial range
        
    dt = (t_rng[1]-t_rng[0])/(nt-1)
    dx = (x_rng[1]-x_rng[0])/(nx-1)
    dy = (y_rng[1]-y_rng[0])/(ny-1)
    
    # Check stability
    check_stability(dx, dy, dt, t_rng, coeff)
    
    # Run simulation
    grid = grid_formation(nt, nx, ny)
    grid = boundary_cond(grid, bd_condition_x, bd_condition_y, nx, ny)
    grid = diffusion_iteration(grid, dt, dx, dy, nt, coeff, show_progress)
    
    # Save frames if requested
    if save_frames:
        if show_progress:
            print("\nSaving simulation frames...")
        save_simulation_frames(grid, nt, output_dir, show_progress)
    
    if show_progress:
        end_time = time.time()
        print(f"\nSimulation completed in {end_time - start_time:.2f} seconds")
    
    return grid

if __name__ == "__main__":
    # Run default simulation
    grid = run_simulation()


    




