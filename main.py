#!/usr/bin/env python3
"""
Main script for running diffusion simulations with adjustable parameters.
This script provides an interactive way to explore different diffusion scenarios.

Author: Agosh Saini - contact@agoshsaini.com
"""

import argparse
import os
import time
from datetime import datetime
from diffuse import (
    random_walk,
    plot_random_walk,
    create_video_from_images,
    ficks_2d_diffusion,
    plot_ficks_diffusion
)
import numpy as np

def create_results_dir():
    """Create a timestamped results directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"simulation_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def run_random_walk_simulation(args, results_dir):
    """Run the random walk simulation with given parameters."""
    print(f"\nRunning random walk simulation with {args.molecules} molecules...")
    print(f"Each molecule will make {args.movements} movements")
    
    start_time = time.time()
    
    # Create frames directory if video is requested
    frames_dir = os.path.join(results_dir, "frames")
    if args.create_video:
        os.makedirs(frames_dir, exist_ok=True)
    
    # Run the random walk simulation with intermediate states
    x_coords, y_coords = np.zeros((args.molecules, 1)), np.zeros((args.molecules, 1))
    
    # Save initial state
    if args.create_video:
        plot_random_walk(
            x=x_coords,
            y=y_coords,
            total_runs=args.molecules,
            runs_per_mol=0,
            save=True,
            output_file=os.path.join(frames_dir, "frame_0000.png"),
            show=False
        )
    
    # Run simulation step by step
    for step in range(args.movements):
        # Update positions
        for i in range(args.molecules):
            # X direction
            direction = np.random.randint(-2, 3)
            if direction == -2:
                x_coords[i] += -1 * args.delta_x
            elif direction == 2:
                x_coords[i] += 1 * args.delta_x
            
            # Y direction
            direction = np.random.randint(-2, 3)
            if direction == -2:
                y_coords[i] += -1 * args.delta_y
            elif direction == 2:
                y_coords[i] += 1 * args.delta_y
        
        # Save frame if video is requested
        if args.create_video and (step % max(1, args.movements // 100) == 0):  # Save ~100 frames
            frame_num = step // max(1, args.movements // 100)
            plot_random_walk(
                x=x_coords,
                y=y_coords,
                total_runs=args.molecules,
                runs_per_mol=step + 1,
                save=True,
                output_file=os.path.join(frames_dir, f"frame_{frame_num:04d}.png"),
                show=False
            )
    
    # Plot and save the final results
    plot_file = os.path.join(results_dir, "final_state.png")
    plot_random_walk(
        x=x_coords,
        y=y_coords,
        total_runs=args.molecules,
        runs_per_mol=args.movements,
        save=True,
        output_file=plot_file,
        show=True  # Show only the final state
    )
    
    # Create video if requested
    if args.create_video:
        print("\nCreating video of the simulation...")
        video_file = os.path.join(results_dir, "simulation.mp4")
        success = create_video_from_images(
            input_dir=frames_dir,
            output_file=video_file,
            fps=args.fps
        )
        if success:
            print(f"Video saved to: {video_file}")
    
    end_time = time.time()
    print(f"\nSimulation completed in {end_time - start_time:.2f} seconds")
    print(f"Results saved in: {results_dir}")

def run_ficks_simulation(args, results_dir):
    """Run Fick's law diffusion simulation with given parameters."""
    print("\nRunning Fick's law diffusion simulation...")
    print(f"Grid size: {args.grid_size}x{args.grid_size}")
    print(f"Time steps: {args.time_steps}")
    print(f"Diffusion coefficient: {args.diffusion_coeff}\n")
    
    # Run simulation
    concentration = ficks_2d_diffusion(
        nt=args.time_steps,
        nx=args.grid_size,
        ny=args.grid_size,
        coeff=args.diffusion_coeff,
        save_frames=args.create_video,
        output_dir=os.path.join(results_dir, "frames")
    )
    
    # Plot final state
    plot_ficks_diffusion(
        concentration[-1],
        save=True,
        show=True,
        output_dir=results_dir,
        output_file=os.path.join(results_dir, "final_state.png")
    )
    
    # Create video if requested
    if args.create_video:
        frames_dir = os.path.join(results_dir, "frames")
        video_path = os.path.join(results_dir, "diffusion_video.mp4")
        create_video_from_images(frames_dir, video_path, fps=args.fps)
        print(f"\nVideo saved to: {video_path}")

def main():
    parser = argparse.ArgumentParser(description='Run diffusion simulations with custom parameters')
    
    # Common arguments
    parser.add_argument('--simulation-type', type=str, choices=['random_walk', 'ficks'],
                       default='random_walk', help='Type of simulation to run (default: random_walk)')
    parser.add_argument('--create-video', action='store_true',
                       help='Create a video of the simulation')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second for video (default: 30)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    
    # Random walk specific arguments
    parser.add_argument('--molecules', type=int, default=10000,
                       help='Number of molecules to simulate (default: 10000)')
    parser.add_argument('--movements', type=int, default=100,
                       help='Number of movements per molecule (default: 100)')
    parser.add_argument('--delta-x', type=float, default=0.1,
                       help='Step size in x direction (default: 0.1)')
    parser.add_argument('--delta-y', type=float, default=0.1,
                       help='Step size in y direction (default: 0.1)')
    
    # Fick's law specific arguments
    parser.add_argument('--grid-size', type=int, default=100,
                       help='Grid size for Fick\'s law simulation (default: 100)')
    parser.add_argument('--time-steps', type=int, default=1000,
                       help='Number of time steps for Fick\'s law simulation (default: 1000)')
    parser.add_argument('--diffusion-coeff', type=float, default=0.1,
                       help='Diffusion coefficient for Fick\'s law (default: 0.1)')
    parser.add_argument('--boundary-x', type=str, default='periodic',
                       choices=['periodic', 'dirichlet', 'neumann'],
                       help='Boundary condition in x direction (default: periodic)')
    parser.add_argument('--boundary-y', type=str, default='periodic',
                       choices=['periodic', 'dirichlet', 'neumann'],
                       help='Boundary condition in y direction (default: periodic)')
    
    args = parser.parse_args()
    
    if args.interactive:
        print("Welcome to the Diffusion Simulation!")
        print("\nPlease enter the simulation parameters:")
        
        args.simulation_type = input("Simulation type (random_walk/ficks) [random_walk]: ") or "random_walk"
        
        if args.simulation_type == "random_walk":
            args.molecules = int(input("Number of molecules [10000]: ") or "10000")
            args.movements = int(input("Movements per molecule [100]: ") or "100")
            args.delta_x = float(input("Step size in x direction [0.1]: ") or "0.1")
            args.delta_y = float(input("Step size in y direction [0.1]: ") or "0.1")
        else:  # ficks
            args.grid_size = int(input("Grid size [100]: ") or "100")
            args.time_steps = int(input("Time steps [1000]: ") or "1000")
            args.diffusion_coeff = float(input("Diffusion coefficient [0.1]: ") or "0.1")
            args.boundary_x = input("Boundary condition in x direction (periodic/dirichlet/neumann) [periodic]: ") or "periodic"
            args.boundary_y = input("Boundary condition in y direction (periodic/dirichlet/neumann) [periodic]: ") or "periodic"
        
        args.create_video = input("Create video? (y/n) [n]: ").lower() == 'y'
        if args.create_video:
            args.fps = int(input("Frames per second [30]: ") or "30")
    
    # Create results directory
    results_dir = create_results_dir()
    
    # Run appropriate simulation
    if args.simulation_type == "random_walk":
        run_random_walk_simulation(args, results_dir)
    else:  # ficks
        run_ficks_simulation(args, results_dir)

if __name__ == "__main__":
    main()

