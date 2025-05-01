"""
Test suite for diffusion simulations.
Tests both random walk and Fick's law diffusion with various parameters and edge cases.
"""

import unittest
import numpy as np
import os
import sys
from pathlib import Path

# Add parent directory to path to import diffuse package
sys.path.append(str(Path(__file__).parent.parent))

from diffuse import (
    random_walk,
    plot_random_walk,
    ficks_2d_diffusion,
    plot_ficks_diffusion,
    create_video_from_images
)

class TestRandomWalk(unittest.TestCase):
    """Test cases for random walk diffusion simulation."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test parameters and directories."""
        cls.test_dir = "test_output"
        os.makedirs(cls.test_dir, exist_ok=True)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test directories."""
        if os.path.exists(cls.test_dir):
            for file in os.listdir(cls.test_dir):
                os.remove(os.path.join(cls.test_dir, file))
            os.rmdir(cls.test_dir)
    
    def setUp(self):
        """Set up test parameters."""
        self.n_molecules = 1000
        self.n_steps = 50
        self.delta_x = 0.1
        self.delta_y = 0.1
    
    def test_output_shape(self):
        """Test that random walk outputs have correct shapes."""
        x, y = random_walk(
            total_runs=self.n_molecules,
            runs_per_mol=self.n_steps,
            delta_x=self.delta_x,
            delta_y=self.delta_y
        )
        self.assertEqual(x.shape, (self.n_molecules, self.n_steps))
        self.assertEqual(y.shape, (self.n_molecules, self.n_steps))
    
    def test_bounds(self):
        """Test that random walk stays within expected bounds."""
        x, y = random_walk(
            total_runs=self.n_molecules,
            runs_per_mol=self.n_steps,
            delta_x=self.delta_x,
            delta_y=self.delta_y
        )
        max_possible = self.n_steps * self.delta_x
        self.assertTrue(np.all(np.abs(x) <= max_possible))
        self.assertTrue(np.all(np.abs(y) <= max_possible))
    
    def test_step_size(self):
        """Test that step size affects the spread of molecules."""
        # Run with small step size
        x_small, y_small = random_walk(
            total_runs=100,
            runs_per_mol=50,
            delta_x=0.05,
            delta_y=0.05
        )
        
        # Run with large step size
        x_large, y_large = random_walk(
            total_runs=100,
            runs_per_mol=50,
            delta_x=0.2,
            delta_y=0.2
        )
        
        # Calculate spread (standard deviation of final positions)
        spread_small = np.std(x_small[:, -1])
        spread_large = np.std(x_large[:, -1])
        
        self.assertGreater(spread_large, spread_small)
    
    def test_plot_saving(self):
        """Test that random walk plots can be saved."""
        x, y = random_walk(
            total_runs=100,
            runs_per_mol=20,
            delta_x=self.delta_x,
            delta_y=self.delta_y
        )
        
        output_file = os.path.join(self.test_dir, "test_plot.png")
        plot_random_walk(
            x=x,
            y=y,
            total_runs=100,
            runs_per_mol=20,
            save=True,
            output_file=output_file,
            show=False
        )
        
        self.assertTrue(os.path.exists(output_file))

class TestFicksLaw(unittest.TestCase):
    """Test cases for Fick's law diffusion simulation."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test parameters and directories."""
        cls.test_dir = "test_output"
        cls.frames_dir = "test_frames"
        os.makedirs(cls.test_dir, exist_ok=True)
        os.makedirs(cls.frames_dir, exist_ok=True)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test directories."""
        for dir_path in [cls.test_dir, cls.frames_dir]:
            if os.path.exists(dir_path):
                for file in os.listdir(dir_path):
                    os.remove(os.path.join(dir_path, file))
                os.rmdir(dir_path)
    
    def setUp(self):
        """Set up test parameters."""
        self.nx = 30
        self.ny = 30
        self.base_coeff = 0.2
        # Calculate required time steps for stability with safety margin
        self.nt = int(np.ceil(169 * (self.base_coeff / 0.2) * 1.1))  # Add 10% safety margin
    
    def calculate_stable_nt(self, nx, ny, coeff):
        """Calculate stable number of time steps for given parameters."""
        base_nt = 169  # Base number of time steps for 30x30 grid and coeff=0.2
        grid_factor = (nx * ny) / (30 * 30)  # Scale by grid size
        coeff_factor = coeff / 0.2  # Scale by diffusion coefficient
        return int(np.ceil(base_nt * grid_factor * coeff_factor * 1.1))  # Add 10% safety margin
    
    def test_output_shape(self):
        """Test that Fick's law simulation outputs have correct shape."""
        concentration = ficks_2d_diffusion(
            nt=self.nt,
            nx=self.nx,
            ny=self.ny,
            coeff=self.base_coeff
        )
        self.assertEqual(concentration.shape, (self.nt, self.nx, self.ny))
    
    def test_conservation(self):
        """Test that total concentration is conserved."""
        concentration = ficks_2d_diffusion(
            nt=self.nt,
            nx=self.nx,
            ny=self.ny,
            coeff=self.base_coeff
        )
        initial_total = np.sum(concentration[0])
        final_total = np.sum(concentration[-1])
        self.assertAlmostEqual(initial_total, final_total, places=10)
    
    def test_boundary_conditions(self):
        """Test different boundary conditions."""
        # Test periodic boundaries
        conc_periodic = ficks_2d_diffusion(
            nt=self.nt, nx=self.nx, ny=self.ny, coeff=self.base_coeff,
            bd_condition_x='periodic', bd_condition_y='periodic'
        )
        
        # Test Dirichlet boundaries
        conc_dirichlet = ficks_2d_diffusion(
            nt=self.nt, nx=self.nx, ny=self.ny, coeff=self.base_coeff,
            bd_condition_x='dirichlet', bd_condition_y='dirichlet'
        )
        
        # Test Neumann boundaries
        conc_neumann = ficks_2d_diffusion(
            nt=self.nt, nx=self.nx, ny=self.ny, coeff=self.base_coeff,
            bd_condition_x='neumann', bd_condition_y='neumann'
        )
        
        # Check that all simulations completed
        self.assertEqual(conc_periodic.shape, (self.nt, self.nx, self.ny))
        self.assertEqual(conc_dirichlet.shape, (self.nt, self.nx, self.ny))
        self.assertEqual(conc_neumann.shape, (self.nt, self.nx, self.ny))
        
        # Check boundary behavior
        # Periodic: values should wrap around
        self.assertTrue(np.allclose(conc_periodic[:, 0, :], conc_periodic[:, -1, :]))
        self.assertTrue(np.allclose(conc_periodic[:, :, 0], conc_periodic[:, :, -1]))
        
        # Dirichlet: boundaries should be zero
        self.assertTrue(np.allclose(conc_dirichlet[:, 0, :], 0))
        self.assertTrue(np.allclose(conc_dirichlet[:, -1, :], 0))
        self.assertTrue(np.allclose(conc_dirichlet[:, :, 0], 0))
        self.assertTrue(np.allclose(conc_dirichlet[:, :, -1], 0))
    
    def test_diffusion_rate(self):
        """Test that higher diffusion coefficient leads to faster diffusion."""
        # Calculate time steps needed for each coefficient
        nt_slow = self.calculate_stable_nt(self.nx, self.ny, 0.1)
        nt_fast = self.calculate_stable_nt(self.nx, self.ny, 0.3)
        
        # Run with low diffusion coefficient
        conc_slow = ficks_2d_diffusion(
            nt=nt_slow, nx=self.nx, ny=self.ny, coeff=0.1
        )
        
        # Run with high diffusion coefficient
        conc_fast = ficks_2d_diffusion(
            nt=nt_fast, nx=self.nx, ny=self.ny, coeff=0.3
        )
        
        # Calculate rate of change for both simulations
        slow_rate = np.mean(np.abs(np.diff(conc_slow, axis=0)))
        fast_rate = np.mean(np.abs(np.diff(conc_fast, axis=0)))
        
        # Higher coefficient should lead to faster changes
        self.assertGreater(fast_rate, slow_rate)
    
    def test_grid_size(self):
        """Test that grid size affects simulation accuracy."""
        # Calculate stable time steps for each grid size
        nt_small = self.calculate_stable_nt(15, 15, self.base_coeff)
        nt_large = self.calculate_stable_nt(60, 60, self.base_coeff)
        
        # Run with small grid
        conc_small = ficks_2d_diffusion(
            nt=nt_small, nx=15, ny=15, coeff=self.base_coeff
        )
        
        # Run with large grid
        conc_large = ficks_2d_diffusion(
            nt=nt_large, nx=60, ny=60, coeff=self.base_coeff
        )
        
        # Larger grid should have smoother concentration gradients
        small_gradients = np.mean(np.abs(np.diff(conc_small, axis=1)))
        large_gradients = np.mean(np.abs(np.diff(conc_large, axis=1)))
        
        self.assertLess(large_gradients, small_gradients)
    
    def test_plot_saving(self):
        """Test that Fick's law plots can be saved."""
        concentration = ficks_2d_diffusion(
            nt=self.nt,
            nx=self.nx,
            ny=self.ny,
            coeff=self.base_coeff
        )
        
        output_file = os.path.join(self.test_dir, "test_plot.png")
        plot_ficks_diffusion(
            concentration[-1],
            save=True,
            show=False,
            output_dir=self.test_dir,
            output_file=output_file
        )
        
        self.assertTrue(os.path.exists(output_file))
    
    def test_video_creation(self):
        """Test that videos can be created from simulation frames."""
        concentration = ficks_2d_diffusion(
            nt=self.nt, nx=self.nx, ny=self.ny, coeff=self.base_coeff,
            save_frames=True,
            output_dir=self.frames_dir
        )
        
        video_path = os.path.join(self.test_dir, "test_video.mp4")
        success = create_video_from_images(
            input_dir=self.frames_dir,
            output_file=video_path,
            fps=30
        )
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(video_path))

if __name__ == '__main__':
    unittest.main() 