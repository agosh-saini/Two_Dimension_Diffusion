"""
Test script for Fick's law diffusion simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from diffuse import ficks_2d_diffusion, plot_ficks_diffusion

def test_basic_diffusion():
    """Test basic diffusion with default parameters."""
    print("Running basic diffusion test...")
    concentration = ficks_2d_diffusion(
        nt=100,  # time steps
        nx=50,   # grid size x
        ny=50,   # grid size y
        coeff=0.1  # diffusion coefficient
    )
    
    # Plot initial and final states
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    plot_ficks_diffusion(concentration[0], show=False)
    plt.title("Initial State")
    
    plt.subplot(122)
    plot_ficks_diffusion(concentration[-1], show=False)
    plt.title("Final State")
    
    plt.savefig("basic_diffusion_test.png")
    plt.close()
    
    print("Test completed. Check basic_diffusion_test.png for results.")

def test_different_boundary_conditions():
    """Test diffusion with different boundary conditions."""
    print("\nTesting different boundary conditions...")
    
    boundary_conditions = ['periodic', 'dirichlet', 'neumann']
    
    plt.figure(figsize=(15, 5))
    for i, bc in enumerate(boundary_conditions, 1):
        concentration = ficks_2d_diffusion(
            nt=100,
            nx=50,
            ny=50,
            coeff=0.1,
            bd_condition_x=bc,
            bd_condition_y=bc
        )
        
        plt.subplot(1, 3, i)
        plot_ficks_diffusion(concentration[-1], show=False)
        plt.title(f"Boundary: {bc}")
    
    plt.savefig("boundary_conditions_test.png")
    plt.close()
    
    print("Test completed. Check boundary_conditions_test.png for results.")

def test_different_diffusion_coefficients():
    """Test diffusion with different coefficients."""
    print("\nTesting different diffusion coefficients...")
    
    coefficients = [0.05, 0.1, 0.2]
    
    plt.figure(figsize=(15, 5))
    for i, coeff in enumerate(coefficients, 1):
        concentration = ficks_2d_diffusion(
            nt=100,
            nx=50,
            ny=50,
            coeff=coeff
        )
        
        plt.subplot(1, 3, i)
        plot_ficks_diffusion(concentration[-1], show=False)
        plt.title(f"Coefficient: {coeff}")
    
    plt.savefig("diffusion_coefficients_test.png")
    plt.close()
    
    print("Test completed. Check diffusion_coefficients_test.png for results.")

def test_conservation():
    """Test that total concentration is conserved."""
    print("\nTesting concentration conservation...")
    
    concentration = ficks_2d_diffusion(
        nt=100,
        nx=50,
        ny=50,
        coeff=0.1
    )
    
    initial_total = np.sum(concentration[0])
    final_total = np.sum(concentration[-1])
    
    print(f"Initial total concentration: {initial_total:.6f}")
    print(f"Final total concentration: {final_total:.6f}")
    print(f"Difference: {abs(initial_total - final_total):.6f}")
    
    if abs(initial_total - final_total) < 1e-10:
        print("Conservation test passed!")
    else:
        print("Warning: Concentration not conserved!")

if __name__ == "__main__":
    print("Starting Fick's law diffusion tests...")
    
    # Run all tests
    test_basic_diffusion()
    test_different_boundary_conditions()
    test_different_diffusion_coefficients()
    test_conservation()
    
    print("\nAll tests completed!") 