# Two-Dimensional Diffusion Simulation

A Python package for simulating diffusion processes in 2D space using both Random Walk and Fick's Law approaches.

## What is Diffusion?

Diffusion is a process where a fluid of high concentration disperses in another fluid completely. 
This process is why we can mix gases together and liquids together. This is also why if you spray Febreze in one corner of a room, it then spreads everywhere in the room.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Two_Dimension_Diffusion.git
cd Two_Dimension_Diffusion

# Install dependencies
pip install numpy matplotlib opencv-python
```

## Usage

### Interactive Mode

The easiest way to run the simulation is in interactive mode:

```bash
python main.py --interactive
```

This will guide you through setting up the simulation parameters:
- Simulation type (random_walk or ficks)
- For random walk:
  - Number of molecules
  - Movements per molecule
  - Step sizes in X and Y directions
- For Fick's law:
  - Grid size
  - Time steps
  - Diffusion coefficient
  - Boundary conditions (periodic/dirichlet/neumann)
- Whether to create a video
- Frame rate for the video

### Command Line Arguments

You can also run the simulation with specific parameters:

```bash
# Random Walk Simulation
python main.py --simulation-type random_walk --molecules 10000 --movements 100 --delta-x 0.1 --delta-y 0.1 --create-video --fps 30

# Fick's Law Simulation
python main.py --simulation-type ficks --grid-size 100 --time-steps 1000 --diffusion-coeff 0.1 --boundary-x periodic --boundary-y periodic --create-video --fps 30
```

Available parameters:
- Common parameters:
  - `--simulation-type`: Type of simulation (random_walk/ficks)
  - `--create-video`: Create a video of the simulation
  - `--fps`: Frames per second for video (default: 30)
  - `--interactive`: Run in interactive mode

- Random walk parameters:
  - `--molecules`: Number of molecules to simulate (default: 10000)
  - `--movements`: Number of movements per molecule (default: 100)
  - `--delta-x`: Step size in X direction (default: 0.1)
  - `--delta-y`: Step size in Y direction (default: 0.1)

- Fick's law parameters:
  - `--grid-size`: Grid size for simulation (default: 100)
  - `--time-steps`: Number of time steps (default: 1000)
  - `--diffusion-coeff`: Diffusion coefficient (default: 0.1)
  - `--boundary-x`: Boundary condition in X direction (periodic/dirichlet/neumann)
  - `--boundary-y`: Boundary condition in Y direction (periodic/dirichlet/neumann)

### Using as a Module

You can also use the package in your own Python code:

```python
# Random Walk Simulation
from diffuse import random_walk, plot_random_walk

x, y = random_walk(
    runs_per_mol=150,
    total_runs=5000,
    delta_x=0.2,
    delta_y=0.2
)

plot_random_walk(
    x=x,
    y=y,
    total_runs=5000,
    runs_per_mol=150,
    save=True,
    output_file="my_simulation.png"
)

# Fick's Law Simulation
from diffuse import ficks_2d_diffusion, plot_ficks_diffusion

concentration = ficks_2d_diffusion(
    nt=1000,
    nx=100,
    ny=100,
    coeff=0.1,
    bd_condition_x='periodic',
    bd_condition_y='periodic'
)

plot_ficks_diffusion(
    concentration[-1],
    save=True,
    output_file="my_ficks_simulation.png"
)
```

## How It Works

### Random Walk

The Random Walk approach simulates individual molecules moving randomly in 2D space. For each molecule:
- It can move in X or Y direction
- It can choose not to move
- The probability of movement in any direction is 1/5 (including no movement)

This random motion is caused by the inherent random energy molecules possess.

### Fick's Law Diffusion

Fick's law diffusion provides a more mathematical approach to modeling diffusion processes. The simulation:
- Uses a finite difference method to solve the diffusion equation
- Supports different boundary conditions:
  - Periodic: Concentration wraps around the grid
  - Dirichlet: Fixed concentration at boundaries
  - Neumann: Fixed flux at boundaries
- Conserves total concentration throughout the simulation
- Can be visualized as a heat map showing concentration changes over time

## Results

All simulation results are saved in the `results` directory with the following structure:
```
results/
├── simulation_20240501_143000/  # Each simulation gets its own directory
│   ├── frames/                  # Directory containing animation frames
│   │   ├── frame_0000.png      # Initial state
│   │   ├── frame_0001.png      # First step
│   │   └── ...
│   ├── final_state.png         # Final state plot
│   └── simulation.mp4          # Video of the entire simulation
└── test/                       # Test results
```

## Running Tests

The package includes comprehensive tests for both simulation types. To run the tests:

```bash
# Run all tests
python -m unittest discover tests

# Run specific test classes
python -m unittest tests.test_diffusion.TestRandomWalkDiffusion
python -m unittest tests.test_diffusion.TestFicksDiffusion

# Run specific test methods
python -m unittest tests.test_diffusion.TestRandomWalkDiffusion.test_random_walk_bounds
python -m unittest tests.test_diffusion.TestFicksDiffusion.test_ficks_conservation
```

The test suite verifies:
- Output shapes and dimensions
- Physical constraints (bounds, conservation)
- File operations (plotting, video creation)
- Different simulation parameters
- Boundary conditions
- Error cases

## Example Videos

Here are examples of both simulation types:

[![Random Walk Diffusion](https://img.youtube.com/vi/Ww4VwPqWFYc/0.jpg)](https://www.youtube.com/watch?v=Ww4VwPqWFYc "Random Walk Diffusion")
[![Fick's Law Diffusion](https://img.youtube.com/vi/Ww4VwPqWFYc/0.jpg)](https://www.youtube.com/watch?v=Ww4VwPqWFYc "Fick's Law Diffusion")

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Agosh Saini - contact@agoshsaini.com
