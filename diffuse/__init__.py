"""
Diffusion simulation package.
Provides both random walk and Fick's law diffusion simulations.

Author: Agosh Saini - contact@agoshsaini.com
"""

from .random_walk_diffusion import (
    random_walk,
    plot_random_walk,
    run_random_walk_simulation
)

from .ficks_2d_diffusion import (
    run_simulation as ficks_2d_diffusion,
    plot_ficks_diffusion,
    save_simulation_frames
)

from .create_video import create_video_from_images

__all__ = [
    'random_walk',
    'plot_random_walk',
    'run_random_walk_simulation',
    'ficks_2d_diffusion',
    'plot_ficks_diffusion',
    'save_simulation_frames',
    'create_video_from_images'
] 