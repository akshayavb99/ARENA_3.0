# %%
import os
import sys
from functools import partial
from pathlib import Path
from typing import Callable

import einops
import plotly.express as px
import plotly.graph_objects as go
import torch as t
from IPython.display import display
from ipywidgets import interact, widgets
from jaxtyping import Bool, Float
from torch import Tensor
from tqdm import tqdm

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part1_ray_tracing"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part1_ray_tracing.tests as tests
from part1_ray_tracing.utils import (
    render_lines_with_plotly,
    setup_widget_fig_ray,
    setup_widget_fig_triangle,
)
from plotly_utils import imshow

MAIN = __name__ == "__main__"

# %% [markdown]
## Rays and Segments
### `make_rays_1d`
# %%
def make_rays_1d(num_pixels: int, y_limit: float) -> Tensor:
    """
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is
        also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains
        (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    """
    result = t.zeros(num_pixels, 2, 3) # Number of rays = number of pixels, each ray is identified by 2 points and each point has 3 dimensions
    y_values = t.linspace(-y_limit, y_limit, num_pixels) # Create evenly spaced y values from -y_limit to y_limit
    result[:, 1, 0] = 1.0 # Set the x-coordinate of the direction point to 1.0 for all rays
    result[:, 1, 1] = y_values # Set the y-coordinate of the direction point to the computed y values
    return result

# %%
rays1d = make_rays_1d(9, 10.0)
fig = render_lines_with_plotly(rays1d)

# %% [markdown]
### Interactive widget to understand intersection of camera ray and object line

# %%
fig: go.FigureWidget = setup_widget_fig_ray()
display(fig)

# %%
@interact(v=(0.0, 6.0, 0.01), seed=(0, 10, 1))
def update(v=0.0, seed=0):
    t.manual_seed(seed)
    L_1, L_2 = t.rand(2, 2)
    P = lambda v: L_1 + v * (L_2 - L_1)
    x, y = zip(P(0), P(6))
    with fig.batch_update():
        fig.update_traces({"x": x, "y": y}, 0)
        fig.update_traces({"x": [L_1[0], L_2[0]], "y": [L_1[1], L_2[1]]}, 1)
        fig.update_traces({"x": [P(v)[0]], "y": [P(v)[1]]}, 2)
# %%
# %%
import ipywidgets as widgets
from IPython.display import clear_output, display

# 1. Create a clean Output widget to hold our plot
out = widgets.Output()

# 2. Define the update function without mutating a global FigureWidget
@interact(v=(0.0, 6.0, 0.01), seed=(0, 10, 1))
def update(v=0.0, seed=0):
    t.manual_seed(seed)
    L_1, L_2 = t.rand(2, 2)
    P = lambda v: L_1 + v * (L_2 - L_1)
    x, y = zip(P(0), P(6))
    
    # Generate a fresh figure on the backend
    new_fig = setup_widget_fig_ray()
    
    # Safely modify the data points using direct structural updates
    new_fig.data[0].x = x
    new_fig.data[0].y = y
    
    new_fig.data[1].x = [L_1[0], L_2[0]]
    new_fig.data[1].y = [L_1[1], L_2[1]]
    
    new_fig.data[2].x = [P(v)[0]]
    new_fig.data[2].y = [P(v)[1]]
    
    # Render the static HTML/JS copy inside the output widget context
    with out:
        clear_output(wait=True)
        display(new_fig)

# 3. Render the output display window beneath the sliders
display(out)
# %% [markdown]
### `intersect_ray_1d`
# %%
def intersect_ray_1d(ray: Float[Tensor, "points dims"], segment: Float[Tensor, "points dims"]) -> bool:
    """
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    """
    
    """
    Equation is O + uD = L_1 + v (L_2 - L_1). 
    The equation can be used for both x and y coordinates.
    O_x + uD_x = L_1_x + v (L_2_x - L_1_x) and O_y + uD_y = L_1_y + v (L_2_y - L_1_y)
    The 2 equations can be stacked together to form a linear system Ax = b, where:
        x = [u, v]^T
        A is [[D_x, -(L_2_x - L_1_x)], [D_y, -(L_2_y - L_1_y)]]
        b = [L_1_x - O_x, L_1_y - O_y]^T
    """
    
    # Defining A, x, b
    O, D = ray[0, :2], ray[1, :2]  # Extracting the origin and direction points from the ray
    L_1, L_2 = segment[0, :2], segment[1, :2] # Extracting the endpoints of the segment
    A = t.tensor([[D[0], -(L_2[0] - L_1[0])], [D[1], -(L_2[1] - L_1[1])]])
    b = t.tensor([L_1[0] - O[0], L_1[1] - O[1]])
    try:
        x = t.linalg.solve(A, b)
        if t.isnan(x).any():
            return False
        else:
            if (x[0] >= 0) and (0 <= x[1] <= 1):
                return True
            else:
                return False
    except RuntimeError:
        return False
    
tests.test_intersect_ray_1d(intersect_ray_1d)
tests.test_intersect_ray_1d_special_case(intersect_ray_1d)

# %%
