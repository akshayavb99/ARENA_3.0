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
## Batched Operations
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

# %% [markdown]
### `intersect_rays_1d` - Batched operations

# %%
def intersect_rays_1d(
    rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]
) -> Bool[Tensor, " nrays"]:
    """
    For each ray, return True if it intersects any segment.
    """
    # Using einops to repeat rays and segments to create a grid of all possible intersection combinations
    rays_expanded = einops.repeat(rays, "nr p c -> nr ns p c", ns = segments.shape[0])
    segments_expanded = einops.repeat(segments, "ns p c -> nr ns p c", nr = rays.shape[0])
    
    # Getting the origin and direction points from the rays and the endpoints from the segments
    O = rays_expanded[:, :, 0, :2]  # shape (nrays, nsegments, 2)
    D = rays_expanded[:, :, 1, :2]  # shape (nrays, nsegments, 2)
    L_1 = segments_expanded[:, :, 0, :2]  # shape (nrays, nsegments, 2)
    L_2 = segments_expanded[:, :, 1, :2]  # shape (nrays, nsegments, 2)
    
    # Construct matrix for each ray and segment pair
    A = t.stack([D, -(L_2 - L_1)], dim=-1)  # shape (nrays, nsegments, 2, 2)
    
    # Construct the RHS vector for the linear system of equations
    b = L_1 - O  # shape (nrays, nsegments, 2)
    
    # Find determinants for each ray-segment pair to check for singularity
    det_A = t.linalg.det(A)  # shape (nrays, nsegments)
    is_singular = det_A.abs() < 1e-8  # shape (nrays, nsegments)
    
    # Unsqueeze is_singular from (nr, ns) -> (nr, ns, 1, 1) to broadcast with (nr, ns, 2, 2)
    # t.eye(2) will automatically broadcast its trailing (2, 2) dimensions
    A_safe = t.where(is_singular.unsqueeze(-1).unsqueeze(-1), t.eye(2), A)

    # Solve the linear system 
    # Note: matrix (nr, ns, 2, 2) and vector b (nr, ns, 2) matches natively, no squeeze needed!
    x = t.linalg.solve(A_safe, b)
    u, v = x[..., 0], x[..., 1]  # shape (nrays, nsegments)
    
    # Apply logic conditions to determine if rays intersect segments in front of the camera and within the segment bounds
    intersects = (u >= 0) & (v >= 0) & (v <= 1) & (~is_singular)  # shape (nrays, nsegments)
    
    return intersects.any(dim=1)  # shape (nrays,)
    
tests.test_intersect_rays_1d(intersect_rays_1d)
tests.test_intersect_rays_1d_special_case(intersect_rays_1d)

# %% [markdown]
### Implement `make_rays_2d`

# %%
def make_rays_2d(num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float) -> Float[Tensor, "nrays 2 3"]:
    """
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    """
    
    result = t.zeros(num_pixels_y * num_pixels_z, 2, 3)  # Initialize the result tensor
    y_values = t.linspace(-y_limit, y_limit, num_pixels_y)  # Create evenly spaced y values from -y_limit to y_limit
    z_values = t.linspace(-z_limit, z_limit, num_pixels_z)  # Create evenly spaced z values from -z_limit to z_limit
    y_grid, z_grid = t.meshgrid(y_values, z_values, indexing='ij')  # Create a grid of y and z values
    result[:, 1, 0] = 1.0  # Set the x-coordinate of the direction point to 1.0 for all rays
    result[:, 1, 1] = y_grid.flatten()  # Set the y-coordinate of the direction point to the flattened y grid
    result[:, 1, 2] = z_grid.flatten()  # Set the z-coordinate of the direction point to the flattened z grid
    return result


rays_2d = make_rays_2d(10, 10, 0.3, 0.3)
render_lines_with_plotly(rays_2d)

# %% [markdown]
## Triangles
### Exercise - implement `triangle_ray_intersects`

# %%
Point = Float[Tensor, "points=3"]


def triangle_ray_intersects(A: Point, B: Point, C: Point, O: Point, D: Point) -> bool:
    """
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the ray and the triangle intersect.
    """
    
    """
    To find if the triangle and ray intersect, we use 2 steps:
    1. Find the intersection point of the ray with the plane defined by the triangle by solving linear system of equations P(u, v) = P(s)
    2. Ensure u and v hold values such that the intersection point lies within the triangle (u >= 0, v >= 0, u + v <= 1) and s >= 0 for the ray to intersect in front of the camera.
    System of equations: A + u(B - A) + v(C - A) = O + s(D - O)
    Rearranging gives: u(B - A) + v(C - A) - s(D - O) = O - A
    This can be expressed in matrix form as Ax = b, where:
        x = [u, v, s]^T
        M = [[B - A, C - A, -(D - O)]]
        b = O - A
    M and b can be split to have 3 rows each corresponding to the x, y, and z coordinates of the points.
    """
    
    M = t.stack([-D, B-A, C-A], dim=1)  # shape (3, 3)
    b = O - A  # shape (3,)
    try:
        x = t.linalg.solve(M, b)  # shape (3,)
        s, u, v = x
        if s >= 0 and u >= 0 and v >= 0 and (u + v) <= 1:
            return True
        else:
            return False
    except t.linalg.LinAlgError:
        return False      
    
tests.test_triangle_ray_intersects(triangle_ray_intersects)
# %% [markdown]
### Exercise - implement `raytrace_traingle`

# %%
def raytrace_triangle(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"],
) -> Bool[Tensor, " nrays"]:
    """
    For each ray, return True if the triangle intersects that ray.
    """
    
    """
    The system of equations needs to be expanded to accomodate nrays instead of 1 ray. 
    This means adding an extra dimension to the traingle points to repeat it across nrays
    """
    triangle_expanded = einops.repeat(triangle, "points dims -> nrays points dims", nrays = rays.shape[0])
    O, D = rays[:, 0, :], rays[:, 1, :]  # shape (nrays, 3)
    A, B, C = triangle_expanded[:, 0, :], triangle_expanded[:, 1, :], triangle_expanded[:, 2, :]  # shape (nrays, 3)
    M = t.stack([-D, B-A, C-A], dim=-1)  # shape (nrays, 3, 3)
    b = O - A  # shape (nrays, 3)
    try:
        s,u,v = t.linalg.solve(M, b).T
        intersects = (s >= 0) & (u >= 0) & (v >= 0) & ((u + v) <= 1)
        return intersects
    except t.linalg.LinAlgError:
        return t.zeros(rays.shape[0], dtype=t.bool)

A = t.tensor([1, 0.0, -0.5])
B = t.tensor([1, -0.5, 0.0])
C = t.tensor([1, 0.5, 0.5])
num_pixels_y = num_pixels_z = 15
y_limit = z_limit = 0.5

# Plot triangle & rays
test_triangle = t.stack([A, B, C], dim=0)
rays2d = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
triangle_lines = t.stack([A, B, C, A, B, C], dim=0).reshape(-1, 2, 3)
render_lines_with_plotly(rays2d, triangle_lines)

# Calculate and display intersections
intersects = raytrace_triangle(rays2d, test_triangle)
img = intersects.reshape(num_pixels_y, num_pixels_z).int()
imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")
# %% [markdown]
### Mesh Loading and Rendering
triangles = t.load(section_dir / "pikachu.pt", weights_only=True)

# %% [markdown]
### Exercise - implement `raytrace_mesh`

# %%
def raytrace_mesh(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"],
) -> Float[Tensor, " nrays"]:
    """
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    """
    """
    The system of equations needs to be expanded to accomodate nrays and ntriangles instead of 1 ray and 1 triangle. 
    This means adding an extra dimension to the traingle points to repeat it across nrays
    """
    triangles_expanded = einops.repeat(triangles, "ntriangles trianglePoints dims -> nrays ntriangles trianglePoints dims", nrays = rays.shape[0])
    assert triangles_expanded.shape == (rays.shape[0], triangles.shape[0], 3, 3)
    
    rays_expanded = einops.repeat(rays, "nrays rayPoints dims -> nrays ntriangles rayPoints dims", ntriangles = triangles.shape[0])
    assert rays_expanded.shape == (rays.shape[0], triangles.shape[0], 2, 3)
    
    O, D = rays_expanded[:, :, 0, :], rays_expanded[:, :, 1, :]  # shape (nrays, ntriangles, 3)
    assert O.shape == (rays.shape[0], triangles.shape[0], 3)
    assert D.shape == (rays.shape[0], triangles.shape[0], 3)
                       
    A, B, C = triangles_expanded[:, :, 0, :], triangles_expanded[:, :, 1, :], triangles_expanded[:, :, 2, :]  # shape (nrays, ntriangles, 3)
    assert A.shape == (rays.shape[0], triangles.shape[0], 3)
    assert B.shape == (rays.shape[0], triangles.shape[0], 3)
    assert C.shape == (rays.shape[0], triangles.shape[0], 3)
    
    M = t.stack([-D, B-A, C-A], dim=-1)  # shape (nrays, ntriangles, 3, 3)
    assert M.shape == (rays.shape[0], triangles.shape[0], 3, 3)
    
    b = O - A  # shape (nrays, ntriangles, 3)
    assert b.shape == (rays.shape[0], triangles.shape[0], 3)
    
    # Handle singular matrices without crashing the entire batch
    dets = t.linalg.det(M)                      # (nrays, ntriangles)
    is_singular = t.abs(dets) <= 1e-8          # (nrays, ntriangles)
    M[is_singular] = t.eye(3)  # Replace singular matrices with identity

    try:
        x = t.linalg.solve(M, b)  # shape (nrays, ntriangles, 3)
        s,u,v = x[..., 0], x[..., 1], x[..., 2]
        s *= D[..., 0]
        intersects = (u >= 0) & (v >= 0) & ((u + v) <= 1) & (~is_singular)
        s[~intersects] = float("inf")
        return einops.reduce(s, "NR NT -> NR", "min")
    except t.linalg.LinAlgError:
        return t.full((rays.shape[0],), float("inf"))


num_pixels_y = 120
num_pixels_z = 120
y_limit = z_limit = 1

rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
rays[:, 0] = t.tensor([-2, 0.0, 0.0])
dists = raytrace_mesh(rays, triangles)
intersects = t.isfinite(dists).view(num_pixels_y, num_pixels_z)
dists_square = dists.view(num_pixels_y, num_pixels_z)
img = t.stack([intersects, dists_square], dim=0)

fig = px.imshow(img, facet_col=0, origin="lower", color_continuous_scale="magma", width=1000)
fig.update_layout(coloraxis_showscale=False)
for i, text in enumerate(["Intersects", "Distance"]):
    fig.layout.annotations[i]["text"] = text
fig.show()
# %%
