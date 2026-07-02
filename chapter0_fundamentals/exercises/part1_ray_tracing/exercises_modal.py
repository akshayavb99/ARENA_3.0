# %%
import modal
import torch as t
from pathlib import Path

# Define your Modal App and Image
app = modal.App("01_ray_tracing_exercises")
local_dir = str(Path(__file__).resolve().parents[1]) if modal.is_local() else "/tmp/unused"
remote_root = "/root/chapter0_fundamentals/exercises"
remote_workdir = f"{remote_root}/part1_ray_tracing"
image = modal.Image.debian_slim(
).pip_install(
    "numpy", 
    "torch",
    "plotly",
    "wandb",
    "einops",
    "torchinfo",
    "ipython",
    "rich",
    "jaxtyping",
    "pillow",
    "torchvision",
    "pandas",
    "scikit-learn",
    "ipywidgets",
    "tqdm",
    "anywidget"
).workdir(remote_workdir).add_local_dir(local_dir, remote_path=remote_root)


@app.function(gpu="T4:1", image=image)
def test_modal():
    from exercises import make_rays_2d    
    print("CUDA available:", t.cuda.is_available())
    if t.cuda.is_available():
        print("GPU:", t.cuda.get_device_name(0))
    
    rays = make_rays_2d(10, 10, 0.3, 0.3)
    print("Rays:", rays.shape)
    #print("Triangles:", triangles.shape)
    
    return "Modal function executed successfully!"

print(test_modal)
# %%
if __name__ == "__main__":
    with modal.enable_output():
        with app.run():
            print("Starting Modal function test run...")
            result = test_modal.remote()
            print(result)
# %%
from pathlib import Path
import os

print("CWD:", os.getcwd())
print("HOME exists:", Path("/home/aksha").exists())
print("ARENA exists:", Path("/home/aksha/ARENA_3.0").exists())
print("ARENA listing:", list(Path("/home/aksha").glob("ARENA*")))

# %%
