# %%
import torch.nn as nn
import torch as t
import torch.nn.functional as F
from plotly_utils import line
from part2_cnns import tests
import numpy as np
import einops
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import modal
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
# %% [markdown]
## Setup Modal app

# %%
# Define your Modal App and Image
app = modal.App("02_cnns_exercises")
local_dir = str(Path(__file__).resolve().parents[1]) if modal.is_local() else "/tmp/unused"
chapter = "chapter0_fundamentals"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
remote_root =  root_dir / chapter / "exercises" if modal.is_local() else "/root/chapter0_fundamentals/exercises"
remote_workdir = f"{remote_root}/part2_cnns"
exercises_dir = remote_root
image = (
    modal.Image.debian_slim()
    .pip_install(
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
        "anywidget",
    )
    .env({
        "PYTHONPATH": str(remote_root),
    })
    .workdir(remote_workdir)
    .add_local_dir(local_dir, remote_path=remote_root)
)

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
# If this is CPU, we recommend figuring out how to get cuda access (or MPS if you're on a Mac).
print(device)
# %% [markdown]
### Exercise 1: Implement ReLU
# %%
class ReLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return x.clamp(min=0)

# %% [markdown]
### Exercise 2: Implement linear

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        """
        A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        sf = 1 / np.sqrt(in_features)

        weight = sf * (2 * t.rand(out_features, in_features) - 1)
        self.weight = nn.Parameter(weight)

        if bias:
            bias = sf * (2 * t.rand(out_features) - 1)
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (*, in_features)
        Return: shape (*, out_features)
        """
        x = einops.einsum(x, self.weight, "... in_feats, out_feats in_feats -> ... out_feats")
        if self.bias is not None:
            x += self.bias
        return x

    def extra_repr(self) -> str:
        return "Input features: {}, Output features: {}, Bias: {}".format(
            self.weight.shape[1], self.weight.shape[0], self.bias is not None)
# %% [markdown]
### Implement Flatten
# %%
class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: t.Tensor) -> t.Tensor:
        """
        Flatten out dimensions from start_dim to end_dim, inclusive of both.
        """
        shape = input.shape

        # Get start & end dims, handling negative indexing for end dim
        start_dim = self.start_dim
        end_dim = self.end_dim if self.end_dim >= 0 else len(shape) + self.end_dim

        # Get the shapes to the left / right of flattened dims, as well as size of flattened middle
        shape_left = shape[:start_dim]
        shape_right = shape[end_dim + 1 :]
        shape_middle = t.prod(t.tensor(shape[start_dim : end_dim + 1])).item()

        return t.reshape(input, shape_left + (shape_middle,) + shape_right)

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["start_dim", "end_dim"]])

# %% [markdown]
### Exercise 3: Implement the simple MLP

# %%
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten1 = Flatten()
        self.linear1 = Linear(28 * 28, 100)
        self.relu = ReLU()
        self.linear2 = Linear(100, 10)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.flatten1(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# %% [markdown]
## Training Neural Networks

# %%
MNIST_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(0.1307, 0.3081),
    ]
)

def get_mnist(trainset_size: int = 10_000, testset_size: int = 1_000) -> tuple[Subset, Subset]:
    """Returns a subset of MNIST training data."""

    # Get original datasets, which are downloaded to "./data" for future use
    mnist_trainset = datasets.MNIST(exercises_dir / "data", train=True, download=True, transform=MNIST_TRANSFORM)
    mnist_testset = datasets.MNIST(exercises_dir / "data", train=False, download=True, transform=MNIST_TRANSFORM)

    # # Return a subset of the original datasets
    mnist_trainset = Subset(mnist_trainset, indices=range(trainset_size))
    mnist_testset = Subset(mnist_testset, indices=range(testset_size))

    return mnist_trainset, mnist_testset


# mnist_trainset, mnist_testset = get_mnist()
# mnist_trainloader = DataLoader(mnist_trainset, batch_size=64, shuffle=True)
# mnist_testloader = DataLoader(mnist_testset, batch_size=64, shuffle=False)

# # Get the first batch of test data, by starting to iterate over `mnist_testloader`
# for img_batch, label_batch in mnist_testloader:
#     print(f"{img_batch.shape=}\n{label_batch.shape=}\n")
#     break

# # Get the first datapoint in the test set, by starting to iterate over `mnist_testset`
# for img, label in mnist_testset:
#     print(f"{img.shape=}\n{label=}\n")
#     break

# t.testing.assert_close(img, img_batch[0])
# assert label == label_batch[0].item()

@app.function(image=image, gpu="T4")
def test_mnist():
    import sys

    sys.path.append("/root/exercises")

    #from part2_cnns.exercises import get_mnist

    mnist_trainset, mnist_testset = get_mnist()
    mnist_trainloader = DataLoader(mnist_trainset, batch_size=64, shuffle=True)
    mnist_testloader = DataLoader(mnist_testset, batch_size=64, shuffle=False)

    # Get the first batch of test data, by starting to iterate over `mnist_testloader`
    for img_batch, label_batch in mnist_testloader:
        print(f"{img_batch.shape=}\n{label_batch.shape=}\n")
        break

    # Get the first datapoint in the test set, by starting to iterate over `mnist_testset`
    for img, label in mnist_testset:
        print(f"{img.shape=}\n{label=}\n")
        break

    t.testing.assert_close(img, img_batch[0])
    assert label == label_batch[0].item()            

# %% [markdown]
### Training Loop
@app.function(image=image, gpu="T4")
def training_loop():
    model = SimpleMLP().to(device)

    batch_size = 128
    epochs = 3

    mnist_trainset, _ = get_mnist()
    mnist_trainloader = DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)

    optimizer = t.optim.Adam(model.parameters(), lr=1e-3)
    loss_list = []

    for epoch in range(epochs):
        pbar = tqdm(mnist_trainloader)

        for imgs, labels in pbar:
            # Move data to device, perform forward pass
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)

            # Calculate loss, perform backward pass
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Update logs & progress bar
            loss_list.append(loss.item())
            pbar.set_postfix(epoch=f"{epoch + 1}/{epochs}", loss=f"{loss:.3f}")

    line(
        loss_list,
        x_max=epochs * len(mnist_trainset),
        labels={"x": "Examples seen", "y": "Cross entropy loss"},
        title="SimpleMLP training on MNIST",
        width=700,
    )
# %% [markdown]
### Exercise - Add a validation loop

# %%
@dataclass
class SimpleMLPTrainingArgs:
    """
    Defining this class implicitly creates an __init__ method, which sets arguments as below, e.g.
    self.batch_size=64. Any of these fields can also be overridden when you create an instance, e.g.
    SimpleMLPTrainingArgs(batch_size=128).
    """

    batch_size: int = 64
    epochs: int = 3
    learning_rate: float = 1e-3


def train(args: SimpleMLPTrainingArgs) -> tuple[list[float], SimpleMLP]:
    """
    Trains & returns the model, using training parameters from the `args` object. Returns the model,
    and loss list.
    """
    model = SimpleMLP().to(device)

    mnist_trainset, mnist_testset = get_mnist()
    mnist_trainloader = DataLoader(mnist_trainset, batch_size=args.batch_size, shuffle=True)
    mnist_testloader = DataLoader(mnist_testset, batch_size=args.batch_size, shuffle=False)

    optimizer = t.optim.Adam(model.parameters(), lr=args.learning_rate)

    loss_list = []
    accuracy_list = []
    accuracy = 0.0

    for epoch in range(args.epochs):
        # Training loop
        pbar = tqdm(mnist_trainloader)
        for imgs, labels in pbar:
            # Move data to device, perform forward pass
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)

            # Calculate loss, perform backward pass
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Update logs & progress bar
            loss_list.append(loss.item())
            pbar.set_postfix(epoch=f"{epoch + 1}/{args.epochs}", loss=f"{loss:.3f}")

        # Validation loop
        num_correct_classifications = 0
        for imgs, labels in mnist_testloader:
            # Move data to device, perform forward pass in inference mode
            imgs, labels = imgs.to(device), labels.to(device)
            with t.inference_mode():
                logits = model(imgs)

            # Compute num correct by comparing argmaxed logits to true labels
            predictions = t.argmax(logits, dim=1)
            num_correct_classifications += (predictions == labels).sum().item()

        # Compute & log total accuracy
        accuracy = num_correct_classifications / len(mnist_testset)
        accuracy_list.append(accuracy)
        
    line(
    y=[loss_list, [0.1] + accuracy_list],  # we start by assuming a uniform accuracy of 10%
    use_secondary_yaxis=True,
    x_max=args.epochs * len(mnist_trainset),
    labels={"x": "Num examples seen", "y1": "Cross entropy loss", "y2": "Test Accuracy"},
    title="SimpleMLP training on MNIST",
    width=800,
    )

    return loss_list, accuracy_list, model



# %% [markdown]
### Exercise - implement `conv2d`

# %%
class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        """
        Same as torch.nn.Conv2d with bias=False.

        Name your weight field `self.weight` for compatibility with the PyTorch version.

        We assume kernel is square, with height = width = `kernel_size`.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        N_in = self.in_channels * self.kernel_size * self.kernel_size
        sf = 1 / np.sqrt(N_in)
        self.weight = nn.Parameter(sf * (2 * t.rand(out_channels, in_channels, kernel_size, kernel_size) - 1))

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Apply the functional conv2d, which you can import."""
        return t.nn.functional.conv2d(x, self.weight, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        keys = ["in_channels", "out_channels", "kernel_size", "stride", "padding"]
        return ", ".join([f"{key}={getattr(self, key)}" for key in keys])
