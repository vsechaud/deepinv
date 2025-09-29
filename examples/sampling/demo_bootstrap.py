# %%
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

import deepinv as dinv
from deepinv.sampling.uncertainty_quantification import UQ
from deepinv.models.bootstrap import Bootstrap

from pathlib import Path

torch.manual_seed(0)
device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# %% Load base image datasets.
# In this example, we use the MNIST dataset for training and testing.
train_dataset = datasets.MNIST(
    root="./data/MNIST", download=False, train=True, transform=transforms.ToTensor()
)
test_dataset = datasets.MNIST(
    root="./data/MNIST", download=False, train=False, transform=transforms.ToTensor()
)

# %% Define physics
# We use A\inR_mxn a random Gaussian matrix as the forward operator, m=256 and n=28*28.
img_size = (1, 28, 28)
sigma = 0.05
physics = dinv.physics.CompressedSensing(
    m=256,
    img_size=img_size,
    device=device,
    noise_model=dinv.physics.GaussianNoise(sigma=sigma),
)

num_workers = 4 if torch.cuda.is_available() else 0

# %% Generate dataset and dataloaders
# We use 6 000 training images and 384 test images.
measurement_dir = Path(".") / "measurements" / "cs_mnist"

deepinv_datasets_path = dinv.datasets.generate_dataset(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    physics=physics,
    device=device,
    save_dir=measurement_dir,
    train_datapoints=6_000,
    test_datapoints=384,
    batch_size=500,
    num_workers=num_workers,
    overwrite_existing=False,
)

train_dataset = dinv.datasets.HDF5Dataset(path=deepinv_datasets_path, train=True)
test_dataset = dinv.datasets.HDF5Dataset(path=deepinv_datasets_path, train=False)

train_dataloader = DataLoader(
    train_dataset, batch_size=20, shuffle=False, num_workers=num_workers
)
test_dataloader = DataLoader(
    test_dataset, batch_size=10, shuffle=False, num_workers=num_workers
)


# %% Set up the reconstruction network
# As a reconstruction network, we use a simple artifact removal network based on a U-Net.

backbone_net = dinv.models.UNet(
    in_channels=1,
    out_channels=1,
    residual=True,
    circular_padding=False,
    cat=True,
    bias=True,
    batch_norm=False,
    scales=4,
).to(device)
model = dinv.models.ArtifactRemoval(
    backbone_net=backbone_net, mode="adjoint", device=device
)
model_path = "/projects/MultivariateDeepSynthesis/phase_retrieval/examples/sampling/25-09-26-09:49:52/ckp_best.pth.tar"
checkpoint = torch.load(model_path, weights_only=False, map_location=device)
model.load_state_dict(checkpoint["state_dict"])

# %% Train and test network
trainer = dinv.Trainer(
    epochs=0,
    model=model,
    physics=physics,
    losses=dinv.loss.SupLoss(),
    device=device,
    train_dataloader=train_dataloader,
    eval_dataloader=test_dataloader,
    eval_interval=2,
    optimizer=torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-8),
    # save_path="",
    ckp_interval=2,
)
model = trainer.train()


# %% Evaluate uncertainty quantification
# We use the Bootstrap model to generate MC samples, compute the true and estimated MSEs,
# and plot the empirical coverage of the uncertainty intervals.

T = dinv.transform.Shift(shift_max=2 / 28)
bootstrap_model = Bootstrap(
    model=model, img_size=img_size, physics=physics, T=T, MC=100, device=device
)
uq = UQ(img_size=img_size, dataloader=test_dataloader, model=bootstrap_model)
uq.plot_coverage()
