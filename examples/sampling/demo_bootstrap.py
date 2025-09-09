# %%
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import os
import deepinv as dinv
from huggingface_hub import hf_hub_download



torch.manual_seed(0)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
train_dataset = dinv.datasets.DIV2K(root="./data/DIV2K", download=False, mode='train')
test_dataset = dinv.datasets.DIV2K(root="./data/DIV2K", download=False, mode='val')


print("Loaded dataset with", len(train_dataset), "images.")



print(train_dataset)

from huggingface_hub import hf_hub_download
import torch

# Download the HDF5 dataset
dataset_path = hf_hub_download(
    repo_id="jtachella/equivariant_bootstrap",
    filename="Inpainting_div2k/dinv_dataset0.h5"
)
print("Dataset path:", dataset_path)

# Download the PyTorch checkpoint
physics_path = hf_hub_download(
    repo_id="jtachella/equivariant_bootstrap",
    filename="Inpainting_div2k/physics0.pt"
)
print("Physics path:", physics_path)

# Load the physics model
physics_state = torch.load(physics_path, map_location="cpu")  # or device
mask = physics_state['mask']

# %%
config = {
    "dataset_name": "DIV2K",  #_shift_invariant
    "operation": "inpainting_rows",
    "noise":None,
    "oversampling_ratio": 0.5,
    "n_images": 800,
}

from torchvision import transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),  # converts PIL Image to torch.Tensor
])

train_dataset = dinv.datasets.DIV2K(root="./data/DIV2K", download=False, mode='train', transform=transform)
test_dataset = dinv.datasets.DIV2K(root="./data/DIV2K", download=False, mode='val', transform=transform)

img_size = (3, 256, 256)
sigma = .05

physics = dinv.physics.Inpainting(
    img_size=img_size,
    mask=mask,
    device=device,
    noise_model=dinv.physics.GaussianNoise(sigma=sigma)
)

num_workers = 4 if torch.cuda.is_available() else 0

deepinv_datasets_path = dinv.datasets.generate_dataset(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    physics=physics,
    device=device,
    save_dir="data/Inpainting_div2k/",
    train_datapoints=config["n_images"],
    test_datapoints=100,
    batch_size=100,
    num_workers=num_workers,
    dataset_filename="dataset",
    overwrite_existing=False,
)

# deepinv_datasets_path = f"ckpts/inpainting/DIV2K_oversampling_ratio={config['oversampling_ratio']}_n_images=800_/dataset0.h5"

# train_dataset = dinv.datasets.HDF5Dataset(path=deepinv_datasets_path, train=True)
test_dataset = dinv.datasets.HDF5Dataset(path=deepinv_datasets_path, train=False)
dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=num_workers)



# %%
# choose a reconstruction architecture
backbone = dinv.models.UNet(in_channels=3, out_channels=3, scales=4,
                            bias=False, batch_norm=False).to(device)

unrolled_iter = 3
model = dinv.unfolded.unfolded_builder(
    "HQS",
    params_algo={"stepsize": [1.0] * unrolled_iter, "g_param": [0.01] * unrolled_iter, "lambda": 1.0},
    trainable_params=["lambda", "stepsize", "g_param"],
    data_fidelity=dinv.optim.L2(),
    max_iter=unrolled_iter,
    prior=dinv.optim.PnP(denoiser=backbone),
    verbose=False,
)
losses = [dinv.loss.SupLoss()]

# ckp_path =  hf_hub_download(
#     repo_id="jtachella/equivariant_bootstrap",
#     filename="Inpainting_div2k/sup/ckp.pth.tar"
# )


# def download_model(path):
#     if not os.path.exists(path):
#         save_dir2 = './datasets/'
#         hf_hub_download(repo_id="jtachella/equivariant_bootstrap", filename=path,
#                         cache_dir=save_dir2, local_dir=save_dir2)
        

# ckp_path = '../deepinv/sampling/UQ/ckp.pth.tar'
# ckp_path = 'equivariant_bootstrap/Inpainting_div2k/sup/ckp.pth.tar'

# # hf_hub_download(repo_id="jtachella/equivariant_bootstrap", filename="Inpainting_div2k/sup/ckp.pth.tar",
# #                         cache_dir="./data/", local_dir="./data/", local_dir_use_symlinks=False)


# # checkpoint = torch.load(
# #                 ckp_path, map_location=device, weights_only=False
# #             )
# # model.load_state_dict(torch.load(checkpoint["state_dict"], map_location=device)['state_dict'])
# # download_model(ckp_path)
# model.load_state_dict(torch.load(ckp_path, map_location=device)['state_dict'])


model = dinv.models.RAM(in_channels=(1, 2, 3), device=device, pretrained=True)
model.eval()
# %%

import deepinv


import importlib
importlib.reload(deepinv.models.bootstrap)
importlib.reload(deepinv.sampling.uncertainty_quantification)
from deepinv.sampling.uncertainty_quantification import UQ
from deepinv.models.bootstrap import Bootstrap
bootstrap_model = Bootstrap(model=model, img_size=img_size, physics=physics, T=dinv.transform.Shift(n_trans=100), MC=100).to(device)

# %%

uq = UQ(img_size=img_size, dataloader=dataloader, model=bootstrap_model, metric=None) # METRIC TO REMOVE !!!!!
uq.compute_estimateMSE()
uq.plot_coverage()