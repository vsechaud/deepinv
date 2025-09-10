# %%

import torch
from torch.utils.data import DataLoader

import deepinv as dinv




torch.manual_seed(0)
device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# %%


from torchvision import transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),  # converts PIL Image to torch.Tensor
])
print("1")
# train_dataset = dinv.datasets.DIV2K(root="./data/DIV2K", download=False, mode='train', transform=transform)
test_dataset = dinv.datasets.DIV2K(root="./data/DIV2K", download=False, mode='val', transform=transform)

img_size = (3, 256, 256)
sigma = .05

physics = dinv.physics.Inpainting(
    img_size=img_size,
    mask=0.5,
    device=device,
    noise_model=dinv.physics.GaussianNoise(sigma=sigma)
)

num_workers = 4 if torch.cuda.is_available() else 0
print("2")
deepinv_datasets_path = dinv.datasets.generate_dataset(
    train_dataset=None,
    test_dataset=test_dataset,
    physics=physics,
    device=device,
    save_dir="data/Inpainting_div2k/",
    train_datapoints=100,
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



model = dinv.models.RAM(in_channels=(1, 2, 3), device=device, pretrained=True)
model.eval()
# %%

from deepinv.sampling.uncertainty_quantification import UQ
from deepinv.models.bootstrap import Bootstrap

bootstrap_model = Bootstrap(model=model, img_size=img_size, physics=physics, T=dinv.transform.Shift(), MC=2, device=device)

# %%
print("3")
uq = UQ(img_size=img_size, dataloader=dataloader, model=bootstrap_model)
uq.compute_estimateMSE()
uq.plot_coverage()