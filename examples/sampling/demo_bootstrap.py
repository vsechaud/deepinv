# %%

import torch
from torch.utils.data import DataLoader

import deepinv as dinv
from deepinv.sampling.uncertainty_quantification import UQ
from deepinv.models.bootstrap import Bootstrap
from deepinv.utils import plot


torch.manual_seed(0)
device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# %%


from torchvision import transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128)),  # converts PIL Image to torch.Tensor
])

# train_dataset = dinv.datasets.DIV2K(root="./data/DIV2K", download=False, mode='train', transform=transform)
test_dataset = dinv.datasets.DIV2K(root="./data/DIV2K", download=False, mode='val', transform=transform)

img_size = (3, 128, 128)
sigma = .05

physics = dinv.physics.Inpainting(
    img_size=img_size,
    mask=0.5,
    device=device,
    noise_model=dinv.physics.GaussianNoise(sigma=sigma)
)

num_workers = 4 if torch.cuda.is_available() else 0

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
test_dataset = torch.utils.data.Subset(test_dataset, range(60))
dataloader = DataLoader(test_dataset, batch_size=20, shuffle=False, num_workers=num_workers)



model = dinv.models.RAM(in_channels=(1, 2, 3), device=device, pretrained=True)
model.eval()
# %%

x,y = next(iter(dataloader))
x = x.to(device)
y = y.to(device)
with torch.no_grad():
    x_net = model(y, physics=physics)


# %%
bootstrap_model = Bootstrap(model=model, img_size=img_size, physics=physics, T=dinv.transform.Shift(), MC=50, device=device)

xhat = bootstrap_model(y, physics)
plot([x, y, x_net, xhat.mean(dim=1), xhat[:,2]])
uq = UQ(img_size=img_size, dataloader=dataloader, model=bootstrap_model)
# true, esti = uq.compute_estimateMSE()
# uq.plot_coverage()