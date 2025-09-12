# %%

import torch
from torch.utils.data import DataLoader

import deepinv as dinv
from deepinv.sampling.uncertainty_quantification import UQ
from deepinv.models.bootstrap import Bootstrap
from deepinv.utils import plot


torch.manual_seed(0)
device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
from torch import nn



# %%

img_size = (3, 128, 128)
sigma = .05

physics = dinv.physics.Inpainting(
    img_size=img_size,
    mask=0.5,
    device=device,
    noise_model=dinv.physics.GaussianNoise(sigma=sigma)
)

num_workers = 4 if torch.cuda.is_available() else 0

# from torchvision import transforms
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize((128, 128)),  # converts PIL Image to torch.Tensor
# ])
#
# train_dataset = dinv.datasets.DIV2K(root="./data/DIV2K", download=False, mode='train', transform=transform)
# test_dataset = dinv.datasets.DIV2K(root="./data/DIV2K", download=False, mode='val', transform=transform)
# test_dataset = torch.utils.data.Subset(test_dataset, range(60))
#
#

#
# deepinv_datasets_path = dinv.datasets.generate_dataset(
#     train_dataset=train_dataset,
#     test_dataset=test_dataset,
#     physics=physics,
#     device=device,
#     save_dir="data/Inpainting_div2k/",
#     train_datapoints=800,
#     test_datapoints=100,
#     batch_size=100,
#     num_workers=num_workers,
#     dataset_filename="dataset",
#     overwrite_existing=True,
# )

deepinv_datasets_path = f"data/Inpainting_div2k//dataset0.h5"

train_dataset = dinv.datasets.HDF5Dataset(path=deepinv_datasets_path, train=True)
test_dataset = dinv.datasets.HDF5Dataset(path=deepinv_datasets_path, train=False)
test_dataset = torch.utils.data.Subset(test_dataset, range(60))
train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=False, num_workers=num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=20, shuffle=False, num_workers=num_workers)


# %%
# model = dinv.models.RAM(in_channels=(1, 2, 3), device=device, pretrained=True)
# model.eval()

backbone_net = dinv.models.DnCNN(in_channels=3, out_channels=3, depth=20, bias=True, nf=64, padding_mode='zeros', device=device)
model = dinv.models.ArtifactRemoval(backbone_net=backbone_net, mode="adjoint", device=device)
trainer = dinv.Trainer(
    epochs=5,
    model=model,
    physics=physics,
    losses=dinv.loss.SupLoss(),
    device=device,
    train_dataloader=train_dataloader,
    eval_dataloader=test_dataloader,
    optimizer=torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-8),
    save_path=None,
    eval_interval=5,
)
model = trainer.train()

# %%
x,y = next(iter(test_dataloader))
x = x.to(device)
y = y.to(device)
with torch.no_grad():
    x_net = model(y, physics=physics)


# %%
bootstrap_model = Bootstrap(model=model, img_size=img_size, physics=physics, T=dinv.transform.Shift(), MC=50, device=device)

xhat = bootstrap_model(y, physics)
plot([x, y, x_net, xhat.mean(dim=1), xhat[:,0], xhat[:,1]])
uq = UQ(img_size=img_size, dataloader=test_dataloader, model=bootstrap_model)
# true, esti = uq.compute_estimateMSE()
# uq.plot_coverage()