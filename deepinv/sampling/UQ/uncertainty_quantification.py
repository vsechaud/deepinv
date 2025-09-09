from torch import nn
import torch
import tqdm
import deepinv as dinv
from deepinv import Reconstructor


class Bootstrap(Reconstructor):
    def __init__(self, model, img_size, T=dinv.transform.Identity(),  method='parametric', MC=100, **kwargs):
        super(Bootstrap, self).__init__(**kwargs)
        self.model = model
        self.method = method
        self.T = T
        self.MC = MC
        if T.n_trans != MC:
            print(f"Warning: T.n_trans ({T.n_trans}) != MC ({MC}), n_trans set to MC")
        T.n_trans = MC
        self.img_size = img_size
        

    def forward(self, y, physics):

        x_net = self.model(y, physics)
        params = self.T.get_params(x_net)
        bootstrap_measurements = physics(self.T(x_net, **params).reshape(-1, *self.img_size))
        samples = self.model(bootstrap_measurements, physics)
        samples = self.T.inverse(samples, **params).reshape(-1, self.MC, *self.img_size)
        
        return samples


