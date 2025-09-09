from torch import nn
import numpy as np
from tqdm import tqdm
from deepinv.loss.metric import MSE
import matplotlib.pyplot as plt
from deepinv.transform import Identity

import torch
from deepinv.models import Reconstructor


class Bootstrap(Reconstructor):
    def __init__(self, model, img_size, T=Identity(),  method='parametric', MC=100, **kwargs):
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

class UQ(nn.modules):

    def __init__(self, img_size, dataloader, model, metric, **kwargs):
        super(UQ, self).__init__(**kwargs)
        self.dataloader = dataloader
        self.model = model
        self.MC = model.MC
        self.img_size = img_size
        self.metric = metric

    def compute_estimateMSE(self):
        N = len(self.dataloader.dataset)
        true_mse = np.zeros(N)
        estimated_mse = np.zeros((N, self.MC))
        k = 0

        for x, y in tqdm(self.dataloader, disable=True):
            x_net = self.model(y)
            B = x.shape[0]
            xhat = x_net.mean(1)
            true_mse_batch = MSE()(x, xhat)
            estimated_mse_batch = MSE()(x.repeat_interleave(self.MC, dim=0), x_net.reshape(-1, *self.img_size)).reshape(B, self.MC)  # (x.repeat_interleave(100, dim=0) == x.unsqueeze(1).expand(-1,100,-1,-1, -1).reshape(-1,3,28,28)).all()

            true_mse[k:k + x.shape[0]] = true_mse_batch
            estimated_mse[k:k + x.shape[0], :] = estimated_mse_batch
            k += x.shape[0]

        self.true_mse = true_mse
        self.estimated_mse = estimated_mse

        return true_mse, estimated_mse

    def plot_coverage(self):
        if not hasattr(self, 'true_mse') or not hasattr(self, 'estimated_mse'):
            true_mse, estimated_mse = self.compute_estimateMSE()
        else:
            true_mse = self.true_mse
            estimated_mse = self.estimated_mse
        N = len(true_mse)
        percentiles = np.linspace(0.1, .99, 100)
        distance = np.sort(estimated_mse, axis=0)
        empirical_coverage = np.zeros(len(percentiles))
        for j in range(len(percentiles)):
            success = 0
            for i in range(N):
                if true_mse[i] < distance[int(distance.shape[0] * percentiles[j]), i]:
                    success += 1

            empirical_coverage[j] = success / N

        empirical_coverage[-1] = 1.
        plt.figure()
        plt.plot(percentiles, empirical_coverage)
        plt.plot(percentiles, percentiles)
        plt.xlabel('Confidence level')
        plt.ylabel('Empirical coverage')
        plt.show()
