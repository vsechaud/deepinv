import torch.nn
from torch import nn
import numpy as np
from tqdm import tqdm
from deepinv.loss.metric import MSE
import matplotlib.pyplot as plt

class UQ(nn.Module):
    r"""
        Uncertainty quantification (UQ) class for evaluating reconstruction models.

        This class estimates and evaluates the uncertainty of a reconstruction model
        (typically a bootstrap-based model) by comparing the true mean squared error (MSE)
        with estimated MSEs computed from multiple Monte Carlo (MC) samples.

        It provides methods to compute error estimates and to visualize the empirical
        coverage of uncertainty intervals.

        Parameters
        ----------
        img_size : tuple of int
            Size of the reconstructed image.
        dataloader : torch.utils.data.DataLoader
            Dataloader providing ground-truth images and measurements.
        model : nn.Module
            Reconstruction model that outputs ``MC`` stochastic reconstructions for each images.
            Must have an attribute ``MC`` (number of samples).
        metric : callable
            Metric function to evaluate reconstructions (e.g., :class:`deepinv.loss.metric.MSE`).
        **kwargs : dict, optional
            Additional arguments passed to :class:`torch.nn.Module`.

        Attributes
        ----------
        true_mse : np.ndarray
            Array of ground-truth MSE values, shape ``(N,)`` with ``N`` number of samples.
        estimated_mse : np.ndarray
            Array of estimated MSE values, shape ``(N, MC)``.

        """
    def __init__(self, img_size, dataloader, model, metric=MSE(), **kwargs):
        super(UQ, self).__init__(**kwargs)
        self.dataloader = dataloader
        self.model = model
        self.MC = model.MC
        self.img_size = img_size
        self.metric = metric
        self.device = model.device

    def compute_estimateMSE(self):
        r"""
        Compute ground-truth and estimated MSE for the dataset.

        For each sample in the dataloader:

        - Compute the mean reconstruction.
        - Evaluate the ground-truth MSE between the reconstruction and the true image.
        - Estimate MSE for each Monte Carlo sample.

        Returns
        -------
        tuple of (np.ndarray, np.ndarray)
            * true_mse : shape ``(N,)`` with ground-truth MSE values.
            * estimated_mse : shape ``(N, MC)`` with estimated MSE values.
        """
        N = len(self.dataloader.dataset)
        true_mse = np.zeros(N)
        estimated_mse = np.zeros((N, self.MC))
        k = 0

        for x, y in tqdm(self.dataloader, disable=True):
            x = x.to(self.device)
            y = y.to(self.device)
            x_hat = self.model(y, physics=None)
            B = x.shape[0]
            x_net = self.model.get_x_net()
            true_mse_batch = self.metric(x, x_net).cpu()
            estimated_mse_batch = self.metric(x_net.repeat_interleave(self.MC, dim=0), x_hat.reshape(-1, *self.img_size)).reshape(B, self.MC).cpu() #faux

            true_mse[k:k + x.shape[0]] = true_mse_batch
            estimated_mse[k:k + x.shape[0], :] = estimated_mse_batch
            k += x.shape[0]

        self.true_mse = true_mse
        self.estimated_mse = estimated_mse

        return true_mse, estimated_mse

    def plot_coverage(self):
        r"""
        Plot empirical coverage of uncertainty intervals.

        This method compares the true MSE with estimated MSE quantiles
        to assess the reliability of the uncertainty estimates.

        It produces a coverage plot where the empirical coverage is compared
        against the confidence levels.

        Returns
        -------
        None
            Displays a matplotlib figure.
        """
        if not hasattr(self, 'true_mse') or not hasattr(self, 'estimated_mse'):
            true_mse, estimated_mse = self.compute_estimateMSE()
        else:
            true_mse = self.true_mse
            estimated_mse = self.estimated_mse
        N = len(true_mse)
        percentiles = np.linspace(0.1, .99, 100)
        distance = np.sort(estimated_mse, axis=1)
        empirical_coverage = np.zeros(len(percentiles))
        for j in range(len(percentiles)):
            success = 0
            for i in range(N):
                if true_mse[i] < distance[i, int(distance.shape[1] * percentiles[j])]:
                    success += 1

            empirical_coverage[j] = success / N

        # empirical_coverage[-1] = 1.
        plt.figure()
        plt.plot(percentiles, empirical_coverage)
        plt.plot(percentiles, percentiles)
        plt.xlabel('Confidence level')
        plt.ylabel('Empirical coverage')
        plt.show()
