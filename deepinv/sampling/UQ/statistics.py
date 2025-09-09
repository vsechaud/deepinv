import torch
from torch import nn
from deepinv.loss.metric.functional import cal_mse


class BaseStatistic(nn.Module):
    """
    Abstract base class for statistical computations in uncertainty quantification.
    """

    def __init__(self, **kwargs):
        super(BaseStatistic, self).__init__()
        self.name = self.__class__.__name__

    def forward(self, **kwargs):
        raise NotImplementedError("Subclasses must implement this method.")


class MeanStatistic(BaseStatistic):
    """
    Computes the mean of the samples.
    """

    def __init__(self, **kwargs):
        super(MeanStatistic, self).__init__(**kwargs)

    def forward(self, samples):
        return samples.mean(dim=1)  # Mean over MC samples

class VarianceStatistic(BaseStatistic):
    """
    Computes the variance of the samples.
    """
    def __init__(self, **kwargs):
        super(VarianceStatistic, self).__init__(**kwargs)

    def forward(self, samples):
        return samples.var(dim=1)  # Variance over MC samples

class StdStatistic(BaseStatistic):
    """
    Computes the standard deviation of the samples.
    """
    def __init__(self, **kwargs):
        super(StdStatistic, self).__init__(**kwargs)

    def forward(self, samples):
        return samples.std(dim=1)  # Standard deviation over MC samples


class QuantileStatistic(BaseStatistic):


    def __init__(self, quantile: float = 0.9, **kwargs):
        super(QuantileStatistic, self).__init__(**kwargs)
        self.quantile = quantile
        self.name = self.__class__.__name__ + "_q_" + str(quantile)

    def forward(self, tensor_list):

        return (
            torch.quantile(tensor_list, self.quantile, dim=0)
            if tensor_list.numel() > 0
            else torch.tensor(0.0)
        )


class MSEStatistic(BaseStatistic):

    def __init__(self, **kwargs):
        super(MSEStatistic, self).__init__( **kwargs)

    def forward(self, a: torch.Tensor, b: torch.Tensor, **kwargs):
        return cal_mse(a, b)

class TrueMSEStatistic(BaseStatistic):

    def __init__(self, **kwargs):
        super(TrueMSEStatistic, self).__init__(**kwargs)

    def forward(self, x: torch.Tensor, y: torch.Tensor, **kwargs):
        xhat = MeanStatistic()

        return cal_mse(x, y)