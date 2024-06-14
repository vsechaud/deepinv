import torch
from deepinv.loss.loss import Loss


class MCLoss(Loss):
    r"""
    Measurement consistency loss

    This loss enforces that the reconstructions are measurement-consistent, i.e., :math:`y=\forw{\inverse{y}}`.

    The measurement consistency loss is defined as

    .. math::

        \|y-\forw{\inverse{y}}\|^2

    where :math:`\inverse{y}` is the reconstructed signal and :math:`A` is a forward operator.

    By default, the error is computed using the MSE metric, however any other metric (e.g., :math:`\ell_1`)
    can be used as well.

    :param torch.nn.Module metric: metric used for computing data consistency, which is set as the mean squared error by default.
    """

    def __init__(self, metric=torch.nn.MSELoss()):
        super(MCLoss, self).__init__()
        self.name = "mc"
        self.metric = metric

    def forward(self, y, x_net, physics, **kwargs):
        r"""
        Computes the measurement splitting loss

        :param torch.Tensor y: measurements.
        :param torch.Tensor x_net: reconstructed image :math:`\inverse{y}`.
        :param deepinv.physics.Physics physics: forward operator associated with the measurements.
        :return: (torch.Tensor) loss.
        """
        return self.metric(physics.sensor(physics.A(x_net)), y)


class MCLoss3(Loss):
    def __init__(self, metric=torch.nn.MSELoss(), vmin=-1, vmax=1, pente="l2"):
        super(MCLoss3, self).__init__()
        self.name = "mc"
        self.metric = metric
        self.lamb1 = vmin
        self.lamb2 = vmax
        self.pente = pente

    def forward(self, y, x_net, physics, **kwargs):
        Ax_net = physics.A(x_net)
        epsi = 1E-5
        r = y-Ax_net
        r[(y>=self.lamb2-epsi) & (Ax_net>=self.lamb2)] = 0
        r[(y<=self.lamb1+epsi) & (Ax_net<=self.lamb1)] = 0

        return r.pow(2).mean() if self.pente=="l2" else r.abs().mean()