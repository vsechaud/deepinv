import torch
import torch.nn as nn
from deepinv.loss.loss import Loss
import numpy as np

class EILoss(Loss):
    r"""
    Equivariant imaging self-supervised loss.

    Assumes that the set of signals is invariant to a group of transformations (rotations, translations, etc.)
    in order to learn from incomplete measurement data alone https://https://arxiv.org/pdf/2103.14756.pdf.

    The EI loss is defined as

    .. math::

        \| T_g \hat{x} - \inverse{\forw{T_g \hat{x}}}\|^2


    where :math:`\hat{x}=\inverse{y}` is a reconstructed signal and
    :math:`T_g` is a transformation sampled at random from a group :math:`g\sim\group`.

    By default, the error is computed using the MSE metric, however any other metric (e.g., :math:`\ell_1`)
    can be used as well.

    :param deepinv.Transform, torchvision.transforms transform: Transform to generate the virtually
        augmented measurement. It can be any torch-differentiable function (e.g., a ``torch.nn.Module``).
    :param torch.nn.Module metric: Metric used to compute the error between the reconstructed augmented measurement and the reference
        image.
    :param bool apply_noise: if ``True``, the augmented measurement is computed with the full sensing model
        :math:`\sensor{\noise{\forw{\hat{x}}}}` (i.e., noise and sensor model),
        otherwise is generated as :math:`\forw{\hat{x}}`.
    :param float weight: Weight of the loss.
    :param bool no_grad: if ``True``, the gradient does not propagate through :math:`T_g`. Default: ``True``.
    """

    def __init__(
        self,
        transform,
        metric=torch.nn.MSELoss(),
        apply_noise=True,
        weight=1.0,
        no_grad=True,
        use_mask=False,
        no_div=False
    ):
        super(EILoss, self).__init__()
        self.name = "ei"
        self.metric = metric
        self.weight = weight
        self.T = transform
        self.noise = apply_noise
        self.no_grad = no_grad
        self.use_mask = use_mask
        self.no_div = no_div

    def forward(self, x_net, y, physics, model, **kwargs):
        r"""
        Computes the EI loss

        :param torch.Tensor x_net: Reconstructed image :math:`\inverse{y}`.
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
        :param torch.nn.Module model: Reconstruction function.
        :return: (torch.Tensor) loss.
        """

        if self.use_mask:
            mask = (y<physics.vmin) | (y>physics.vmax)
            x_net[mask] = y[mask]

        if self.no_grad:
            with torch.no_grad():
                x2 = self.T(x_net)
        else:
            x2 = self.T(x_net)

        if self.no_div:
            B, C, H, W = x2.shape
            mask_compute = torch.count_nonzero(x2.view(B,-1)>physics.vmax, dim=1)/(C*H*W) < .7
            x2 = x2[mask_compute]

        if self.noise:
            y2 = physics(x2)
        else:
            y2 = physics.A(x2)

        # log_ei = torch.load("/projects/MultivariateDeepSynthesis/Victor/These/own-project/log_loss_ei/log_ei.pt")
        # rang = torch.nonzero(log_ei)[-1].item()
        # log_ei[rang+1] = x2.max().item()
        # torch.save(log_ei, "/projects/MultivariateDeepSynthesis/Victor/These/own-project/log_loss_ei/log_ei.pt")

        x3 = model(y2, physics)

        loss_ei = self.weight * self.metric(x3, x2)
        return loss_ei
