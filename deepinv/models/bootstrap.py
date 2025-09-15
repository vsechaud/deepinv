from .base import Reconstructor
from deepinv.transform import Identity
import torch
class Bootstrap(Reconstructor):
    r"""
       Bootstrap reconstruction method for uncertainty quantification.

    This class generates multiple reconstructions of an image from the same
    measurement by combining a reconstruction network (:attr:`model`) with
    a stochastic transform (:attr:`T`). It produces :attr:`MC` samples that
    capture variability in the reconstruction.

        The class follows the :class:`deepinv.models.Reconstructor` interface.

        Parameters
        ----------
        model : deepinv.models.reconstructor, torch.nn.Module
            Base reconstruction network that maps measurements :math:`y` to reconstructions :math:`x`.
            Must be callable as ``model(y, physics)``.
        img_size : tuple of int
            Size of the reconstructed image, used for reshaping outputs.
        T : deepinv.transform.Transform, optional
            Stochastic transformation applied to the reconstruction. Default is
            :class:`deepinv.transform.Identity`.
        physics : deepinv.physics.Physics
            Forward operator modeling the measurement process.
            If you want to do parametric bootstrap, make sure your physics contains stochasticity.
        MC : int, optional
            Number of Monte Carlo samples to generate. Default: ``100``.
        **kwargs : dict, optional
            Additional arguments passed to the parent class
            :class:`deepinv.models.Reconstructor`.

        Warning
        -------
        If ``T.n_trans`` is different from ``MC``, it will be overridden to match ``MC``.
    """


    def __init__(self, img_size, model, physics, T=Identity(), MC=100, device='cpu', with_inverse=True):
        super(Bootstrap, self).__init__()
        self.model = model
        self.T = T
        self.MC = MC
        if T.n_trans != 1:
            print(f"Warning: T.n_trans set to 1")
            T.n_trans = 1
        self.img_size = img_size
        self.physics = physics
        self.device = device  
        self.with_inverse = with_inverse


    def forward(self, y, physics, **kwargs):
        """
        Generate :math:`MC` bootstrap reconstructions from the measurement :math:`y`.

        :param torch.Tensor y: measurements.
        :param deepinv.physics.Physics physics: forward operator.
        :return: (:class:`torch.Tensor`): A tensor of shape ``(batch_size, MC, *img_size)`` containing the bootstrap samples.
        """
        realized_samples = []
        samples = []
        self.model.eval()
        with torch.no_grad():
            x_net = self.model(y, self.physics)
            self.x_net = x_net.clone()
            for k in range(self.MC):
                params = self.T.get_params(x_net)
                realized_samples.append(self.T(x_net, **params))
                bootstrap_measurements = self.physics(realized_samples[k])
                samples.append(self.model(bootstrap_measurements, self.physics))
                if self.with_inverse:
                    samples[k] = self.T.inverse(samples[k], batchwise=False, **params)
        samples = torch.stack(samples, dim=1).reshape(-1, self.MC, *self.img_size)
        self.realized_samples = torch.stack(realized_samples, dim=1).reshape(-1, self.MC, *self.img_size)

        return samples

    def get_x_net(self):
        return  self.x_net
    
    def get_realized_samples(self):
        r"""
        Get the realized samples after a forward pass.

        :return: (:class:`torch.Tensor`): A tensor of shape ``(batch_size, MC, *img_size)`` containing the realized samples.
        """
        return self.realized_samples
