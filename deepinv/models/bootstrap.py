from .base import Reconstructor
from deepinv.transform import Identity

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
        MC : int, optional
            Number of Monte Carlo samples to generate. Default: ``100``.
        **kwargs : dict, optional
            Additional arguments passed to the parent class
            :class:`deepinv.models.Reconstructor`.

        Warning
        -------
        If ``T.n_trans`` is different from ``MC``, it will be overridden to match ``MC``.
    """

    def __init__(self, model, img_size, T=Identity(), MC=100, **kwargs):
        super(Bootstrap, self).__init__(**kwargs)
        self.model = model
        self.T = T
        self.MC = MC
        if T.n_trans != MC:
            print(f"Warning: T.n_trans ({T.n_trans}) != MC ({MC}), n_trans set to MC")
        T.n_trans = MC
        self.img_size = img_size

    def forward(self, y, physics):
        """
        Generate :math:`MC` bootstrap reconstructions from the measurement :math:`y`.

        :param torch.Tensor y: measurements.
        :param deepinv.physics.Physics physics: forward operator.
        :return: (:class:`torch.Tensor`): A tensor of shape ``(batch_size, MC, *img_size)`` containing the bootstrap samples.
        """
        x_net = self.model(y, physics)
        params = self.T.get_params(x_net)
        bootstrap_measurements = physics(self.T(x_net, **params).reshape(-1, *self.img_size))
        samples = self.model(bootstrap_measurements, physics)
        samples = self.T.inverse(samples, **params).reshape(-1, self.MC, *self.img_size)

        return samples
