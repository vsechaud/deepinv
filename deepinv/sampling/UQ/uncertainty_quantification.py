from torch import nn


class UQ(nn.modules):
    """
    Abstract base class for uncertainty quantification methods.
    """

    def __init__(self, **kwargs):
        super(UQ, self).__init__()
        self.name = self.__class__.__name__

    def sample(self, **kwargs):
        raise NotImplementedError("Subclasses must implement this method.")

    def coverage(self, **kwargs):
        raise NotImplementedError("Subclasses must implement this method.")


class Bootsrap(UQ):
    def __init__(self, **kwargs):
        super(Bootsrap, self).__init__(**kwargs)

    def sample(self, y, model, MC, statistics):
        pass

    def coverage(self, alpha, statstics):
        pass

