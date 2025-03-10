from .consistency import (
    SupAdversarialGeneratorLoss,
    SupAdversarialDiscriminatorLoss,
    UnsupAdversarialGeneratorLoss,
    UnsupAdversarialDiscriminatorLoss,
    SplittingGeneratorLoss,
    SplittingDiscriminatorLoss,
)
from .uair import UAIRGeneratorLoss
from .base import DiscriminatorLoss, GeneratorLoss, DiscriminatorMetric
