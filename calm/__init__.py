"""
CALM: Class-anchor-ALigned generative Modeling

A novel approach for multi-modal representation learning that leverages
class anchors and cross-modal probabilistic modeling.
"""

__version__ = '0.1.0'

from . import models
from . import data
from . import utils

__all__ = ['models', 'data', 'utils']
