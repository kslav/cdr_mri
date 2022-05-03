"""
Deep inverse problems in Python

forwards submodule
A Forward object takes an input image x and returns measurements y
"""

from .mcmri.mcmri import MultiChannelMRI
from .mcmri.mcmri import maps_forw
from .mcmri.mcmri import maps_adj
from .mcmri.dataset import MultiChannelMRIDataset
