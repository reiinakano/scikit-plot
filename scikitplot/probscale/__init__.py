"""
Real probability scales for matplotlib.
"""
from .viz import *

from .probscale import ProbScale
from matplotlib import scale
scale.register_scale(ProbScale); del scale;


__version__ = "0.2.6dev0"
__author__ = "Paul Hobson (Herrera Environmental Consultants)"
__author_email__ = "phobson@herrerainc.com"
