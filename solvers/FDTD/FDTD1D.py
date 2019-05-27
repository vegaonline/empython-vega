""" FDTD 1 D simulation
"""

from builtins import zip
from builtins import range
from builtins import object

__author__ = 'Abhijit Bhattacharyya'

import scipy as S
from scipy.linalg import inv
from bases.utility import snell, norm
from bases.constants import cLight, mu0

class FDTD1D(object):
    """ It uses 
    absorbing boundary condition, 
    normal and 
    lossy dielectric medium
    """
    def __init__(
            self, name='FDTD1D', ngridx = 200, centrePulseInc = 40.0, pulseSpread = 12.0,
            centreProbSpace = 100
    ):
        object.__init(self, name)
        object.ngridx = ngridx
        object.centreProbSpace = 0.5 * ngridx
        object.centrePulseInc = centrePulseInc
        object.pulseSpread = pulseSpread

    def computeFDTD1D(self, 
