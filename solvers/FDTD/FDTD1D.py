""" FDTD 1 D simulation
"""

from builtins import zip
from builtins import range
from builtins import object

__author__ = 'Abhijit Bhattacharyya'

import scipy as S
from scipy.linalg import inv
from bases.utility import snell, norm
from bases.constants import cLight, mu0, eps0

class FDTD1D(object):
    """ It uses 
    absorbing boundary condition, 
    normal and 
    lossy dielectric medium
    """
    def __init__(self, name='FDTD1D', ngridx = 200, centrePulseInc = 40.0, pulseSpread = 12.0, centreProbSpace = 100, numSteps = 1):
        object.__init(self, name)
        object.ngridx = ngridx
        object.centreProbSpace = 0.5 * ngridx
        object.centrePulseInc = centrePulseInc
        object.pulseSpread = pulseSpread
        object.numSteps = numSteps

    def computeFDTD1D(self, ex, hy):
        """ FDTD1D solver
        INPUT :
        ngridx          = number cells along x direction as Electric field propagates along X
        centrePulseInc  = Centre of the incident pulse
        pulseSpread     = Width of the incident pulse
        centreProbSpace = Centre of the problem space
        numSteps        = total number of times the main loop to be executed
        OUTPUT :
        
        """
        self.ex = S.asarray(ex)
        self.hy = S.asarray(hy)

        tCount = 0         # keeps track of total number

        while (numSteps > 0):
            for nIter in xrange(0, numSteps):
                tCount += 1

                # MAIN FDTD 1D Loop
                for k in xrange(0, ngridx):
                    ex
                

        
        
