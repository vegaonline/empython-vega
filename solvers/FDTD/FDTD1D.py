""" FDTD 1 D simulation
"""

from builtins import zip
from builtins import range
from builtins import object

__author__ = 'Abhijit Bhattacharyya'

import scipy as S
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from scipy.linalg import inv
from bases.utility import snell, norm
from bases.constants import cLight, mu0, eps0

class FDTD1D(object):
    """ It uses 
    absorbing boundary condition, 
    normal and 
    lossy dielectric medium
    """
    def __init__(self, name='FDTD1D',  ex = S.zeros(200, dtype = float), hy = S.zeros(200, dtype = float), ngridx = 200, centrePulseInc = 40.0, pulseSpread = 12.0, centreProbSpace = 100, numSteps = 1, plotOK = 1):

        object.ex = ex
        object.hy = hy
        object.ngridx = ngridx
        object.centreProbSpace = 0.5 * ngridx
        object.centrePulseInc = centrePulseInc
        object.pulseSpread = pulseSpread
        object.numSteps = numSteps
        object.plotOK = plotOK

    # def computeFDTD1D(self, ex, hy, plotOK):
    def computeFDTD1D(self):
        """ FDTD1D solver
        INPUT :
        ngridx          = number cells along x direction as Electric field propagates along X
        centrePulseInc  = Centre of the incident pulse
        pulseSpread     = Width of the incident pulse
        centreProbSpace = Centre of the problem space
        numSteps        = total number of times the main loop to be executed
        plotOK          = 0 : not to plot || 1 : plot
        OUTPUT :
        
        """
        # self.ex = S.asarray(ex)
        # self.hy = S.asarray(hy)
        
        ex              = self.ex
        hy              = self.hy
        numSteps        = self.numSteps
        ngridx          = self.ngridx
        centrePulseInc  = self.centrePulseInc
        pulseSpread     = self.pulseSpread
        centreProbSpace = self.centreProbSpace
        plotOK          = self.plotOK
        

        tCount = 0         # keeps track of total number

        while (numSteps > 0):
            for nIter in xrange(0, numSteps):
                tCount += 1

                # MAIN FDTD 1D Loop
                for k in xrange(1, ngridx - 1):
                    ex[k] += 0.5 * (hy[k - 1] - hy[k])

                pulse = S.exp(-0.5 * ((centrePulseInc - tCount) / pulseSpread)**2)
                ex[centreProbpace] = pulse

                for k in xrange(0, ngridx - 2):
                    hy[k] += 0.5 * (ex[k] - ex[k + 1])

                # END of MAIN FDTD 1D Loop

                if plotOK == 1:
                    plot(ex, hy, ngridx) 
                

    def plot(self, ex, hy, ngridx):
        x = S.array(i for i in range(ngridx))
        ymin1 = S.min(ex)
        ymin2 = S.min(hy)
        ymax1 = S.max(ex)
        ymax2 = S.max(hy)
        yminimum = min(ymin1, ymin2)
        ymaximum = max(ymax1, ymax2)
        title1 = "EX field in FDTD 1D simulation."
        title2 = "HY field in FDTD 1D simulation."

        Writer = anim.Writers['ffmpeg']
        Writer = Writer(fps = 20, metadata = dict(artist = 'Abhijit'), bitrate = 1800)

        #fig = plt.figure(figsize = (10, 6))
        fig = plt.subplot(2, 1)
        plt.xlim(0, ngridx)
        # plt.ylim(S.min(ex)[0], S.max(ex)[0])
        plt.xlabel('FDTD Cells', fontsize = 20)
        plt.ylabel(title, fontsize = 20)
        plt.title(title1, fontsize = 20)
        plt.plot(x, ex, x, hy)
        fig.tight_layout()
        plt.show()
        

        
        
        
        
