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
    def __init__(self, name='FDTD1D',  ex = S.zeros(100, dtype = float), hy = S.zeros(100, dtype = float), ngridx = 100, centrePulseInc = 10.0, pulseSpread = 5.0, centreProbSpace = 50, numSteps = 1, plotOK = 1):

        self.ex = ex
        self.hy = hy
        self.ngridx = ngridx
        self.centreProbSpace = int(ngridx / 2)
        self.centrePulseInc = centrePulseInc
        self.pulseSpread = pulseSpread
        self.numSteps = numSteps
        self.plotOK = plotOK
        
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
        centrePulseInc  = int(self.centrePulseInc)
        pulseSpread     = self.pulseSpread
        centreProbSpace = self.centreProbSpace
        plotOK          = self.plotOK
        
        tCount = 0         # keeps track of total number

        if (numSteps > 0):
            for nIter in range(0, numSteps):
                tCount += 1

                # MAIN FDTD 1D Loop
                for k in range(1, ngridx - 1):
                    ex[k] += 0.5 * (hy[k - 1] - hy[k])

                pulse = S.exp(-0.5 * ((centrePulseInc - tCount) / pulseSpread)**2)
                ex[centreProbSpace] = pulse

                for k in range(0, ngridx - 2):
                    hy[k] += 0.5 * (ex[k] - ex[k + 1])

                # END of MAIN FDTD 1D Loop

        return self
        

    #def plot(self, ex, hy, ngridx):
    def plot(self):
        ex     = self.ex
        hy     = self.hy
        ngridx = self.ngridx
        x = S.array(i for i in range(ngridx))
        ymin1 = S.amin(ex)
        ymin2 = S.amin(hy)
        ymax1 = S.amax(ex)
        ymax2 = S.amax(hy)
        yminimum = min(ymin1, ymin2)
        ymaximum = max(ymax1, ymax2)
        title1 = "EX field in FDTD 1D simulation."
        title2 = "HY field in FDTD 1D simulation."

        # Writer = anim.Writers['ffmpeg']
        # Writer = Writer(fps = 20, metadata = dict(artist = 'Abhijit'), bitrate = 1800)

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

        
        
        
        
