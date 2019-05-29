""" FDTD 1 D simulation
"""

from __future__ import print_function
from __future__ import absolute_import

from builtins import zip
from builtins import range
from builtins import object

__author__ = 'Abhijit Bhattacharyya'

import scipy as S
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.lines as Line2D

from scipy.linalg import inv
from bases.utility import snell, norm
from bases.constants import cLight, mu0, eps0

class FDTD1D(object):
    """ It uses 
    absorbing boundary condition, 
    normal and 
    lossy dielectric medium
    """
    def __init__(self, name='FDTD1D',  ex = S.zeros(100, dtype = float), hy = S.zeros(100, dtype = float), ngridx = 100, distTravel = 100.0, signalFreq = 50.0, centrePulseInc = 10.0, pulseSpread = 5.0, centreProbSpace = 50, numSteps = 1, epsRmedium = 1.0, sigmaMedium = 1.0, plotOK = 1, isABC = 1, isLossy = 0):

        self.ex = ex
        self.hy = hy
        self.ngridx = ngridx
        self.distTravel = distTravel
        self.signalFreq = signalFreq * 1.0e6  # converted to Hertz
        self.centreProbSpace = centreProbSpace
        self.centrePulseInc = centrePulseInc
        self.pulseSpread = pulseSpread
        self.numSteps = numSteps
        self.epsRmedium = epsRmedium
        self.sigmaMedium = sigmaMedium
        self.plotOK = plotOK
        self.isABC = isABC       # is absorbing boundary condition to avoid reflection from boundary?
        self.isLossy = isLossy   # is the dielectric medium a lossy medium?
        

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
        distTravel      = self.distTravel
        signalFreq      = self.signalFreq
        centrePulseInc  = int(self.centrePulseInc)
        pulseSpread     = self.pulseSpread
        centreProbSpace = self.centreProbSpace
        epsRmedium      = self.epsRmedium
        sigmaMedium     = self.sigmaMedium
        plotOK          = self.plotOK
        isABC           = self.isABC
        isLossy         = self.isLossy
        
        tCount = 0         # keeps track of total number
        delX = distTravel / (1.0 * ngridx)
        lambda0 = cLight / signalFreq
        delT = delX / (2.0 * cLight)
        ca = S.zeros(ngridx, dtype = float)
        cb = S.zeros(ngridx, dtype = float)
        lossStart = 100.0

        if isABC == 1:
            ex_low_m1 = 0.0
            ex_low_m2 = 0.0
            ex_hi_m1 = 0.0
            ex_hi_m2 = 0.0

        if isLossy == 1:
            for k in range(1, ngridx):
                ca[k] = 1.0
                cb[k] = 0.5
            eaf = delT * sigmaMedium / (2.0 * eps0 * epsRmedium)
            for k in range(lossStart, ngridx):
                ca[k] = (1.0 - eaf) / (1.0 + eaf)
                cb[k] = 0.5 / (epsRmedium * (1.0 + eaf))
                
            
        
        if (numSteps > 0):
            for nIter in range(0, numSteps):
                tCount += 1

                # MAIN FDTD 1D Loop
                for k in range(1, ngridx - 1):

                    if isLossy == 0:
                        ex[k] += 0.5 * (hy[k - 1] - hy[k])    # this is for non lossy medium
                    else:
                        ex[k] = ca[k] * ex[k] + cb[k] *  (hy[k - 1] - hy[k])    # this is for lossy medium
                        
                    # print('k = ', k, 'Ex: ', ex[k], '  Hy: ', hy[k])

                pulse = S.exp(-0.5 * ((centrePulseInc - tCount) / pulseSpread)**2)   # for Gaussian
                ex[centreProbSpace] = pulse

                if isABC == 1:
                    ex[0] = ex_low_m2
                    ex_low_m2 = ex_low_m1
                    ex_low_m1 = ex[1]

                    ex[ngridx - 1] = ex_hi_m2
                    ex_hi_m2 = ex_hi_m1
                    ex_hi_m1 = ex[ngridx - 2]
                

                for k in range(0, ngridx - 2):
                    hy[k] += 0.5 * (ex[k] - ex[k + 1])

                # END of MAIN FDTD 1D Loop

        return self
            

    def plot(self):
        ex     = self.ex
        hy     = self.hy
        ngridx = self.ngridx
        nSteps = self.numSteps
        
        x = np.linspace(0, ngridx, ngridx)
        ymin1 = S.amin(ex)
        ymin2 = S.amin(hy)
        ymax1 = S.amax(ex)
        ymax2 = S.amax(hy)
        yminimum = min(ymin1, ymin2)
        ymaximum = max(ymax1, ymax2)

        title1 = 'EX and Hy field in FDTD 1D simulation.'        

        fig = plt.figure()        

        ax1 = fig.add_subplot(121)
        ax1.set_xlabel('FDTD Cells', fontsize = 12)
        ax1.plot(x, ex, 'tab:blue', label = 'Ex')
        ax1.set_xlim([0, ngridx])
        ax1.legend(loc = 'upper center', bbox_to_anchor=(0.5, 0.1),  shadow=True, ncol=2)


        ax2 = fig.add_subplot(122)
        ax2.set_xlabel('FDTD Cells', fontsize = 12)        
        ax2.plot(x, hy, 'tab:red', label = 'Hy')
        ax2.set_xlim([0, ngridx])
        ax2.legend(loc = 'upper center', bbox_to_anchor=(0.5, 0.1),  shadow=True, ncol=2)


        plt.suptitle(title1, fontsize = 20)
        plt.savefig('Figure.png')
        #plt.show()

        
        


        
        
        
        
