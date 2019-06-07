""" FDTD 3 D simulation
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

class FDTD3D(object):
    """ It uses 
    absorbing boundary condition, 
    normal and 
    lossy dielectric medium
    """
    def __init__(self,
                 ex, ey, ez, hx, hy, hz,
                 epsr, sigr,
                 matType, delta, dT, radiusd, gridSize, TotalTimeStep,
                 name='FDTD3D',                
                 ngridx = 100, ngridy = 100, ngridz = 100,
                 origXd= 0, origYd = 0, origZd = 0,
                 signalFreq = 50.0,
                 centrePulseInc = 10.0, pulseSpread = 5.0, centreProbSpace = 50,
                 plotOK = 1, animOK = 1, isABC = 1, isLossy = 1):

        self.ex = ex
        self.ey= ey
        self.ez = ez
        self.hx = hx
        self.hy = hy
        self.hz = hz
        self.epsr = epsr
        self.sigr = sigr
        self.matType = matType
        self.delta = delta
        self.dT = dT
        self.radiusd = radiusd
        self.gridSize = gridSize
        self.TotalTimeStep = TotalTimeStep
        self.ngridx = ngridx
        self.ngridy = ngridy
        self.ngridz = ngridz
        self.origXd = origXd
        self.origYd = origYd
        self.origZd = origZd
        self.signalFreq = signalFreq          # already converted to Hz
        self.centreProbSpace = int(centreProbSpace)
        self.centrePulseInc = centrePulseInc
        self.pulseSpread = pulseSpread
        self.plotOK = plotOK     # is OK to plot ?
        self.animOK = animOK     # is OK to animate ?
        self.isABC = isABC       # is absorbing boundary condition to avoid reflection from boundary?
        self.isLossy = isLossy   # is the dielectric medium a lossy medium?
        

    def computeFDTD3D(self):
        """ FDTD3D solver
        INPUT :
        ngridx          = number cells along x direction as Electric field propagates along X
        centrePulseInc  = Centre of the incident pulse
        pulseSpread     = Width of the incident pulse
        centreProbSpace = Centre of the problem space
        numSteps        = total number of times the main loop to be executed
        plotOK          = 0 : not to plot || 1 : plot
        OUTPUT :
        
        """        
        ex              = self.ex
        ey              = self.ey
        ez              = self.ez
        hx              = self.hx
        hy              = self.hy
        hz              = self.hz
        epsr            = self.epsr
        sigr            = self.sigr
        matType         = self.matType
        delta           = self.delta
        dT              = self.dT
        radiusd         = self.radiusd
        gridSize        = self.gridSize
        TotalTimeStep   = self.TotalTimeStep
        ngridx          = self.ngridx
        ngridy          = self.ngridy
        ngridz          = self.ngridz
        origXd          = self.origXd
        origYd          = self.origYd
        origZd          = self.origZd
        signalFreq      = self.signalFreq
        centrePulseInc  = int(self.centrePulseInc)
        pulseSpread     = self.pulseSpread
        centreProbSpace = self.centreProbSpace
        plotOK          = self.plotOK
        animOK          = self.animOK
        isABC           = self.isABC
        isLossy         = self.isLossy
        
        tCount = 0         # keeps track of total number

        lambda0 = cLight / signalFreq
        RA = ((cLight * dT) / delta) ** 2
        RB = dT / (mu0 * delta)
        ca = S.zeros(ngridx, dtype = float)
        cb = S.zeros(ngridx, dtype = float)

        # Create media array for each grid point
        # so that sigma and mu_r could be determined for a particular (x, y, z)
        xv = S.arange(0, ngridx+1)
        yv = S.arange(0, ngridy+1)
        zv = S.arange(0, ngridz+1)
        iMedia = lambda x, y, z: (S.sqrt((x - origXd + 0.5)**2 + (y - origYd + 0.5)**2 + (z - origZd + 0.5)**2) >= gridSize) * 1   # declaring media function
        

        print('origX: ', origXd, ' origY: ', origYd, ' origZ: ', origZd, ' gridSize: ', gridSize)

        exit()



        lossStart = int(ngridx / 2)

        # print('centreProbSpace: ', centreProbSpace)

        if animOK == 1:
            xx = np.linspace(0, ngridx, ngridx)            
            plt.ion()
            ax = plt.gca()
            ax.set_xlabel('FDTD Cells', fontsize = 12)            
            ax.set_autoscale_on(True)
            line1, = ax.plot(xx, ex)
            line1.set_label('EX ')
            line2, = ax.plot(xx, hy)
            line2.set_label('HY ')            
       
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
                        ex[k] += 0.5 * (hy[k - 1] - hy[k])                           # this is for non lossy medium
                    else:
                        ex[k] = ca[k] * ex[k] + cb[k] *  (hy[k - 1] - hy[k])         # this is for lossy medium

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
                if animOK == 1:
                    title1 = 'EX and Hy field in FDTD 1D simulation.'
                    plot_label1 = 'EX (Normalized) \n' + 'Time step: ' + str(nIter)
                    plot_label2 = 'HY \n' + 'Time step: ' + str(nIter)
                    line1.set_ydata(ex)                    
                    line1.set_label(plot_label1)
                    line2.set_ydata(hy)                    
                    line2.set_label(plot_label2)
                    ax.legend(loc = 'best')
                    ax.relim()
                    ax.autoscale_view(True, True, True)
                    plt.draw()
                    plt.pause(0.3)
        if plotOK == 1:
            self.plot()
            
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
        ax1.plot(x, ex, 'tab:blue', label = 'Ex (Normalized)')
        ax1.set_xlim([0, ngridx])
        ax1.legend(loc = 'best',  shadow=True, ncol=2)
        #  ax1.legend(loc = 'upper center', bbox_to_anchor=(0.5, 0.1),  shadow=True, ncol=2)

        ax2 = fig.add_subplot(122)
        ax2.set_xlabel('FDTD Cells', fontsize = 12)        
        ax2.plot(x, hy, 'tab:red', label = 'Hy')
        ax2.set_xlim([0, ngridx])
        ax2.legend(loc = 'best',  shadow=True, ncol=2)
        # ax2.legend(loc = 'upper center', bbox_to_anchor=(0.5, 0.1),  shadow=True, ncol=2)

        plt.suptitle(title1, fontsize = 20)
        plt.savefig('Figure.png')
        plt.show()
        
        


        
        


        
        
        
        
