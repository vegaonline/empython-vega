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
                 name='FDTD3D', ngridx = 100, ngridy = 100, ngridz = 100, origXd= 0, origYd = 0, origZd = 0, signalFreq = 50.0, plotOK = 1, animOK = 1):

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
        self.plotOK = plotOK     # is OK to plot ?
        self.animOK = animOK     # is OK to animate ?

    def computeFDTD3D(self):
        """ FDTD3D solver
        INPUT :

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
        plotOK          = self.plotOK
        animOK          = self.animOK
        
        tCount = 0         # keeps track of total number

        lambda0 = cLight / signalFreq
        RA = ((cLight * dT) / delta) ** 2
        RB = dT / (mu0 * delta)
        iMedia = lambda x, y, z: (S.sqrt((x - origXd + 0.5) ** 2 + (y - origYd + 0.5) ** 2 + (z - origZd + 0.5)**2) >= gridSize) * 1   # declaring media function
        R      = lambda l : dT / epsr[l]
        CA     = lambda l : 1.0 - ((R[l] * sigr[l]) / epsr[l])
        CB     = lambda l : RA / epsr[l]

        # Create media array for each grid point
        # so that sigma and mu_r could be determined for a particular (x, y, z)
        xv = S.arange(0, ngridx + 1)
        yv = S.arange(0, ngridy + 1)
        zv = S.arange(0, ngridz + 1)
        
        """
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
        """

        ncur = 3 # index for time current time t
        npr1 = 2 # index for time (t - 1)
        npr2 = 1 # index for time (t - 2)
        
        if (TotalTimeStep > 0):            
            tCount += 1
            npr2 = npr1
            npr1 = ncur
            ncur = (ncur % 3) + 1
            
            # MAIN FDTD 3D Loop
            

            for k in range ( 1, ngridz ):            # Z
                for j in range ( 1, ngridy ):        # Y
                    for i in range (1, ngridx ):     # X
                        
                        # this is for absorbing layer simulation

                        hx[i + 1, j + 1, k + 1, ncur] = hx[i + 1, j + 1, k + 1, npr1] + RB * (ey[i + 1, j + 1, k + 1 + 1, npr1] - ey[i + 1, j + 1, k + 1, npr1] + ez[i + 1, j + 1, k + 1, npr1] - ez[i + 1, j + 1 + 1, k + 1, npr1])
                        hy[i + 1, j + 1, k + 1, ncur] = hy[i + 1, j + 1, k + 1, npr1] + RB * (ez[i + 1 + 1, j , k, npr1]        - ez[i + 1, j + 1, k + 1, npr1] + ex[i + 1, j + 1, k + 1, npr1] - ex[i + 1, j + 1, k + 1 + 1, npr1])
                        hz[i + 1, j + 1, k + 1, ncur] = hz[i + 1, j + 1, k + 1, npr1] + RB * (ex[i + 1, j + 1 + 1, k + 1, npr1] - ex[i + 1, j + 1, k + 1, npr1] + ey[i + 1, j + 1, k + 1, npr1] - ey[i + 1 + 1, j + 1, k + 1, npr1])
                        media = iMedia(i + 1, j + 1, k + 1)

                        ex[i + 1, j + 1, k + 1, ncur] = CA[media] * ex[i + 1, j + 1, k + 1, npr1] + CB[media] * (hz[i + 1, j + 1, k + 1, ncur] - hz[i + 1, j + 1 - 1, k + 1, ncur] + hy[i + 1, j + 1, k + 1 - 1, ncur]     - hy[ i + 1, j + 1, k + 1, ncur])
                        ey[i + 1, j + 1, k + 1, ncur] = CA[media] * ey[i + 1, j + 1, k + 1, npr1] + CB[media] * (hx[i + 1, j + 1, k + 1, ncur] - hx[i + 1, j + 1, k + 1 - 1, ncur] + hz[i + 1 - 1, j + 1, k + 1 - 1, ncur] - hz[ i + 1, j + 1, k + 1, ncur])
                        ez[i + 1, j + 1, k + 1, ncur] = CA[media] * ez[i + 1, j + 1, k + 1, npr1] + CB[media] * (hy[i + 1, j + 1, k + 1, ncur] - hy[i + 1 - 1, j + 1, k + 1, ncur] + hx[i + 1, j + 1 - 1, k + 1 - 1, ncur] - hx[ i + 1, j + 1, k + 1, ncur])

                        print(i + 1, '  ', j + 1, '  ', k + 1, '  ', ncur, '  ', ex[i + 1, j + 1, k + 1, ncur], '  ', ey[i + 1, j + 1, k + 1, ncur], '  ', ez[i + 1, j + 1, k + 1, ncur])
                        # END of MAIN FDTD 1D Loop
                        """
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
                        """
            
        return self
            

    def plot(self):
        ex     = self.ex
        hy     = self.hy
        ngridx = self.ngridx
        ngridy = self.ngridy
        ngridz = self.ngridz        
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
        
        


        
        


        
        
        
        
