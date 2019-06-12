""" Testing FDTD 3D
"""
from __future__ import print_function
from __future__ import absolute_import
from builtins import zip
from builtins import str
from builtins import range
from builtins import object
import scipy as S
import matplotlib
from bases.constants import cLight, mu0, eps0, smallNumber
from solvers.FDTD.FDTD3D import FDTD3D

__author__ = 'Abhijit Bhattacharyya'

"""
Let us consider a+y directed plane wave is moving along dielectric sphere of radius 4.5 cm
"""
epsa_r_l   = 1.0                         # epsilon_r for lab 
epsa_r_d   = 4.0                         # epsilon_r for dielectric
mu_d       = mu0                         # permeability for dielectric
sigma_l    = smallNumber                 # For vac = 0 = small number to avoid spurious reflection
sigma_d    = 0.0                         # for dielectric
ud         = cLight / S.sqrt(epsa_r_d)   # group velocity in dielectric
signalFreq = 2500.0                      # in MHZ
signalFreq *= 1.0e6                        # in Hz
signalPeriod= 1.0 / signalFreq           # period = 1/T
lambdad    = ud / signalFreq
delta      = lambdad / 20.0              # delta = Del_X = Del_Y = Del_Z
dT         = delta / (2.0 * cLight)      #
radiusd    = 4.5 / 100.0                 # radius = 4.5 cm expressed in m
gridSize   = int((radiusd / delta) + 0.5)
ngridx = ngridy = ngridz = 3 * gridSize
origXd     = 0                           # X coord of centre of dielectric
origYd     = 0                           # Y coord of centre of dielectric
origZd     = 0                           # Z coord of centre of dielectric
matType    = 2                           # materials: vacuum and dielectric
waveNumber    = 6
TotalTimeStep = int(waveNumber * signalPeriod)

# Create required arrays
ex = S.zeros((ngridx + 2, ngridy + 2, ngridz + 2, TotalTimeStep + 1), dtype = float)
ey = S.zeros((ngridx + 2, ngridy + 2, ngridz + 2, TotalTimeStep + 1), dtype = float)
ez = S.zeros((ngridx + 2, ngridy + 2, ngridz + 2, TotalTimeStep + 1), dtype = float)
hx = S.zeros((ngridx + 2, ngridy + 2, ngridz + 2, TotalTimeStep + 1), dtype = float)
hy = S.zeros((ngridx + 2, ngridy + 2, ngridz + 2, TotalTimeStep + 1), dtype = float)
hz = S.zeros((ngridx + 2, ngridy + 2, ngridz + 2, TotalTimeStep + 1), dtype = float)
epsr = S.zeros(matType, dtype = float)
sigr = S.zeros(matType, dtype = float)
epsr[0] = epsa_r_l
epsr[1] = epsa_r_d
sigr[0] = sigma_l
sigr[1] = sigma_d

plotOK = 0   # will it plot, yes by default
animOK = 1   # will it animate on screen ?

FDTD3DObj = FDTD3D(ex, ey, ez, hx, hy, hz, epsr, sigr, matType, delta, dT, radiusd, gridSize, TotalTimeStep, 'FDTD3D', ngridx, ngridy, ngridz, origXd, origYd, origZd, signalFreq, plotOK, animOK)
FDTD3DSolution = FDTD3DObj.computeFDTD3D()
# FDTD3DObj.plot()
