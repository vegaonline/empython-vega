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
sigma_l    = smallNumber                 # Though it must be zero for vac, a small number is used to avoid spurious reflection
sigma_d    = 0.0                         # for dielectric
ud         = cLight / S.sqrt(epsa_r_d)   # group velocity in dielectric
signalFreq = 2500.0                      # in MHZ
signalFreq *= 1e6                        # in Hz
signalPeriod= 1.0 / signalFreq           # period = 1/T
lambdad    = ud / signalFreq
delta      = lambdad / 20.0              # delta = Del_X = Del_Y = Del_Z
dT         = delta / (2.0 * cLight)      #
radiusd    = 4.5 / 100.0                 # radius = 4.5 cm which is expressed in m
gridSize   = int((radiusd / delta) + 0.5)
ngridx = ngridy = ngridz = 3 * gridSize

matType    = 2                           # 2 Types of material: vacuum and dielectric

# Create required arrays
ex = S.zeros(ngridx, dtype = float)
ey = S.zeros(ngridy, dtype = float)
ez = S.zeros(ngridz, dtype = float)
hx = S.zeros(ngridx, dtype = float)
hy = S.zeros(ngridy, dtype = float)
hz = S.zeros(ngridz, dtype = float)
epsr = S.zeros(matType, dtype = float)
sigr = S.zeros(matType, dtype = float)

epsr[0] = epsa_r_l
epsr[1] = epsa_r_d
sigr[0] = sigma_l
sigr[1] = sigma_d

waveNumber    = 6
TotalTimeStep = waveNumber * signalPeriod


centrePulseInc = 40.0    #  5.0
pulseSpread = 30 # 12         # 30
centreProbSpace = pulseSpread / 2 + 2    # pulseSpread/2   # ngridx / 2


plotOK = 0   # will it plot, yes by default
animOK = 1   # will it animate on screen ?
isABC = 1    # Add absorbing boundary conditions
isLossy = 1  # is the medium of propagation a lossy dielectric mediium?

FDTD3DObj = FDTD3D(
    'FDTD3D', ex, ey, ez, hx, hy, hz, epsr, sigr, matType, delta, dT, radiusd, TotalTimeStep, ngridx, ngridy, ngridz, signalFreq,
    centrePulseInc, pulseSpread, centreProbSpace, plotOK, animOK, isABC, isLossy)
FDTD3DSolution = FDTD3DObj.computeFDTD3D()
# FDTD3DObj.plot()
