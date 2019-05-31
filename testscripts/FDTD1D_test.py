""" Testing FDTD 1D
"""
from __future__ import print_function
from __future__ import absolute_import

from builtins import zip
from builtins import str
from builtins import range
from builtins import object

import scipy
import matplotlib
import bases
from solvers.FDTD.FDTD1D import FDTD1D

__author__ = 'Abhijit Bhattacharyya'

ngridx = 100
ex = scipy.zeros(ngridx, dtype = float)
hy = scipy.zeros(ngridx, dtype = float)

distTravel = float(ngridx)               # distance through which the EM wave travels
signalFreq = 400.0               # in MHZ
epsRMedium = 4.0
sigmaMedium = 4.0    # S/m
centrePulseInc = 40.0    #  5.0
pulseSpread = 30 # 12         # 30
centreProbSpace = pulseSpread / 2 + 2    # pulseSpread/2   # ngridx / 2
numSteps = 450
plotOK = 0   # will it plot, yes by default
animOK = 1   # will it animate on screen ?
isABC = 1    # Add absorbing boundary conditions
isLossy = 1  # is the medium of propagation a lossy dielectric mediium?

FDTD1DObj = FDTD1D('FDTD1D', ex, hy, ngridx, distTravel, signalFreq, centrePulseInc, pulseSpread, centreProbSpace, numSteps, epsRMedium, sigmaMedium, plotOK, animOK, isABC, isLossy)
FDTD1DSolution = FDTD1DObj.computeFDTD1D()
# FDTD1DObj.plot()
