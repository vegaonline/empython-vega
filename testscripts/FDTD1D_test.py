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

ngridx = 200
ex = scipy.zeros(ngridx, dtype = float)
hy = scipy.zeros(ngridx, dtype = float)

distTravel = 400.0               # distance through which the EM wave travels
signalFreq = 400.0               # in MHZ
epsRMedium = 4.0
sigmaMedium = 0.04    # S/m
centrePulseInc = 40.0    #  5.0
pulseSpread = 12         # 30
centreProbSpace = ngridx / 2    # pulseSpread/2   # ngridx / 2
numSteps = 500
plotOK = 1   # will it plot, yes by default
isABC = 1    # Add absorbing boundary conditions
isLossy = 1  # is the medium of propagation a lossy dielectric mediium?

FDTD1DObj = FDTD1D('FDTD1D', ex, hy, ngridx, distTravel, signalFreq, centrePulseInc, pulseSpread, centreProbSpace, numSteps, epsRMedium, sigmaMedium, plotOK, isABC, isLossy)
FDTD1DSolution = FDTD1DObj.computeFDTD1D()
FDTD1DObj.plot()
