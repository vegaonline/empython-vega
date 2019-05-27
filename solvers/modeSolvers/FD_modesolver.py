"""Finite difference mode solver
@see: Fallahkhair, "Vector Finite Difference Modesolver for Anisotropic Dielectric Waveguides",
@see: JLT 2007 
   <http://www.photonics.umd.edu/wp-content/uploads/pubs/ja-20/Fallahkhair_JLT_26_1423_2008.pdf>}
@see: DOI of above reference <http://doi.org/10.1109/JLT.2008.923643>
@see: http://www.mathworks.com/matlabcentral/fileexchange/loadFile.do?objectId=12734&objectType=FILE

"""
from __future__ import print_function
from builtins import zip
from builtins import str
from builtins import range

import numpy
import scipy
import scipy.optimize
import bases.utility
import solvers.modeSolvers.interface import *

class semiVectorialFDModeSolver(ModeSolver):
    """
Calculates modes of a dielectric waveguide using semivectorial finite difference method. Literature demands it is a bit faster than full vectorial mode solver. It DOES NOT accept non-isotropic permittivity. That is different RI along different dimensions cannot be used
