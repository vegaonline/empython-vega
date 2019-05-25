""" Functions and objects for Materials class
    
    A material is an object with a refractive index function
"""

__author__ = 'Abhijit Bhattacharyya'

from builtins import str
from builtins import object
from functools import partial

import numpy
from scipy.integrate import quad
from bases.constants import eps0

class Material(object):
    """ Generic class for Materials to get isotropic and anisotropic materials
    """
    def __init__(self, name = ''):
        """ Set material name. """
        self.name = name


class RefractiveIndex(object):
    """ Refractive Index
    Temperature ?
    Parameters
    
    n0_const : float
    A single value of RI independent of wavelength.
    e.g. n0_const = 1.448 for SiO2
    
    n0_poly : list
    Use polynomial dispersion function providing coeffs only to numpy.polyvals
    e.g. n0_poly=(9, 5, 3, 1) for n = 9(wl**3) + 5 * (wl**2) + 3 * wl + 1

    n0_sell : 6 element list for Sellmeier coefficients
    n(wls) = sqrt(1.0 +
    B1 * wls**2 / (wls**2 - C1) +
    B2 * wls**2 / (wls**2 - C2) +
    B3 * wls**2 / (wls**2 - C3))        n0_sell = [B1, B2, B3, C1, C2, C3]

    n0_func: function
    Arbitrary function to return RI verus wavelength
    e.g. def Sine_func(wl):
             x = wl * 1e6 # convert to microns
             return 1.887 + 0.01929 / x**2 + 1.6626e-4 / x**4 # Cauchy func
         sine_RI = RefractiveIndex(n0_func = Sine_func)
    OR
         sine_RI = RefractiveIndex(n0_func=lambda wl:1.887 + 0.1929/(wl*1e6)**2+1.6662e-4/(wl*1e6)**4)

    n0_known : dictionary
    Use if RI be evaluated at specific set of wls.
    n0_known should be a dictionary of key:value pairs corresponding to wavelength RI
    e.g. n0_known = {1500e-9:1.445, 1550e-9:1.446, 1600e-9:1.447}
    
    """
    def __init(self, n0_const = None, n0_poly = None, n0_sell = None, n0_func = None, n0_known = None):

        if n0_const is not None:
            self.get_RI = partial(self.__from_constant, n0_const)

        elif n0_poly is not None:
            self.get_RI = partial(self.__from_polynomial, n0_poly)

        elif n0_sell is not None:
            self.get_RI = partial(self.__from_sellmeier, n0_sell)

        elif n0_func is not None:
            self.get_RI = partial(self.__from_function, n0_func)

        elif n0_known is not None:
            self.get_RI = partial(self.__from_known, n0_known)

        else:
            raise ValueError('Please provide atleast one parameter.')

    @staticmethod
    def __from_constant(n0, wls):
        wls = numpy.atleast_1d(wls)
        return n0 * numpy.ones_like(wls)

    @staticmethod
    def __from_polynomial(n0, wls):
        wls = numpy.atleast_1d(wls)
        return numpy.polyval(n0_poly, wls) * numpy.ones_like(wls)

    @staticmethod
    def __from_sellmeier(n0_sell, wls):
        wls = numpy.atleast_1d(wls)
        B1, B2, B3, C1, C2, C3 = n0_sell
        return numpy.sqrt(
            1.0 +
            B1 * wls**2 / (wls**2 - C1) +
            B2 * wls**2 / (wls**2 - C2) +
            B3 * wls**2 / (wls**2 - C3)) * numpy.ones_like(wls)

    @staticmethod
    def __from_function(n0_func, wls):
        wls = numpy.atleast_1d(wls)
        return n0_func(wls) * numpy.ones_like(wls)
    
    @staticmethod
    def __from_known(n0_known, wls):
        wls = numpy.atleast_1d(wls)
        return numpy.array([n0_known.get(wlsVal, 0) for wlsVal in wls])  # interpolation

    def __call__(self, wls):
        return self.get_RI(wls)


class ThermalOpticCoefficients(object):
    """ Thermal Optic Coefficient """
    def __init__(self, data = None, T0 = 300.0):
        self.__data = data
        self.T0 = T0

    def TOC(self, T):
        if self.__data is not None:
            return numpy.polyval(self.__data, T)
        else:
            return 0.0

    def __call__(self, T):
        return self.TOC(T)

    def dnT(self, T):
        """ Integrate TOC to get RI variation."""
        return quad(self.TOC, self.T), T)[0]

    
class IsotropicMaterial(Material):
    """ subclasses Material for isotropic materials. wls must be ndarray.
    """

    def __init__(self, name = '', n0 = RefractiveIndex(n0_const = 1.0), toc = ThermalOpticCoefficient()):
        """Set Name, default temp, RI and TOC"""
        Material.__init__(self, name)
        self.n0 = n0
        self.toc = toc

    def n(self, wls, T = None):
        """ return the RI at T as [1 X wls] array """
        if T is None:
            T = self.toc.T0
        return self.n(wls, T)**2 * eps0

    def epsilon(self, wls, T = None):
        """Return epsilon at T as [1 X wls] array"""
        if T is None:
            T = self.toc.T0
        return self.n(wls, T)**2 * eps0

    def epsilonTensor(self, wls, T = None):
        """return the epsilon at T as a [3 X 3 X wls] array"""
        if T is None:
            T = self.toc.T0
        tmp = numpy.eye(3)
        return tmp[:, :, numpy.newaxis] * self.epsilon(wls, T)

    @staticmethod
    def isIsotropic():
        """Return True, because the material is isotropic."""
        return True

    def __str__(self):
        """Return material name."""
        return self.name + ', isotropic'

    
class EpsilonTensor(object):

    def __init__(self, epsilon_tensor_const = eps0 * numpy.eye(3), epsilon_tensor_known = None):
        if epsilon_tensor_known is None:
            epsilon_tensor_known = {}
            self.epsilon_tensor_const = epslon_tensor_const
            self.epsilon_tensor_known = epsilon_tensor_known

    def __call__(self, wls):
        """Return the epsilon tensor as a [3 X 3 X wls.size] matrix."""
        wls = numpy.atleast_1d(wls)
        if wls.size == 1:
            if wls.item() in self.epsilon_tensor_known:
                return self.epsilon_tensor_known[wls.item()][:, :, numpy.newaxis]
        return self.epsilon_tensor_const[:, :, numpy.newaxis] * numpy.ones_like(wls)

    
class AnisotropicMaterial(Material):

    """subclass for anisotropic materials.
    wls must be ndarray. No frequency dispersion nor thermic aware
    """

    def __init__(self, name = '', epsilon_tensor = EpsilonTensor()):
        """ Set name and default epsilon tensor. """
        Material.__init__(self, name)
        self.epsilonTensor = epsilon_tensor

    @staticmethod
    def isIsotropic():
        """Return False, because the material is isotropic."""
        return False

    def __str__(self):
        """Return material name."""
        return self.name + ', anisotropic'

    
#Vacuum
Vacuum = IsotropicMaterial(name = 'Vacuum')

#Air
Air = IsotropicMaterial(name = 'Air')

#Silicon
Si = IsotropicMaterial(
    name = 'Silicon',
    n0 = RefractiveIndex(n0_poly = (0.076006e12, -0.31547e6, 3.783)),
    toc = ThermalOpticCoefficient((-1.49e-10, 3.47e-7, 9.48e-5)))
    
# SiO2
SiO2 = IsotropicMaterial(
    name = 'Silica',
    n0 = RefractiveIndex(n0_const = 1.446),
    toc = ThermalOpticCoefficient((1.1e-4,)))

# BK7 glass (see http://en.wikipedia.org/wiki/Sellmeier_equation)
BK7 = IsotropicMaterial(
    name = 'Borosilicate crown glass',
    n0 = RefractiveIndex(n0_smcoeffs = (
        1.03961212, 2.31792344e-1, 1.01046945, 6.00069867e-15, 2.00179144e-14, 1.03560653e-10)))


class LiquidCrystal(Material):

    """Liquid Crystal.
    A liquid crystal is determined by it ordinary and extraordinary refractive indices, its 
    elastic tensor and its chirality. 
    see http://www.ee.ucl.ac.uk/~rjames/modelling/constant-order/oned/
    @ivar name: Liquid Crystal name.
    @ivar nO: Ordinary refractive index.
    @ivar nE: Extraordinary refractive index.
    @ivar K11: Elastic tensor, first component.
    @ivar K22: Elastic tensor, second component.
    @ivar K33: Elastic tensor, third component.
    @ivar q0: Chirality.
    """

    def __init__(self, name = '', nO = 1.0, nE = 1.0, nO_electrical = 1.0, nE_electrical = 1.0,
                 K11 = 0.0, K22 = 0.0, K33 = 0.0, q0 = 0.0):
        """Set name, the refractive indices, the elastic constants and
        the chirality."""

        Material.__init__(self, name)
        self.nO = nO
        self.nE = nE
        self.nO_electrical = nO_electrical
        self.nE_electrical = nE_electrical
        self.K11 = K11
        self.K22 = K22
        self.K33 = K33
        self.q0 = q0
        self.epslow = self.nO_electrical** 2
        self.deleps = self.nE_electrical** 2 - self.epslow


def get_10400_000_100(conc000):
    """Return a LiquidCrystal made of conc% 000 and (100-conc)% 100.
    Copied from net. Forgot to copy the site from where values are taken. Searching
    """

    conc = [0, 100]
    epsO_electrical = [3.38, 3.28]
    epsE_electrical = [5.567, 5.867]
    epsO = [1.47551 ** 2, 1.46922**2]
    epsE = [1.61300 ** 2, 1.57016**2]

    K11 = 13.5e-12  # elastic constant [N] (splay)
    K22 = 6.5e-12  # elastic constant [N] (twist)
    K33 = 20e-12  # elastic constant [N] (bend)
    q0 = 0  # chirality 2*pi / pitch

    nO_electrical_ = numpy.interp(conc000, conc, epsO_electrical)**0.5
    nE_electrical_ = numpy.interp(conc000, conc, epsE_electrical)**0.5
    nO_ = numpy.interp(conc000, conc, epsO)**0.5
    nE_ = numpy.interp(conc000, conc, epsE)**0.5

    return LiquidCrystal('10400_000_100_' + str(conc000) + '_' + str(100 - conc000),
        nO_, nE_, nO_electrical_, nE_electrical_, K11, K22, K33, q0)
