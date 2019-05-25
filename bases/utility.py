""" Some useful functions and objects"""

from __future__ import print_function
from builtins import zip
from builtins import str
from builtins import range
from builtins import object

__author__ = 'Abhijit Bhattacharyya'

import numpy
import bases.constants
import bases.materials
import scipy.linalg
import scipy.interpolate
import scipy.optimize
import time
import sys


class Layer(object):

    """A layer is defined by a material (iso or aniso) and a thickness."""

    def __init__(self, mat, thickness):
        """Set the material and the thickness."""

        self.mat = mat
        self.thickness = thickness

    def isIsotropic(self):
        """Return True if the material is isotropic, False if anisotropic."""

        return self.mat.isIsotropic()

    def getEPSFourierCoeffs(self, wl, n, anisotropic=True):
        """Return the Fourier coefficients of eps and eps**-1, orders [-n, n]."""

        nood = 2 * n + 1
        hmax = nood - 1
        if not anisotropic:
            # isotropic
            EPS = numpy.zeros(2 * hmax + 1, dtype=complex)
            EPS1 = numpy.zeros_like(EPS)
            rix = self.mat.n(wl)
            EPS[hmax] = rix**2
            EPS1[hmax] = rix**(-2)
            return EPS, EPS1
        else:
            # anisotropic
            EPS = numpy.zeros((3, 3, 2 * hmax + 1), dtype=complex)
            EPS1 = numpy.zeros_like(EPS)
            EPS[:, :, hmax] = numpy.squeeze(
                self.mat.epsilonTensor(wl)) / EMpy.constants.eps0
            EPS1[:, :, hmax] = scipy.linalg.inv(EPS[:, :, hmax])
            return EPS, EPS1

    def capacitance(self, area=1., wl=0):
        """Capacitance = eps0 * eps_r * area / thickness."""

        if self.isIsotropic():
            eps = EMpy.constants.eps0 * numpy.real(self.mat.n(wl).item()**2)
        else:
            # suppose to compute the capacitance along the z-axis
            eps = self.mat.epsilonTensor(wl)[2, 2, 0]

        return eps * area / self.thickness

    def __str__(self):
        """Return the description of a layer."""

        return "%s, thickness: %g" % (self.mat, self.thickness)

    
def deg2rad(x):
    """Convert deg to rad"""
    return x / 180.0 * numpy.pi

    
def rad2deg(x):
    """Convert rad to deg"""
    return x / numpy.pi * 180.0

def norm(x):
    """Return norm of 1D vector"""
    return numpy.sqrt(numpy.vdot(x, x))

def normalize(x):
    """Return a normalized vector"""
    return x / norm(x)

def euler_rotate(X, phi, theta, psi):
    """Euler Rotate
    Rotate matrix X by angles phi, theta, psi
    
    INPUT
    X = 2D numpy.array
    phi, theta, psi = rotated angles

    OUTPUT
    Rotated Matrix = 2D numpy.array
    
    see http://mathworld.wolfram.com/EulerAngles.html
    """

    A = numpy.array([
        [numpy.cos(psi) * numpy.cos(phi) - numpy.cos(theta) * numpy.sin(phi) * numpy.sin(psi),
         -numpy.sin(psi) * numpy.cos(phi) - numpy.cos(theta) * numpy.sin(phi) * numpy.cos(psi),
         numpy.sin(theta) * numpy.sin(phi)
        ],
        [numpy.cos(psi) * numpy.sin(phi) + numpy.cos(theta) * numpy.cos(phi) * numpy.sin(psi),
         -numpy.sin(psi) * numpy.sin(phi) + numpy.cos(theta) * numpy.cos(phi) * numpy.cos(psi),
         -numpy.sin(theta) * numpy.cos(phi)
        ],
        [numpy.sin(theta) * numpy.sin(psi), numpy.sin(theta) * numpy.cos(psi), numpy.cos(theta)]
    ])
    return numpy.dot(A, numpy.dot(X, scipy.linalg.inv(A)))

def snell(theta_inc, n):
    """Snell's law
    
    INPUT
    theta_inc = angle of incidence
    n = 1D numpy.array of refractive indices

    OUTPUT
    theta = 1D numpy.array
    
    """

    theta = numpy.zeros_like(n)
    theta[0] = theta_inc
    for i in range(1, n.size):
        theta[i] = numpy.arcsin(n[i - 1] / n[i] * numpy.sin(theta[i - 1]))
    return theta

def group_delay and dispersion(wls, y):
    """ Compute group delay and dispersion
    INPUT
    wls = wavlengths(ndarray)
    y = function(ndarray)

    OUTPUT
    phi = phase of function in rad
    tau = group delay in ps
    Dpsnm = dispersion in ps / nm

    NOTE
    wls and y must have same shape
    phi has same shape as wls
    tau has wls.shape - (...., 1)
    Dpsnm has wls.shape - (....., 2)
    """

    # transform the input in ndarrays
    wls = numpy.asarray(wls)
    y = numpy.asarray(y)

    # checking for good input
    if wls.shape != y.shape:
        raise ValueError('wls and y must have the same shape.')

    f = bases.constants.cLight / wls

    df = numpy.diff(f)
    toPSNM = 1E12 / 1E9
    cnmps = EMpy.constants.cLight / toPSNM

    # phase
    phi = numpy.unwrap(4.0 * numpy.angle(y)) / 4.0

    # group delay
    tau = -0.5 / numpy.pi * numpy.diff(phi) / df * 1E12

    # dispersion in ps/nm
    Dpsnm = -0.5 / numpy.pi / cnmps * f[1 : -1]**2 * numpy.diff(phi, 2) / df[0 : -1]**2

    return phi, tau, Dpsnm


def rix2losses(n, wl):
    """Return real(n), imag(n), alpha, alpha_cm1, alpha_dBcm1, 
    given a complex refractive index.  Power goes as: P = P0 exp(-alpha*z)."""
    nr = numpy.real(n)
    ni = numpy.imag(n)
    alpha = 4.0 * numpy.pi * ni / wl
    alpha_cm1 = alpha / 100.0
    alpha_dBcm1 = 10.0 * numpy.log10(numpy.exp(1)) * alpha_cm1
    return nr, ni, alpha, alpha_cm1, alpha_dBcm1


def loss_cm2rix(n_real, alpha_cm1, wl):
    """Return complex refractive index, 
    given real index (n_real), absorption coefficient (alpha_cm1) in cm^-1, and wavelength (wl) in meters.
    ----> Passing more than one argument as array, will return erroneous result."""
    ni = 100.0 * alpha_cm1 * wl /(numpy.pi * 4.0)
    return (n_real - 1j*ni)


def loss_m2rix(n_real, alpha_m1, wl):
    """Return complex refractive index, 
    given real index (n_real), absorption coefficient (alpha_m1) in m^-1, and wavelength (wl) in meters.
    Passing more than one argument as array, will return erroneous result."""
    ni = alpha_m1 * wl /(numpy.pi * 4.0)
    return (n_real - 1j * ni)


def loss_dBcm2rix(n_real, alpha_dBcm1, wl):
    """Return complex refractive index, 
    given real index (n_real), absorption coefficient (alpha_dBcm1) in dB/cm, and wavelength (wl) in meters.
    Passing more than one argument as array, will return erroneous result."""
    ni = 10.0 * alpha_dBcm1 * wl / (numpy.log10(numpy.exp(1)) * 4.0 * numpy.pi)
    return (n_real - 1j * ni)

def wl2f(wl0, dwl):
    """Convert a central wavelength and an interval to frequency."""
    wl1 = wl0 - dwl / 2.0
    wl2 = wl0 + dwl / 2.0
    f1 = bases.constants.cLight / wl2
    f2 = bases.constants.cLight / wl1
    f0 = (f1 + f2) / 2.0
    df = (f2 - f1)
    return f0, df


def f2wl(f0, df):
    """Convert a central frequency and an interval to wavelength."""
    return wl2f(f0, df)

def find_peaks(x, y, threshold=1e-6):
    # find peaks' candidates
    dy = numpy.diff(y)
    ddy = numpy.diff(numpy.sign(dy))
    # idxs = numpy.where(ddy < 0)[0] + 1
    idxs = numpy.where(ddy < 0)

    if len(idxs) == 0:
        # there is only 1 min in f, so the max is on either boundary
        # get the max and set FWHM = 0
        idx = numpy.argmax(y)
        p = Peak(x[idx], y[idx], idx, x[idx], y[idx], x[idx], x[idx])
        # return a list of one element
        return [p]

    # refine search with splines
    tck = scipy.interpolate.splrep(x, y)
    # look for zero derivative
    absdy = lambda x_: numpy.abs(scipy.interpolate.splev(x_, tck, der=1))

    peaks = []
    for idx in idxs:

        # look around the candidate
        xtol = (x.max() - x.min()) * 1e-6
        xopt = scipy.optimize.fminbound(
            absdy, x[idx - 1], x[idx + 1], xtol=xtol, disp=False)
        yopt = scipy.interpolate.splev(xopt, tck)

        if yopt > threshold * y.max():

            # FWHM
            tckFWHM = scipy.interpolate.splrep(x, y - 0.5 * yopt)
            roots = scipy.interpolate.sproot(tckFWHM)

            idxFWHM = numpy.searchsorted(roots, xopt)
            if idxFWHM <= 0:
                xFWHM_1 = x[0]
            else:
                xFWHM_1 = roots[idxFWHM - 1]
            if idxFWHM >= len(roots):
                xFWHM_2 = x[-1]
            else:
                xFWHM_2 = roots[idxFWHM]

            p = Peak(xopt, yopt, idx, x[idx], y[idx], xFWHM_1, xFWHM_2)
            peaks.append(p)

    def cmp_y(x_, y_):
        # to sort in descending order
        if x_.y == y_.y:
            return 0
        if x_.y > y_.y:
            return -1
        return 1

    peaks.sort(cmp=cmp_y)
    return peaks

def cond(M):
    """Return the condition number of the 2D array M."""
    svdv = scipy.linalg.svdvals(M)
    return svdv.max() / svdv.min()


def interp2(x, y, xp, yp, fp):
    """Interpolate a 2D complex array.
    :rtype : numpy.array
    """
    f1r = numpy.zeros((len(xp), len(y)))
    f1i = numpy.zeros((len(xp), len(y)))
    for ixp in range(len(xp)):
        f1r[ixp, :] = numpy.interp(y, yp, numpy.real(fp[ixp, :]))
        f1i[ixp, :] = numpy.interp(y, yp, numpy.imag(fp[ixp, :]))
    fr = numpy.zeros((len(x), len(y)))
    fi = numpy.zeros((len(x), len(y)))
    for iy in range(len(y)):
        fr[:, iy] = numpy.interp(x, xp, f1r[:, iy])
        fi[:, iy] = numpy.interp(x, xp, f1i[:, iy])
    return fr + 1j * fi


def trapz2(f, x = None, y = None, dx = 1.0, dy = 1.0):
    """Double integrate."""
    return numpy.trapz(numpy.trapz(f, x = y, dx = dy), x = x, dx = dx)


def centered1d(x):
    return (x[1:] + x[:-1]) / 2.0


def centered2d(x):
    return (x[1:, 1:] + x[1:, :-1] + x[:-1, 1:] + x[:-1, :-1]) / 4.0

def blackbody(f, T):
    return 2.0 * bases.constants.hPlank * f**3 / (bases.constants.cLight**2) * 1.0 / (
        numpy.exp(EMpy.constants.hPlank * f / (bases.constants.kB * T)) - 1)


def warning(s):
    """Print a warning on the stdout.
    :param s: warning message
    :type s: str
    :rtype : str
    """
    print('WARNING --- {}'.format(s))

                    
