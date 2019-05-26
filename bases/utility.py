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

    def getEPSFourierCoeffs(self, wl, n, anisotropic = True):
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
                self.mat.epsilonTensor(wl)) / bases.constants.eps0
            EPS1[:, :, hmax] = scipy.linalg.inv(EPS[:, :, hmax])
            return EPS, EPS1

    def capacitance(self, area=1.0, wl = 0):
        """Capacitance = eps0 * eps_r * area / thickness."""

        if self.isIsotropic():
            eps = bases.constants.eps0 * numpy.real(self.mat.n(wl).item()**2)
        else:
            # suppose to compute the capacitance along the z-axis
            eps = self.mat.epsilonTensor(wl)[2, 2, 0]

        return eps * area / self.thickness

    def __str__(self):
        """Return the description of a layer."""

        return "%s, thickness: %g" % (self.mat, self.thickness)


class Multilayer(object):
    """A Multilayer is a list of layers with some more methods"""

    def __init__(self, data = None):
        """Initialize the data lilst."""
        if data is None:
            data = []
        self.data = data[:]

    def __delitem__(self, i):
        """Delete an item from the list."""
        del self.data[i]

    def __getitem__(self, i):
        """Get an item of the list of layers."""
        return self.data[i]

    def __getslice__(self, i, j):
        """Get a Multilayer from a slice of layers."""
        return Multilayer(self.data[i:j])

    def __len__(self):
        """Return the number of layers."""
        return len(self.data)

    def __setitem__(self, i, item):
        """Set an item of the list of layers."""
        self.data[i] = item

    def __setslice__(self, i, j, other):
        """Set a slice of layers."""
        self.data[i:j] = other

    def append(self, item):
        """Append a leyer to the list of layers."""
        self.data.append(item)

    def extend(self, other):
        """Extend the layers list with other layers."""
        self.data.extend(other)

    def insert(self, i, item):
        """Insert a new layer in the layer's list at the position i."""
        self.data.insert(i, item)

    def remove(self, item):
        """Remove item from a layers list"""
        self.data.remove(item)


    def pop(self, i = -1):
        return self.data.pop(i)

    def isIsotropic(self):
        """Return True of all the layers of Multilayer are isotropic else False."""
        return numpy.asarray([m.isIsotropic() for m in self.data]).all()

    def __str__(self):
        """Return a description of the Multilayer"""
        if self.__len__() == 0.0:
            list_str = "<empty>"
        else:
            list_str = '\n'.join(['%d: %s' % (i1, 1.0__str__()) for il, l in enumerate(self.data)])
            return 'Multilayer \n...............\n' + list_str


class Slice(Multilayer):
    def __init__(self, width, *argv):
        Multilayer.__init__(self, *argv)
        self.width = width

    def heights(self):
        return numpy.array([l.thickness for l in self])

    def ys(self):
        return numpy.r_[0.0, self.heights().cumsum()]

    def height(self):
        return self.heights().sum()

    def find_layer(self, y):
        l = numpy.where(self.ys() <= y)[0]
        if len(l) > 0:
            return self[min(l[-1], len(self) - 1)]
        else:
            return self[0]

    def plot(self, x0, x1, nmin, nmax, wl = 1.55e-6):
        try:
            import pylab
        except ImportError:
            warning('no pylab installed')
            return
        y0 = 0
        # ytot = sum([l.thickness for l in self])
        for l in self:
            y1 = y0 + l.thickness
            n = l.mat.n(wl)
            r = 1.0 - (1.0 * (n - nmin) / (nmax - nmin))
            pylab.fill([x0, x1, x1, x0], [y0, y0, y1, y1], ec = 'yellow', fc = (r, r, r), alpha = 0.5)
            y0 = y1
        pylab.axis('image')

    def __str__(self):
        return 'width = %e\n%s' % (self.width, Multilayer.__str__(self))


class CrossSection(list):

    def __str__(self):
        return '\n'.join('%s' % s for s in self)

    def widths(self):
        return numpy.array([s.width for s in self])

    def xs(self):
        return numpy.r_[0.0, self.widths().cumsum()]

    def ys(self):
        tmp = numpy.concatenate([s.ys() for s in self])
        # get rid of numerical errors
        tmp = numpy.round(tmp * 1e10) * 1e-10
        return numpy.unique(tmp)

    def width(self):
        return self.widths().sum()

    def grid(self, nx_per_region, ny_per_region):

        xs = self.xs()
        ys = self.ys()

        nxregions = len(xs) - 1
        nyregions = len(ys) - 1

        if numpy.isscalar(nx_per_region):
            nx = (nx_per_region,) * nxregions
        elif len(nx_per_region) != nxregions:
            raise ValueError('wrong nx_per_region dim')
        else:
            nx = nx_per_region

        if numpy.isscalar(ny_per_region):
            ny = (ny_per_region,) * nyregions
        elif len(ny_per_region) != nyregions:
            raise ValueError('wrong ny_per_region dim')
        else:
            ny = ny_per_region

        X = []
        x0 = xs[0]
        for x, n in zip(xs[1:], nx):
            X.append(numpy.linspace(x0, x, n + 1)[:-1])
            x0 = x
        X = numpy.concatenate(X)
        X = numpy.r_[X, x0]

        Y = []
        y0 = ys[0]
        for y, n in zip(ys[1:], ny):
            Y.append(numpy.linspace(y0, y, n + 1)[:-1])
            y0 = y
        Y = numpy.concatenate(Y)
        Y = numpy.r_[Y, y0]

        return X, Y

    def find_slice(self, x):
        s = numpy.where(self.xs() <= x)[0]
        if len(s) > 0:
            return self[min(s[-1], len(self) - 1)]
        else:
            return self[0]

    def _epsfunc(self, x, y, wl):
        if numpy.isscalar(x) and numpy.isscalar(y):
            return self.find_slice(x).find_layer(y).mat.n(wl)**2
        else:
            raise ValueError('only scalars, please!')

    def epsfunc(self, x, y, wl):
        eps = numpy.ones((len(x), len(y)), dtype=complex)
        for ix, xx in enumerate(x):
            for iy, yy in enumerate(y):
                eps[ix, iy] = self._epsfunc(xx, yy, wl)
        return eps

    def plot(self, wl = 1.55e-6):
        try:
            import pylab
        except ImportError:
            warning('no pylab installed')
            return
        x0 = 0
        ns = [[l.mat.n(wl) for l in s] for s in self]
        nmax = max(max(ns))
        nmin = min(min(ns))
        for s in self:
            x1 = x0 + s.width
            s.plot(x0, x1, nmin, nmax, wl = wl)
            x0 = x1
        pylab.axis('image')

class Peak(object):

    def __init__(self, x, y, idx, x0, y0, xFWHM_1, xFWHM_2):
        self.x = x
        self.y = y
        self.idx = idx
        self.x0 = x0
        self.y0 = y0
        self.xFWHM_1 = xFWHM_1
        self.xFWHM_2 = xFWHM_2
        self.FWHM = numpy.abs(xFWHM_2 - xFWHM_1)

    def __str__(self):
        return '(%g, %g) [%d, (%g, %g)] FWHM = %s' % ( self.x, self.y, self.idx, self.x0, self.y0, self.FWHM)
        
        
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
    cnmps = bases.constants.cLight / toPSNM

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
    ni = 100.0 * alpha_cm1 * wl / (numpy.pi * 4.0)
    return (n_real - 1j * ni)


def loss_m2rix(n_real, alpha_m1, wl):
    """Return complex refractive index, 
    given real index (n_real), absorption coefficient (alpha_m1) in m^-1, and wavelength (wl) in meters.
    Passing more than one argument as array, will return erroneous result."""
    ni = alpha_m1 * wl / (numpy.pi * 4.0)
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
    absdy = lambda x_: numpy.abs(scipy.interpolate.splev(x_, tck, der = 1))

    peaks = []
    for idx in idxs:

        # look around the candidate
        xtol = (x.max() - x.min()) * 1e-6
        xopt = scipy.optimize.fminbound(
            absdy, x[idx - 1], x[idx + 1], xtol = xtol, disp = False)
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

    peaks.sort(cmp = cmp_y)
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
        numpy.exp(bases.constants.hPlank * f / (bases.constants.kB * T)) - 1)


def warning(s):
    """Print a warning on the stdout.
    :param s: warning message
    :type s: str
    :rtype : str
    """
    print('WARNING --- {}'.format(s))

                    
