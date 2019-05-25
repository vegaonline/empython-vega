"""Transfer matrix for isotropic and anisotropic multilayer

It computes the power reflected and transmitted by a multilayer
The multilayer can be made of isotropic an anisotropic layers with any thickness
"""

from builtins import zip
from builtins import range
from builtins import object

__author__ = 'Abhijit Bhattacharyya'

import scipy as S
from scipy.linalg import inv
from bases.utils import snell, norm
from bases.constants import cLight, mu0

#import gnuplot  ?????? I wish matpotlib


class TransferMatrix(object):
    """Class to handle the transfer matrix solvers."""
    def __init__(self, multilayer):
        """Set the multilayer
        INPUT
        multilayer = Multilayer obj describing the sequence of layers
        """

        self.setMultilayer(multilayer)

    def setMultilayer(self, m):
        self.multilayer = m.simplify()

        
class IsotropicTranferMatrix(TransferMatrix):
    def __init__(self, multilayer, theta_inc):

        """ set multilayer and incident angle
        INPUT
        multilayer = Multilayer obj describing sequence of layers
        theta_inc = angle of incidence wave (in radian) with respect to the normal
        """
        if not multilayer.isIsotropic():
            raise ValueError('Cannot use IsotropicTransferMatrix with anisotropic multilayer')
        TransferMatrix.__init__(self, multilayer)
        self.theta_inc = theta_inc

        def solve(self, wls):
        """Isotropic solver.
        INPUT
        wls = wavelengths to scan (any asarray-able object).
        OUTPUT
        self.Rs, self.Ts, self.Rp, self.Tp = power reflected and
        transmitted on s and p polarizations.
        """

        self.wls = S.asarray(wls)

        multilayer = self.multilayer
        theta_inc = self.theta_inc

        nlayers = len(multilayer)
        d = S.array([l.thickness for l in multilayer]).ravel()

        Rs = S.zeros_like(self.wls)
        Ts = S.zeros_like(self.wls)
        Rp = S.zeros_like(self.wls)
        Tp = S.zeros_like(self.wls)

        Dp = S.zeros((2, 2), dtype=complex)
        Ds = S.zeros((2, 2), dtype=complex)
        P = S.zeros((2, 2), dtype=complex)
        Ms = S.zeros((2, 2), dtype=complex)
        Mp = S.zeros((2, 2), dtype=complex)
        k = S.zeros((nlayers, 2), dtype=complex)

        ntot = S.zeros((self.wls.size, nlayers), dtype=complex)
        for i, l in enumerate(multilayer):
            #      ntot[:,i] = l.mat.n(self.wls,l.mat.T0)
            ntot[:, i] = l.mat.n(self.wls, l.mat.toc.T0)

        for iwl, wl in enumerate(self.wls):
            
            n = ntot[iwl, :]
            theta = snell(theta_inc, n)

            k[:, 0] = 2.0 * S.pi * n / wl * S.cos(theta)
            k[:, 1] = 2.0 * S.pi * n / wl * S.sin(theta)

            Ds = [[1.0, 1.0], [n[0] * S.cos(theta[0]), -n[0] * S.cos(theta[0])]]
            Dp = [[S.cos(theta[0]), S.cos(theta[0])], [n[0], -n[0]]]
            Ms = inv(Ds)
            Mp = inv(Dp)

            for nn, dd, tt, kk in zip(
                    n[1:-1], d[1:-1], theta[1:-1], k[1:-1, 0]):

                Ds = [[1.0, 1.0], [nn * S.cos(tt), -nn * S.cos(tt)]]
                Dp = [[S.cos(tt), S.cos(tt)], [nn, -nn]]
                phi = kk * dd
                P = [[S.exp(1j * phi), 0], [0, S.exp(-1j * phi)]]
                Ms = S.dot(Ms, S.dot(Ds, S.dot(P, inv(Ds))))
                Mp = S.dot(Mp, S.dot(Dp, S.dot(P, inv(Dp))))

            Ds = [
                [1.0, 1.0], [n[-1] * S.cos(theta[-1]), -n[-1] * S.cos(theta[-1])]]
            Dp = [[S.cos(theta[-1]), S.cos(theta[-1])], [n[-1], -n[-1]]]
            Ms = S.dot(Ms, Ds)
            Mp = S.dot(Mp, Dp)

            rs = Ms[1, 0] / Ms[0, 0]
            ts = 1.0 / Ms[0, 0]

            rp = Mp[1, 0] / Mp[0, 0]
            tp = 1.0 / Mp[0, 0]

            
            Rs[iwl] = S.absolute(rs)**2
            Ts[iwl] = S.absolute( (n[-1] * S.cos(theta[-1])) / (n[0] * S.cos(theta[0]))) * S.absolute(ts)**2
            Rp[iwl] = S.absolute(rp)**2
            Tp[iwl] = S.absolute((n[-1] * S.cos(theta[-1])) / (n[0] * S.cos(theta[0]))) * S.absolute(tp)**2

        self.Rs = Rs
        self.Ts = Ts
        self.Rp = Rp
        self.Tp = Tp
        return self
    
    def __str__(self):
        return 'ISOTROPIC TRANSFER MATRIX SOLVER\n\n%s\n\ntheta inc = %g' % \
               (self.multilayer.__str__(), self.theta_inc)
