""" Here some useful constants are kept
    
    @var cLight: speed of light [m/s]
    @var mu0: Magnetic permeability [N/A^2]
    @var eps0: Electric permittivity [F/m]
    @var hPlank: Plank's constant [W s^2]
    @var kB: Boltzmann's constant [J/K]

"""

__author__ = 'Abhijit Bhattacharyya'

from numpy import pi

cLight = 299792458.0
mu0 = 1.2566370614e-6   # From May 20, 2019 mu0 is NOT a constant
eps0 = 1.0 / (cLight * cLight * mu0)
hPlank = 6.62606896e-34
hBar = hPlank / (2.0 * pi)
kB = 1.38065040e-23
