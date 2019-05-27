from builtins import range
from builtins import object
import numpy

class ModeSolver(object):
    
    def solve(self, *argv):
        raise NotImplementedError()        

class Mode(object):
    
    def get_x(self, x = None, y = None):
        raise NotImplementedError()

    def get_y(self, x = None, y = None):
        raise NotImplementedError()
    
    def intensity(self, x = None, y = None):
        raise NotImplementedError()

    def TEfrac(self, x = None, y = None):
        raise NotImplementedError()

    def overlap(self, x = None, y = None):
        raise NotImplementedError()

    def get_fields_for_FDTD(self, x, y):
        raise NotImplementedError()


def overlap(m1, m2, x = None, y = None):
    return m1.overlap(m2, x, y)

def interface_matrix(solver1, solver2, x = None, y = None):

    neigs = solver1.nmodes
    
    O11 = numpy.zeros((neigs, neigs), dtype=complex)
    O22 = numpy.zeros((neigs, neigs), dtype=complex)
    O12 = numpy.zeros((neigs, neigs), dtype=complex)
    O21 = numpy.zeros((neigs, neigs), dtype=complex)

    for i in range(neigs):
        for j in range(neigs):
            
            O11[i, j] = overlap(solver1.modes[i], solver1.modes[j], x, y)
            O22[i, j] = overlap(solver2.modes[i], solver2.modes[j], x, y)
            O12[i, j] = overlap(solver1.modes[i], solver2.modes[j], x, y)
            O21[i, j] = overlap(solver2.modes[i], solver1.modes[j], x, y)

    return (O11, O22, O12, O21) 
