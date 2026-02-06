# MagnoFallas - A Python-based method for annihilating magnons
# Copyright (C) 2025-2026  Andrei Shumilin
#
# e-mail: andrei.shumilin@uv.es, hegnyshu@gmail.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.




r'''
Module for the calculation of 3-magnon processes contribution to the damping
Is not finished
'''

import numpy as np
import numpy.linalg
import scipy as sp

import numba as nb
from numba.experimental import jitclass

import MagnoFallas.OldRadtools as rad
from MagnoFallas.Interface import PseudoRad as prad

#from MagnoFallas.Utils import SpinPhonon as sph
from MagnoFallas.Utils import util2 as ut2
from MagnoFallas.Utils import SurfaceMath as smath
from MagnoFallas.Utils import generalMath as gmath


zerovec = np.zeros(3, dtype = np.float64)
roles1 = np.array((-1,-1,1), dtype=np.int32)    



Tscattering3MLST = [
    ('k0', nb.float64[::1]),              
    ('lk0', nb.int32),
    ('k1corners', nb.float64[:,::1]),  #### order: BL - TL - BR - TR
    ('lk1', nb.int32),
    ('lk2', nb.int32),    ##band index for k3
    ('k2g', nb.float64[::1]),  ## k-lattice vector for k3             
###########################################################
    ('k1real', nb.float64[::1]),
    ('k2real', nb.float64[::1]),
    ('Exist', nb.int32),
    ('DeltaInt', nb.float64),
###########################################################
    ('e0', nb.float64),
    ('e1', nb.float64),
    ('e2', nb.float64),
###########################################################
    ('Mel', nb.complex128),
###########################################################
    ('Status', nb.int32)
]



