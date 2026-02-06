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



r""" A strategy that calculates the dipole-dipole interaction based on coordinates
"""

import numpy as np
import numpy.linalg
import scipy as sp
import numbers
from copy import deepcopy

import numba as nb
from numba.experimental import jitclass

import MagnoFallas.OldRadtools as rad
from MagnoFallas.SpinPhonon import SPhUtil as sphut

from MagnoFallas.Utils import util2 as ut2
from MagnoFallas.Utils import DipoleDipole as didi


ex = np.array((1,0,0), dtype=np.float64)
ey = np.array((0,1,0), dtype=np.float64)
ez = np.array((0,0,1), dtype=np.float64)


eaxes =(ex,ey,ez)


def Estimate_auto_dipole(SH0, pos, mag_correspond_list, lst=None,   disMax = 10, dPhi=0.01):
    r"""
    automatically includes matrix elements of the magnon-phonon interaction 
    based on derivations of the dipole-dipole interaction up to the distance disMax
    the dipole-dipole interactions are supposed to be already included into the spin Hamiltonian
    """
    cvec0 = np.array((0,0,0), dtype = np.int32)

    SH = deepcopy(SH0)
    SH.notation = (True, True, -1)
    double_counting_factor = 0.5
    
    l_i1, l_i2, l_cvec = ut2.formLists(SH, pos, disMax = disMax)
    Ncalc = len(l_i1)

    if lst==None:
        lst = sphut.EmptydJList()
    
    for ica in range(Ncalc):
        i1 = l_i1[ica]
        i2 = l_i2[ica]
        
        n1 = mag_correspond_list[i1]
        n2 = mag_correspond_list[i2]
        
        S1 = SH.magnetic_atoms[i1].spin
        S2 = SH.magnetic_atoms[i2].spin
        
        po0 = pos[i1]
        po1 = pos[i2]
        
        cvec = l_cvec[ica]
        rvec = cvec[0]*SH.a1 + cvec[1]*SH.a2 + cvec[2]*SH.a3
        rvec += po1-po0

        for e1 in eaxes:
            drJ = didi.DipDipDeriv(rvec, e1, S1=S1, S2=S2).astype(complex) * double_counting_factor
            dJline1 = sphut.TderivJ(i1,i2, np.array(cvec), n2, e1.copy(), np.array(cvec), drJ)
            dJline2 = sphut.TderivJ(i1,i2, np.array(cvec), n1, e1.copy(), np.array(cvec0), -drJ)
            lst.append(dJline1)
            lst.append(dJline2)
        
    return lst
