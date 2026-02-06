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




import numpy as np
import numpy.linalg
import numba as nb
import scipy as sp
import copy

import MagnoFallas.OldRadtools as rad

from MagnoFallas.Utils import util2 as ut2



def vec_to_RM(vec0):
    r"""
    for arbitrary vector vec0
    generates a transformation matrix which would rotate ez to its direction
    """
    vec = vec0/np.linalg.norm(vec0)
    if (vec@ut2.ez) == 1:
        return(np.eye(3))
    elif (vec@ut2.ez) == -1:
        a1 = np.array((1,-1,-1))
        return np.diag(a1)
    else:
        rv = np.cross(ut2.ez, vec)
        arv = np.linalg.norm(rv)
        erot = rv/arv
        angle = np.arcsin(arv)
        if vec@ut2.ez<0:
            angle = np.pi-angle
        rv2 = erot*angle
        R = sp.spatial.transform.Rotation.from_rotvec(rv2)
        M = R.as_matrix()
    
        return M


def make_FerroHam(SH0):
    r"""
    calculates the "equivalent Ferromagnetic Hamiltonian"
    where spin coordinate frame is rotated (for each spin separately) 
    along the equilibrium spin direction.
    It is required for magnon transformation in non-ferromagnetic materials
    SH0 - initial Hamiltonian
    """
    SH = copy.deepcopy(SH0)
    for at in SH.magnetic_atoms:
        s0 = at.spin
        sv = np.array((0,0,s0))
        at.spin_vector = sv
        
    for at1, at2, (i, j, k), J0 in SH0:
        Rm1 = vec_to_RM(at1.spin_vector)
        Rm2 = vec_to_RM(at2.spin_vector)
        Jmat0 = J0.matrix
        Jmat = np.transpose(Rm1) @ Jmat0 @ Rm2
        Jrad = rad.ExchangeParameter(matrix=Jmat)
        SH[at1.name, at2.name, (i,j,k)] = Jrad
    return SH

def FerroHam_rotation_dict(SH0):
    r"""
    calculate the dictionary of rotation matricies
    for the "ferromagnetic" transformation
    SH0 - initial Hamiltonian
    """
    RotDict = {}
    for at in SH0.magnetic_atoms:
        sv = at.spin_vector
        Rm = vec_to_RM(sv)
        RotDict[at.name] = Rm
    return RotDict




