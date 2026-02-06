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


r"""
A module controlling introduction of the dipole-dipole interaction
to the spin Hamiltonian
"""

import numpy as np
import numba as nb
import copy


import MagnoFallas.OldRadtools as rad

from MagnoFallas.SpinPhonon import SPhUtil as sphut
from MagnoFallas.Utils import util2 as ut2


from MagnoFallas.Math import Dipole_Integrations_3D as integr3D
from MagnoFallas.Math import Dipole_Integrations_2D as integr2D
from MagnoFallas.Math import Dipole_Integrations_3D_deriv as integr3Dd

ex = np.array((1.0,0.0,0.0))
ey = np.array((0.0,1.0,0.0))
ez = np.array((0.0,0.0,1.0))


### constant for the interaction of two mag. moments of one m_B at the distance of 1 angst
### given in meV
Const_dipdip0 =  0.5368151120274394e-1



###################   procedured for Long-ranged dipole-dipole interaction ###########
# interface with other modules

@nb.njit
def longrangeDDmatr(k, R0, dim, g1=2, g2=2):
    r"""
    Calculates the "exchange matrix" for the long-range dipole-dipole interaction 
    integrated with the exponent exp(ikr), 
    k - wavevector
    R0 - minimum distance for the long-range approximation
    dim = 2 or 3 - system dimension
    g1, g2 - g-factors of participating electrons
    """
    if dim==2:
        DDM = integr2D.interpolImatr2D(k,R0)
    elif dim==3:
        DDM = integr3D.kv_integrated_M(k, R0)
    else:
        DDM = np.zeros((3,3), dtype=np.complex128)
    DDM = DDM*g1*g2*(1+0j)  ##S1*S2
    DDM = DDM*Const_dipdip0
    return DDM

@nb.njit
def longrangeDDderivs(k, R0, dim, g1=2, g2=2):
    r"""
    derivatives of LR dipole-dipole matricies
    """
    
    if dim==3:
        Dx, Dy, Dz = integr3Dd.kv_integrated_D(k, R0)
    else:
        Dx = np.zeros((3,3), dtype=np.complex128)
        Dy = np.zeros((3,3), dtype=np.complex128)
        Dz = np.zeros((3,3), dtype=np.complex128)
    C1 = g1*g2*(1+0j)*Const_dipdip0
    Dx = Dx*C1
    Dy = Dy*C1
    Dz = Dz*C1
    return Dx, Dy, Dz


###################   procedureds for Short-ranged dipole-dipole interaction ##############

#### calculate the shifts in unit cell required for introduction of the short-range
#### dipole-dipole interaction up to the distance R0
def calculateNdd(SH, R0, dim=2):
    
    Nx = int(R0/np.linalg.norm(SH.a1)) + 1
    
    ex1 = SH.a1 / np.linalg.norm(SH.a1)
    ez1 = np.cross(SH.a1,SH.a2)
    ez1 /= np.linalg.norm(ez1)
    ey1 = np.cross(ez1,ex1)
    ey1 /= np.linalg.norm(ey1)

    a2eff = np.abs(SH.a2@ey1)
    Ny = int(R0/a2eff) + 1
    
    if dim==3:
        a3eff = np.abs(SH.a3@ez1)
        Nz = int(R0/a3eff) + 1
    else:
        Nz = 0
    return Nx, Ny, Nz



#### calculates the matrix of dipole-dipole interaction based on the radius-vector r0 (in A)
###  like this it corresponds to:
###  - spins   (no not!) normalized
###  - no double counting
###  - prefactor equal to -1
###  (False, False, -1 notation)
@nb.njit
def DDmatr(r0, g1=2, g2=2):
    r"""
    Dipole-dipole exchange interaction matrix corresponding to the vector distance r0
    """
    r = np.array((r0[0],r0[1],r0[2]))
    Mat0 = np.zeros((3,3))
    ar = np.sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2])
    er = r/ar
    Mat0 [0,...]= np.array([3*er[0]*er[0]-1, 3*er[0]*er[1], 3*er[0]*er[2]])
    Mat0 [1,...]= np.array([3*er[1]*er[0], 3*er[1]*er[1]-1, 3*er[1]*er[2]])
    Mat0 [2,...]= np.array([3*er[2]*er[0], 3*er[2]*er[1], 3*er[2]*er[2]-1])
    Mat = Mat0*g1*g2  #*S1*S2
    Mat *= Const_dipdip0/(ar*ar*ar)
    return Mat


def AddSRDD(SH0, R0, dim=3):
    r"""
    Automatically adds short-range dipole-dipole interaction (SRDD) to the spin-Hamiltonian
    Rmax - cutoff distance of SRDD
    dim - system dimensionality
    """
    if R0<=0:
        return SH0
    Nx, Ny, Nz = calculateNdd(SH0, R0, dim=dim)
    SH = AddDD(SH0, NxMax=Nx, NyMax=Ny, NzMax=Nz, Rmax = R0)
    return SH


### adds Short-Range dipole-dipole interactrion to a spin Hamiltoniam
### NxMax - NzMax is the maximum distance in unit cell vectors
def AddDD(SH0, NxMax=3, NyMax=3, NzMax=0, exs=ex, eys=ey,ezs=ez, Rmax = None):
    r"""
    Addition of the short-range dipole-dipole interaction (SRDD) to the spin Hamiltonian
    SH0 - initial spin Hamiltonian
    NxMax, NyMax, NzMax - maximum displacements in unit cells required to calculate the interaction
    exs, eys, ezs - effective ex, ey and ez axes for spin-orbit interaction. In most of the cases should not be modified
    Rmax - the cutoff distance for SRDD
    """
    ###  exs, eys, ezs controls spin Coordinate frame (for now it is the same for all the spins, i.e. colinear state)
    ###
    ###
    SH = copy.deepcopy(SH0)
    SHref = copy.deepcopy(SH0)
    Nat = len(SH.magnetic_atoms)
    OldNotations = SH.notation
    SH.notation = (True,False,-1)    ### +double-counting, -normalization,  -1 multipliyer
    SHref.notation = (True,False,-1)
    for ia1 in range(Nat):
        at1 = SH.magnetic_atoms[ia1]
        r1 = ut2.realposition(SH, at1)  
        for ia2 in range(Nat):
            at2 = SH.magnetic_atoms[ia2]
            r2 = ut2.realposition(SH, at2)    
            for idx in range(-NxMax, NxMax+1):
                for idy in range(-NyMax, NyMax+1):
                    for idz in range(-NzMax, NzMax+1):
                        if not((idx==0) and (idy==0) and (idz==0) and (ia1==ia2)):
                            r = idx*SH.a1 + idy*SH.a2 + idz*SH.a3 + r2-r1
                            Rabs = np.linalg.norm(r)
                            if Rmax is None:
                                calc = True
                            else:
                                calc = Rabs<Rmax
                            if calc:
                                rnew = np.array((r@exs, r@eys, r@ezs))
                                #Jm = DDmatr(rnew, S1=at1.spin, S2=at2.spin) / 2   ## /2 due to double-counting
                                Jm = DDmatr(rnew) / 2   ## /2 due to double-counting
                                if ( (at1, at2, (idx,idy,idz)) in SHref ):
                                    Mold = SHref[at1, at2, (idx,idy,idz)].matrix
                                    Jm1 = Jm+Mold
                                    Jrad = rad.ExchangeParameter(matrix=Jm1)
                                    SH[at1, at2, (idx,idy,idz)] = Jrad
                                else:
                                    Jrad = rad.ExchangeParameter(matrix=Jm)
                                    SH[at1, at2, (idx,idy,idz)] = Jrad
    SH.notation = OldNotations
    return SH



####calculates the derivative of dipole energy matrix with respect to small modifications
##### of the vector rv along the direction eu
def DipDipDeriv(rv0, eu0, g1=2, g2=2, S1=1, S2=1, exs=ex, eys=ey,ezs=ez):
    r"""
    calculates the derivative of dipole energy matrix with respect to small modifications
    of the vector rv along the direction eu
    """
    rv = np.array((rv0@exs, rv0@eys, rv0@ezs))
    eu = np.array((eu0@exs, eu0@eys, eu0@ezs))
    ar = np.linalg.norm(rv)
    er = rv/ar

    pre1 = Const_dipdip0 / (ar**4)
    pre1 *= g1*g2 #*S1*S2

    mat1 = np.zeros((3,3))
    mat1 [0,...]= np.array([3*eu[0]*er[0], 3*eu[0]*er[1], 3*eu[0]*er[2]])
    mat1 [1,...]= np.array([3*eu[1]*er[0], 3*eu[1]*er[1], 3*eu[1]*er[2]])
    mat1 [2,...]= np.array([3*eu[2]*er[0], 3*eu[2]*er[1], 3*eu[2]*er[2]])
    mat1 *= pre1

    mat2 = np.zeros((3,3))
    mat2 [0,...]= np.array([3*er[0]*eu[0], 3*er[0]*eu[1], 3*er[0]*eu[2]])
    mat2 [1,...]= np.array([3*er[1]*eu[0], 3*er[1]*eu[1], 3*er[1]*eu[2]])
    mat2 [2,...]= np.array([3*er[2]*eu[0], 3*er[2]*eu[1], 3*er[2]*eu[2]])
    mat2 *= pre1

    mat3 = -2*np.eye(3)
    mat3 *= pre1* (er@eu)

    mat4 = DDmatr(rv, g1=g1, g2=g2)
    mat4 *= ( (-5) * (er@eu)/ar )
    return mat1 + mat2 + mat3 + mat4





