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
A strategy that assumes that bonds are rotated as whole due to TA-phonons
"""

import numpy as np
import numpy.linalg
import scipy as sp
import numbers

import numba as nb
from numba.experimental import jitclass

import MagnoFallas.OldRadtools as rad
from MagnoFallas.SpinPhonon import SPhUtil as sphut




#### base function: determine the exchange interaction based on a spin hamiltonian and its rotations
#### and than calculate the dJ/dr lines of spin-displacement Hamiltonian
def Estimate_auto_rot(SH, pos, mag_correspond_list, lst=None, sp0=3/2,  disMax = 5, dPhi=0.01):
    l_i1, l_i2, l_cvec = formLists(SH, pos, disMax = disMax)
    HlineLst = Estimate_bonds_rot(l_i1, l_i2, l_cvec, SH, pos, mag_correspond_list, lst=lst, sp0=sp0,  dPhi = dPhi)
    return HlineLst

### finds the directions perpendicular to a vector v
def normaldirections(v, dmin=0.01):
    ex = np.array((1,0,0))
    ez = np.array((0,0,1))
    v1 = np.cross(v,ez)
    if np.linalg.norm(v1) < dmin:
        v1 = np.cross(v,ex)
    ev1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(v,ev1)
    ev2 = v2 / np.linalg.norm(v2)
    return ev1,ev2


###### calculate the lists l_i1, l_i2, l_cvec from spin Hamiltonian and real positions 
###### with the maximum distance equal to disMax
def formLists(SH0, pos, disMax = 5):
    l_i1 = []
    l_i2 = []
    l_cvec = []
    numD = {}
    for iat, at in enumerate(SH0.magnetic_atoms):
        numD[at.name] = iat
    for at1,at2, cvec, Jint in SH0:
        i1 = numD[at1.name]
        i2 = numD[at2.name]
        vec = cvec[0]*SH0.a1 + cvec[1]*SH0.a2 + cvec[2]*SH0.a3
        vec += pos[i2]-pos[i1]
        dis = np.linalg.norm(vec)
        if dis < disMax:
            l_i1.append(i1)
            l_i2.append(i2)
            l_cvec.append(cvec)
    return l_i1, l_i2, l_cvec



### caclulates the derivative due to rotation of a single bond
### Jm is the matrix of exchange interactions
### rJ is the bond vector
### eu is atomic displacemnt direction
def dJmatrRot(Jm, rJ, eu, dPhi = 0.01, rmin = 0.01):
    arJ = np.linalg.norm(rJ)
    rotV = np.cross(rJ,eu)
    arotV = np.linalg.norm(rotV)
    if arotV < rmin:
        return np.zeros((3,3))

    rotV2 = dPhi * rotV / arotV
    Rrot1 = sp.spatial.transform.Rotation.from_rotvec(rotV2)
    T = Rrot1.as_matrix()
    Tt = np.transpose(T)
    dJ = (T@Jm@Tt - Jm)/dPhi
    dJ /= arJ
    return dJ


###### Gets list of dJ/dr based on the lists "l_i1, l_i2, l_cvecJ" describing the desired exchange interactions
def Estimate_bonds_rot(l_i1, l_i2, l_cvecJ, SH, pos, mag_correspond_list, lst=None, sp0=3/2, dPhi = 0.01):
    ###    l_i1, l_i2, l_cvecJ : lists describing bonds to study i1,i2 - numbers in magnetic_atoms, cvec - distance in unit cells
    ###    mag_correspond_lis - correspondence list between magnetic_atoms and results of phonopy
    ###    lst - list of already calculated terms (usually should be left "None")
    ###    sp0 - spins of magnetic atoms (single value or list)
    ###    dPhi - rotation angle used to obtain derivatives

    if lst==None:
        lst = sphut.EmptydJList()
        
    cvec0 = np.array((0,0,0), dtype = np.int32)

    #preliminary calculations with zero-Hamiltonian
    Ncalc = len(l_i1)
    mat0 = SH.magnetic_atoms
    J0matr = [SH[mat0[l_i1[i]],mat0[l_i2[i]], l_cvecJ[i]].matrix  for i in range(Ncalc)   ]
    evec0 = np.zeros((Ncalc,3))
    vecs0 = np.zeros((Ncalc,3))
    for i in range(Ncalc):
        po0 = pos[l_i1[i]]
        po1 = pos[l_i2[i]]
        cvec = l_cvecJ[i]
        vec = cvec[0]*SH.a1 + cvec[1]*SH.a2 + cvec[2]*SH.a3
        vec += po1-po0
        vecs0[i] = vec
        vecN = vec / np.linalg.norm(vec)
        evec0[i] = vecN

    
    for ica in range(Ncalc):
        i1 = l_i1[ica]
        i2 = l_i2[ica]
        n1 = mag_correspond_list[i1]
        n2 = mag_correspond_list[i2]
        e1,e2 = normaldirections(evec0[i])
        cvec = l_cvecJ[ica]
        
        drJ1 = dJmatrRot(J0matr[i], vecs0[i], e1).astype(complex)
        drJ2 = dJmatrRot(J0matr[i], vecs0[i], e2).astype(complex)
        
        dJline11 = sphut.TderivJ(i1,i2, np.array(cvec), n2, e1.copy(), np.array(cvec), drJ1)
        dJline12 = sphut.TderivJ(i1,i2, np.array(cvec), n1, e1.copy(), np.array(cvec0), -drJ1)
        
        dJline21 = sphut.TderivJ(i1,i2, np.array(cvec), n2, e2.copy(), np.array(cvec), drJ2)
        dJline22 = sphut.TderivJ(i1,i2, np.array(cvec), n1, e2.copy(), np.array(cvec0), -drJ2)

        lst.append(dJline11)
        lst.append(dJline12)
        lst.append(dJline21)
        lst.append(dJline22)
        
    return lst





