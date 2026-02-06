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
Set of general tools to play with spin Hamiltonians
"""

import numbers
import copy

import numpy as np
import numpy.linalg
import numba as nb

import MagnoFallas.OldRadtools as rad
from MagnoFallas.Utils import util2 as ut2

def SHreadTB2J(str, sp0=3/2):
    r"""
    reade TB2J results, assigns spins to magnetic atoms
    and calculates the real positions
    """
    SH = rad.load_tb2j_model(str, quiet=True, standardize=False)
    realpos = []

    is1S = isinstance(sp0, numbers.Number)
    
    for i,at in enumerate(SH.magnetic_atoms):
        if is1S:
            SH.magnetic_atoms[i].spin = sp0
        else:
            SH.magnetic_atoms[i].spin = sp0[i]
        realpos.append( ut2.realposition(SH, at)  )
    realpos = np.array( realpos )
    SH.notation = (False,False,-1)
    return SH, realpos

##-------------------------------------------------------------------------------------------------------

def lookForAtom(SH, rco, prec=1e-3):
    r"""
    tries to find an atom in spin Hamiltoniam "SH" (atoms)
    based on its Cartesian coordinates "rco"
    prec - the precision
    """
    found = False
    name = ''
    for at in SH.atoms:
        rc1 = ut2.realposition2(at.position, SH.cell) 
        if ut2.EqualVectors(rc1, rco, prec=prec):
            found = True
            name = at.name
    return found, name


def lookForMagAtom(SH, rco, prec=1e-3):
    r"""
    tries to find an atom in spin Hamiltoniam "SH" (magnetic_atoms)
    based on its Cartesian coordinates "rco"
    prec - the precision
    """
    found = False
    name = ''
    for at in SH.magnetic_atoms:
        rc1 = ut2.realposition2(at.position, SH.cell) 
        if ut2.EqualVectors(rc1, rco, prec=prec):
            found = True
            name = at.name
    return found, name
    

##-------------------------------------------------------------------------------------------------------



def project(SH0, a1,a2, a3=np.zeros(3), dim=2, zeroShift=np.zeros(3), MaxN = 3, MaxNz=-1, prec=1e-1,
           relPrecision = 0.02, absPrecision=0.03, quite=False, returnMap=False):
    r"""
    projects the spin Hamiltonian "SH0" to the new unit cell defined by the unit vectors "a1", "a2", "a3" 
    (in Cratesian coordinates corresponding to SH0)
    dim = system dimension (for dim==2, a3 and MaxNz are not used)
    MaxN - maximum displacement in unit vectors in SH0 to check
    prec --- precision for the atom coordinates (in Angstr)
    relPrecision, absPrecision --- relative and absolute precision for exchange interactions
    quite --- wether to print erroe warnings
    returnMap --- wether to return the mapping between spin Hamiltonians
    """

    def compareJ(J1, J2):
        good = True
        mabs = np.max(np.abs(J1.matrix - J2.matrix))
        if mabs>absPrecision:
            good = False
        if np.abs(J2.iso) > ut2.Global_zero:
            rel = np.abs( (J1.iso - J2.iso)/(J2.iso) )
            if rel>relPrecision:
                good = False
        return good
    
    if dim==2:
        a3x = np.cross(a1,a2)
    else:
        a3x = a3

    lNx, lNy, lNz = MaxN, MaxN, MaxN
    if dim == 2:
        lNz = 0
    elif MaxNz>0:
        lNz = MaxNz

    map = {}
    ats = {}
    
    pcell = np.array((a1,a2,a3x))
    SHp = rad.SpinHamiltonian(cell=pcell, standardize=False)
    SHp.notation = SH0.notation

    zCo0 = SH0.magnetic_atoms[0].position + zeroShift
    for iat, at in enumerate(SH0.magnetic_atoms):
        co0 = ut2.realposition2(at.position, SH0.cell)  + ut2.Global_zero + zeroShift
        for idx in range(-lNx, lNx+1):
            for idy in range(-lNy, lNy+1):
                for idz in range(-lNz, lNz+1):
                    co1 = co0 + idx*SH0.a1 + idy*SH0.a2 + idz*SH0.a3
                    co2, dv = ut2.PutIntoCell2(pcell, co1)
                    found, newname = lookForAtom(SHp, co2, prec=prec)
                    if found:
                        map[(at.name, (idx, idy,idz))] = (newname, dv)
                    else:
                        type = ut2.removeDigits(at.name)
                        if type in ats.keys():
                            ats[type] += 1
                        else:
                            ats[type] = 1
                        newname = type + str(ats[type])
                        svec= at.spin_vector
                        newpos = ut2.RelativePosition(pcell, co2)
                        newAt = rad.Atom(newname, spin=svec, position=newpos)
                        SHp.add_atom(newAt)
                        map[(at.name, (idx, idy,idz))] = (newname, dv)

    for at1, at2, dv0, J in SH0:
        for idx in range(-lNx, lNx+1):
            for idy in range(-lNy, lNy+1):
                for idz in range(-lNz, lNz+1):
                    dvX0 = (idx, idy, idz)
                    dvY0 = (idx + dv0[0], idy+dv0[1], idz+dv0[2])
                    exist1 = ((at1.name, dvX0) in map) and ((at2.name, dvY0) in map)
                    if exist1:
                        newN1, ndv1 = map[at1.name, dvX0]
                        newN2, ndv2 = map[at2.name, dvY0]
                        dv1 = ndv2 - ndv1
                        
                        exist2 = (newN1, newN2, tuple(dv1)) in SHp
                        if exist2:
                            Jold = SHp[newN1, newN2, tuple(dv1)]
                            good = compareJ(J, Jold)
                            if (not good) and (not quite):
                                print('disagreemnt exchange :', newN1, ', ', newN2, ', ',dv1)
                                print('J1 = ', Jold.matrix)
                                print('J2 = ', J.matrix)
                                print('--------------------------')
                        else:
                            SHp.add_bond(newN1, newN2, tuple(dv1), matrix=J.matrix) 
        
    if returnMap:
        return SHp, map
    else:
        return SHp   


##-------------------------------------------------------------------------------------------------------

def rotateSH(SH0, basis):
    r"""
    rotates spin Hamiltonian in physical space, bringing it to the new basis (new x,y and z axes)
    SH0 - old spin Hamiltonain
    basis - 3x3 array of axes (ex,ey,ez) must be orthonormal vectors
    Comment: because all the information in Spin Hamiltonian is in relative units, only changes the unit cell
    of spin Hamiltonian
    """
    ex1 = basis[0]
    ey1 = basis[1]
    ez1 = basis[2]
    a1 = np.array([ SH0.a1 @ ex1,  SH0.a1 @ ey1, SH0.a1 @ ez1  ])
    a2 = np.array([ SH0.a2 @ ex1,  SH0.a2 @ ey1, SH0.a2 @ ez1  ])
    a3 = np.array([ SH0.a3 @ ex1,  SH0.a3 @ ey1, SH0.a3 @ ez1  ])
    cell1 = np.array((a1,a2,a3))
    SH = ut2.cloneSH(SH0, cell1)
    return SH




