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
Module for the matrix elements of 4-magnon intewraction
"""

import numpy as np
import numpy.linalg
import scipy as sp
import matplotlib.pyplot as plt
import copy

import numba as nb
from numba.experimental import jitclass
from itertools import permutations

from MagnoFallas.Utils import util2 as ut
from MagnoFallas.Quantum import quantum_util as qut


######### class for magnon-magono interpretation of a single bond in spin-Hamiltonian
#### serves as an intermediate state for creation of whole 4-magnon Hamiltonian
Tbond_lst = [
    ('r1', nb.float64[::1]),              ### positions of interaction atoms
    ('r2', nb.float64[::1]),
    ('ty1', nb.int32),       ##### "types" - numbers in SH.magnetic_atoms of interactin atoms
    ('ty2', nb.int32),
    ('Jmat', nb.float64[:,::1]),              ## matrix of the exchange interaction
########################################################
    ### these values would be copy-pasted to the 4-magnon Hamiltonias
    ('A', nb.complex128[::1]),    ### amplitudes of the future Hamiltonian lines
    ('r', nb.float64[:,:,:]),     ### participaing atoms position (absolute unite, relative to the zero in the unit cell (0,0,0))
    ('n', nb.int32[:,::1]),       ### number of operator acording to rad-tools order
########################################################
    ('Nlines', nb.int32)
]


##----------------------------------------------------------------
@jitclass(Tbond_lst)
class Tbond(object):
    def __init__(self, r1,r2,ty1,ty2, J0, S1=3/2, S2=3/2, Nat=2, inverse=True):
        self.r1 = np.asarray(r1, dtype=np.float64)
        self.r2 = np.asarray(r2, dtype=np.float64)
        self.ty1 = ty1
        self.ty2 = ty2
        ##### usually we deal in the notations where FM interaction is positive
        ##### latter expressions are written in the notations where FM interaction is negative
        ##### like in hardcore theory. We use inverse for this
        if inverse:    
            J = -J0
        else:
            J = J0
        
        self.Jmat = np.asarray(J, dtype=np.float64)

        Nlines = 9
        self.Nlines = Nlines
        self.A = np.zeros(self.Nlines, dtype=np.complex128)
        self.r = np.zeros((self.Nlines, 4, 3), dtype=np.float64)
        self.n = np.zeros((self.Nlines, 4), dtype=np.int32)  ###number of operator acording to rad-tools order

############ The parts with 2 creation, 2 anihilation operators
        
        self.A[0] = J[2,2]
        self.r[0,0] = r1
        self.r[0,1] = r2
        self.r[0,2] = r1
        self.r[0,3] = r2
        self.n[0,0] = ty1 + Nat
        self.n[0,1] = ty2 + Nat
        self.n[0,2] = ty1
        self.n[0,3] = ty2

        As = -1.0/8.0
        As *= np.sqrt(S2/S1)
        As *= (J[0,0] + J[1,1] + 1j*(J[0,1]-J[1,0]))
        self.A[1] = As
        self.r[1,0] = r1
        self.r[1,1] = r2
        self.r[1,2] = r1
        self.r[1,3] = r1
        self.n[1,0] = ty1 + Nat
        self.n[1,1] = ty2 + Nat
        self.n[1,2] = ty1
        self.n[1,3] = ty1


        As = -1.0/8.0
        As *= np.sqrt(S1/S2)
        As *= (J[0,0] + J[1,1] + 1j*(J[0,1]-J[1,0]))
        self.A[2] = As
        self.r[2,0] = r2
        self.r[2,1] = r2
        self.r[2,2] = r1
        self.r[2,3] = r2
        self.n[2,0] = ty2 + Nat
        self.n[2,1] = ty2 + Nat
        self.n[2,2] = ty1
        self.n[2,3] = ty2

        As = -1.0/8.0
        As *= np.sqrt(S2/S1)
        As *= (J[0,0] + J[1,1] - 1j*(J[0,1]-J[1,0]))
        self.A[3] = As
        self.r[3,0] = r1
        self.r[3,1] = r1
        self.r[3,2] = r1
        self.r[3,3] = r2
        self.n[3,0] = ty1 + Nat
        self.n[3,1] = ty1 + Nat
        self.n[3,2] = ty1
        self.n[3,3] = ty2

        As = -1.0/8.0
        As *= np.sqrt(S1/S2)
        As *= (J[0,0] + J[1,1] - 1j*(J[0,1]-J[1,0]))
        self.A[4] = As
        self.r[4,0] = r1
        self.r[4,1] = r2
        self.r[4,2] = r2
        self.r[4,3] = r2
        self.n[4,0] = ty1 + Nat
        self.n[4,1] = ty2 + Nat
        self.n[4,2] = ty2
        self.n[4,3] = ty2

######################## The parts with odd numbers of creation/anihilation operators
        As = -1.0/8.0
        As *= np.sqrt(S2/S1)
        As *= (J[0,0] - J[1,1] - 1j*(J[0,1]+J[1,0]))
        self.A[5] = As
        self.r[5,0] = r1
        self.r[5,1] = r1
        self.r[5,2] = r1
        self.r[5,3] = r2
        self.n[5,0] = ty1 + Nat
        self.n[5,1] = ty1
        self.n[5,2] = ty1
        self.n[5,3] = ty2

        As = -1.0/8.0
        As *= np.sqrt(S1/S2)
        As *= (J[0,0] - J[1,1] - 1j*(J[0,1]+J[1,0]))
        self.A[6] = As
        self.r[6,0] = r2
        self.r[6,1] = r1
        self.r[6,2] = r2
        self.r[6,3] = r2
        self.n[6,0] = ty2 + Nat
        self.n[6,1] = ty1
        self.n[6,2] = ty2
        self.n[6,3] = ty2

        As = -1.0/8.0
        As *= np.sqrt(S2/S1)
        As *= (J[0,0] - J[1,1] + 1j*(J[0,1]+J[1,0]))
        self.A[7] = As
        self.r[7,0] = r1
        self.r[7,1] = r1
        self.r[7,2] = r2
        self.r[7,3] = r1
        self.n[7,0] = ty1 + Nat
        self.n[7,1] = ty1 + Nat
        self.n[7,2] = ty2 + Nat
        self.n[7,3] = ty1

        As = -1.0/8.0
        As *= np.sqrt(S1/S2)
        As *= (J[0,0] - J[1,1] + 1j*(J[0,1]+J[1,0]))
        self.A[8] = As
        self.r[8,0] = r1
        self.r[8,1] = r2
        self.r[8,2] = r2
        self.r[8,3] = r2
        self.n[8,0] = ty1 + Nat
        self.n[8,1] = ty2 + Nat
        self.n[8,2] = ty2 + Nat
        self.n[8,3] = ty2
###-----------------------------------------------------------


###-----------------------------------------------------------
## Convers rad-tools spin Hamiltonian into the typed list of bonds

def H4magnons_bonds(SH0, RemoveIso = False):
    ### important: bond vector is between atoms, not unit cells
    ### "normal" magnon bonds are constructed from "rotated-ferromagnetic" spin Hamiltonian
    SH = qut.make_FerroHam(SH0)  ## rotation to "ferromagnetic notations is made here"
    Nat = len(SH.magnetic_atoms)
    AtInd = {}
    for i,at in enumerate(SH.magnetic_atoms):
        AtInd[at] = i
    
    bond_list = nb.typed.List()
    for at1, at2, (i, j, k), J in SH:
        r1 = ut.realposition(SH, at1)
        drcell = 1.0*i*SH.a1 + 1.0*j*SH.a2 + 1.0*k*SH.a3
        r2 = ut.realposition(SH, at2) + drcell
        ty1 = AtInd[at1]
        ty2 = AtInd[at2]
        J1 = J.matrix.copy() 
        if RemoveIso:
            J1-= J.iso*np.eye(3)
        bnd = Tbond(r1,r2,ty1,ty2, J1, S1 = at1.spin, S2=at2.spin, Nat=Nat)
        bond_list.append(bnd)
    return bond_list
###-----------------------------------------------------------
####################################################################################



######### class to store a (single term=line of) 4-magnon Hamiltonian 
####   each line has Value (amplitude A) types of creation/anihilation operators n
####    and the positions in real space r corresponding to these operators
Tline_lst = [
    ('A', nb.complex128),
    ('r', nb.float64[:,::1]),
    ('n', nb.int32[::1]),  
]

@jitclass(Tline_lst)
class Tline(object):
    def __init__(self, A,r,n):
        self.A = A
        self.r = r
        self.n = n
#######################################################




###----------------------------------------------------------------------------
##### Takes the "bond" object and writes down (adds) all the information to the list of lines in the 
#### corresponding format
### we write not only the "normal lines" but also the lines with all the permutations of magnon operators
## while it is not physical, it is useful to calculate Boltzmann matrix elements
def AddPermutedLines(lst, bond):
    for iln in range(bond.Nlines): 
        for p in permutations(range(4)):
            Ax = bond.A[iln]
            rx = np.array( (bond.r[iln][p[0]],bond.r[iln][p[1]],bond.r[iln][p[2]],bond.r[iln][p[3]]) )
            nx = np.array( (bond.n[iln][p[0]],bond.n[iln][p[1]],bond.n[iln][p[2]],bond.n[iln][p[3]]) )
            line = Tline(Ax, rx, nx)
            lst.append(line)
###------------------------------------------------------------------------------



####-------------------------------------------------------------------------------
### converts a list of bonds into a list of lines
### permutations included
def PermutedLines(H4bonds):
    lst = nb.typed.List()
    for bnd in H4bonds:
        AddPermutedLines(lst, bnd)
    return lst
####-------------------------------------------------------------------------------



####-------------------------------------------------------------------------------
#### tekes rad-tools Hamiltonian SH and created 4-magnon Hamiltonian
### permutations included
def PermutedMagnon4_Ham(SH0):
    r"""
    construct quantum "magnon" Hamiltonian
    from the rad-tools Hamiltonian SH

    it is not nessecery for SH0 to be converted to "ferromagnetic notation"
    as the conversion is made within this module
    """
    SH = copy.deepcopy(SH0)
    SH.notation = (False, False, -1)
    BondList = H4magnons_bonds(SH)
    LineList = PermutedLines(BondList)
    return LineList
####-------------------------------------------------------------------------------


#####################################################################################################
#### the following part is for the calculation of matrix elements
################################################################################################


@nb.jit(nopython=True)
def lineMel(kx4, lamx4, Gix4, line,  Nat, roles):
    ### The order is: k0,k1 - out; k2,k3 - in
    ### lamx4: lam0, lam1, lam2, lam3 - magnon branches
    ### Gix4: Gi[k0], Gi[k1], Gi[-k3], Gi[-k4]
    ### all the matricies Gi should be modified according to ut.MogGi
    A = line.A
    ex = -roles[0]*kx4[0]@line.r[0] - roles[1]* kx4[1]@line.r[1] - roles[2]*kx4[2]@line.r[2] - roles[3]*kx4[3]@line.r[3]
    ex = np.exp(1j*ex)

    pG = 1.0
    for ima in range(4):
        if roles[ima] == -1:
            noper = lamx4[ima]
        else:
            noper = 2*Nat -1 - lamx4[ima]
        pG *= Gix4[ima][line.n[ima], noper]

    return A*ex*pG

@nb.jit(nopython=True)
def fullMel(kx4, lamx4, Gix4, lineList, Nat, roles):
    ### lamx4: lam0, lam1, lam2, lam3 - magnon branches
    ### Gix4: Gi[k0], Gi[k1], Gi[-k3], Gi[-k4]
    ### all the matricies Gi should be modified according to ut.MogGi
    M = 0.0
    for l in lineList:
        M += lineMel(kx4, lamx4, Gix4, l, Nat, roles)
    return M


