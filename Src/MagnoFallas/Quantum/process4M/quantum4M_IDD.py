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
import scipy as sp

import numba as nb
from numba.experimental import jitclass
from itertools import permutations

from MagnoFallas.Utils import util2 as ut
from MagnoFallas.Utils import DipoleDipole as dd
from MagnoFallas.Quantum import quantum_util as qut

eye1c = np.eye(3, dtype = np.complex128)


####
####  "Integrated dipole-dipole bond" 
####   Atom of ty1 is in the center, while atoms of ty2 are distributed with density 1/Vcell at r>R0
####   in 2D or 3D space
TIDDbond_lst = [
    ('ty1', nb.int32),       ##### "types" - numbers in SH.magnetic_atoms of interactin atoms
    ('ty2', nb.int32),
    ('dim', nb.int32),    #### dimension
    ('R0',  nb.float64),  #### minimum distance
########################################################
    ('R1T', nb.complex128[:,::1]),    ##### matrices required for spin rotation
    ('R2', nb.complex128[:,::1]),     ##### Jeff = R1T @ J @ R2
########################################################
    ### these values would be copy-pasted to the 4-magnon Hamiltonias
    ('A', nb.complex128[:,:,:]),    ### amplitudes for different components of J-matricies separately
    ('r', nb.float64[:,:]),       ### True (1) means typ2 (to be integrated), False (0) means central atom of typ1
    ('n', nb.int32[:,::1]),       ### number of operator acording to rad-tools order
########################################################
    ('Nlines', nb.int32)
]


##----------------------------------------------------------------
@jitclass(TIDDbond_lst)
class TIDDbond(object):
    def __init__(self, ty1,ty2, R0, Vuc, S1=3/2, S2=3/2, R1T =eye1c, R2 = eye1c, g1=2, g2=2, dim=3, Nat=2, inverse=True):
        self.ty1 = ty1
        self.ty2 = ty2
        self.R0 = R0
        self.dim = dim

        self.R1T = np.asarray(R1T, dtype=np.complex128)
        self.R2 =  np.asarray(R2, dtype=np.complex128)

        prefactor = g1*g2 * 0.5 / Vuc   #### 0.5 is due to the double conting     
        if inverse:                      #### most of the Hamiltonians, including dipole-dipole are written with positive ferromagnetic energies
            prefactor = -1*prefactor

        Nlines = 9
        self.Nlines = Nlines
        self.A = np.zeros((self.Nlines,3,3), dtype=np.complex128)
        self.r = np.zeros((self.Nlines, 4), dtype=np.float64)   
        self.n = np.zeros((self.Nlines, 4), dtype=np.int32)  ###number of operator acording to rad-tools order

############ The parts with 2 creation, 2 anihilation operators
        
        #self.A[0] = J[2,2]
        self.A[0,2,2] = 1
        #---------------------------
        self.r[0,0] = 0.0 #,r1
        self.r[0,1] = 1.0 #r2
        self.r[0,2] = 0.0 #r1
        self.r[0,3] = 1.0 #r2
        self.n[0,0] = ty1 + Nat
        self.n[0,1] = ty2 + Nat
        self.n[0,2] = ty1
        self.n[0,3] = ty2

        As = -1.0/8.0
        As *= np.sqrt(S2/S1)
        #-----------------------------
        #As *= (J[0,0] + J[1,1] + 1j*(J[0,1]-J[1,0]))
        #self.A[1] = As
        self.A[1,0,0] = As
        self.A[1,1,1] = As
        self.A[1,0,1] = 1j*As
        self.A[1,1,0] = -1j*As
        #-----------------------------
        self.r[1,0] = 0.0 #r1
        self.r[1,1] = 1.0 #r2
        self.r[1,2] = 0.0 #r1
        self.r[1,3] = 0.0 #r1
        self.n[1,0] = ty1 + Nat
        self.n[1,1] = ty2 + Nat
        self.n[1,2] = ty1
        self.n[1,3] = ty1


        As = -1.0/8.0
        As *= np.sqrt(S1/S2)
        #-----------------------------------
        #As *= (J[0,0] + J[1,1] + 1j*(J[0,1]-J[1,0]))
        #self.A[2] = As
        self.A[2,0,0] = As
        self.A[2,1,1] = As
        self.A[2,0,1] = 1j*As
        self.A[2,1,0] = -1j*As
        #----------------------------------
        self.r[2,0] = 1.0 #r2
        self.r[2,1] = 1.0 #r2
        self.r[2,2] = 0.0 #r1
        self.r[2,3] = 1.0 #r2
        self.n[2,0] = ty2 + Nat
        self.n[2,1] = ty2 + Nat
        self.n[2,2] = ty1
        self.n[2,3] = ty2

        As = -1.0/8.0
        As *= np.sqrt(S2/S1)
        #----------------------------------
        # As *= (J[0,0] + J[1,1] - 1j*(J[0,1]-J[1,0]))
        # self.A[3] = As
        self.A[3,0,0] = As
        self.A[3,1,1] = As
        self.A[3,0,1] = -1j*As
        self.A[3,1,0] = 1j*As
        #----------------------------------
        self.r[3,0] = 0.0 #r1
        self.r[3,1] = 0.0 #r1
        self.r[3,2] = 0.0 #r1
        self.r[3,3] = 1.0 #r2
        self.n[3,0] = ty1 + Nat
        self.n[3,1] = ty1 + Nat
        self.n[3,2] = ty1
        self.n[3,3] = ty2

        As = -1.0/8.0
        As *= np.sqrt(S1/S2)
        #----------------------------------
        # As *= (J[0,0] + J[1,1] - 1j*(J[0,1]-J[1,0]))
        # self.A[4] = As
        self.A[4,0,0] = As
        self.A[4,1,1] = As
        self.A[4,0,1] = -1j*As
        self.A[4,1,0] = 1j*As
        #----------------------------------
        self.r[4,0] = 0.0 #r1
        self.r[4,1] = 1.0 #r2
        self.r[4,2] = 1.0 #r2
        self.r[4,3] = 1.0 #r2
        self.n[4,0] = ty1 + Nat
        self.n[4,1] = ty2 + Nat
        self.n[4,2] = ty2
        self.n[4,3] = ty2

######################## The parts with odd numbers of creation/anihilation operators
        As = -1.0/8.0
        As *= np.sqrt(S2/S1)
        #----------------------------------
        # As *= (J[0,0] - J[1,1] - 1j*(J[0,1]+J[1,0]))
        # self.A[5] = As
        self.A[5,0,0] = As
        self.A[5,1,1] = -As
        self.A[5,0,1] = -1j*As
        self.A[5,1,0] = -1j*As
        #----------------------------------
        self.r[5,0] = 0.0 #r1
        self.r[5,1] = 0.0 #r1
        self.r[5,2] = 0.0 #r1
        self.r[5,3] = 1.0 #r2
        self.n[5,0] = ty1 + Nat
        self.n[5,1] = ty1
        self.n[5,2] = ty1
        self.n[5,3] = ty2

        As = -1.0/8.0
        As *= np.sqrt(S1/S2)
        #----------------------------------
        # As *= (J[0,0] - J[1,1] - 1j*(J[0,1]+J[1,0]))
        # self.A[6] = As
        self.A[6,0,0] = As
        self.A[6,1,1] = -As
        self.A[6,0,1] = -1j*As
        self.A[6,1,0] = -1j*As
        #----------------------------------
        self.r[6,0] = 1.0 #r2
        self.r[6,1] = 0.0 #r1
        self.r[6,2] = 1.0 #r2
        self.r[6,3] = 1.0 #r2
        self.n[6,0] = ty2 + Nat
        self.n[6,1] = ty1
        self.n[6,2] = ty2
        self.n[6,3] = ty2

        As = -1.0/8.0
        As *= np.sqrt(S2/S1)
        #----------------------------------
        # As *= (J[0,0] - J[1,1] + 1j*(J[0,1]+J[1,0]))
        # self.A[7] = As
        self.A[7,0,0] = As
        self.A[7,1,1] = -As
        self.A[7,0,1] = 1j*As
        self.A[7,1,0] = 1j*As
        #----------------------------------
        self.r[7,0] = 0.0 #r1
        self.r[7,1] = 0.0 #r1
        self.r[7,2] = 1.0 #r2
        self.r[7,3] = 0.0 #r1
        self.n[7,0] = ty1 + Nat
        self.n[7,1] = ty1 + Nat
        self.n[7,2] = ty2 + Nat
        self.n[7,3] = ty1

        As = -1.0/8.0
        As *= np.sqrt(S1/S2)
        #----------------------------------
        # As *= (J[0,0] - J[1,1] + 1j*(J[0,1]+J[1,0]))
        # self.A[8] = As
        self.A[8,0,0] = As
        self.A[8,1,1] = -As
        self.A[8,0,1] = 1j*As
        self.A[8,1,0] = 1j*As
        #----------------------------------
        self.r[8,0] = 0.0  #r1
        self.r[8,1] = 1.0  #r2
        self.r[8,2] = 1.0  #r2
        self.r[8,3] = 1.0  #r2
        self.n[8,0] = ty1 + Nat
        self.n[8,1] = ty2 + Nat
        self.n[8,2] = ty2 + Nat
        self.n[8,3] = ty2

        #----------------------------------
        self.A = self.A * prefactor
        #----------------------------------
###-----------------------------------------------------------




###-----------------------------------------------------------
## Convers rad-tools spin Hamiltonian into the typed list of IDD bonds

#### must be created from non-ferromagnetic Hamiltonian to correctly include spin rotatio matricies
def H4magnons_bonds_IDD(SH, R0, dim):
    ### important: bond vector is between atoms, not unit cells
    Nat = len(SH.magnetic_atoms)
    Vuc = ut.cellVolume(SH.cell, (dim==2) )
    AtInd = {}
    for i,at in enumerate(SH.magnetic_atoms):
        AtInd[at] = i
    
    bond_list = nb.typed.List()
    for at1 in SH.magnetic_atoms:
        for at2 in SH.magnetic_atoms:
            ty1 = AtInd[at1]
            ty2 = AtInd[at2]
            R1 = (1+0j)*qut.vec_to_RM(at1.spin_vector)
            R1T = np.transpose(R1).copy()
            R2 = qut.vec_to_RM(at1.spin_vector)
            R2 = (1+0j)*R2
            bndIDD = TIDDbond(ty1,ty2, R0, Vuc, S1 = at1.spin, S2=at2.spin, R1T =R1T, R2 = R2, dim=dim, Nat=Nat)
            bond_list.append(bndIDD)
            
    return bond_list
###-----------------------------------------------------------
####################################################################################


######### class to store a (single term=line of) 4-magnon Hamiltonian 
######### reimagined for the long-range (integrated) dipole-dipole coupling
TIDDline_lst = [
    ('A', nb.complex128[:,:]),
    ('r', nb.float64[::1]),
    ('n', nb.int32[::1]),  
    ('R0', nb.float64),
    ('dim', nb.int32),
    ('R1T', nb.complex128[:,::1]),    ##### matrices required for spin rotation
    ('R2', nb.complex128[:,::1])     ##### Jeff = R1T @ J @ R2
]

@jitclass(TIDDline_lst)
class TlineIDD(object):
    def __init__(self, A,r,n,R0, dim, R1T, R2):
        self.A = A
        self.r = r
        self.n = n
        self.R0 = R0
        self.dim = dim
        self.R1T = R1T
        self.R2 = R2
#######################################################


###----------------------------------------------------------------------------
##### Takes the "bond" object and writes down (adds) all the information to the list of lines in the 
####   corresponding format
###   we write not only the "normal lines" but also the lines with all the permutations of magnon operators
##    while it is not physical, it is useful to calculate Boltzmann matrix elements
def AddPermutedLinesIDD(lst, bond):
    for iln in range(bond.Nlines): 
        for p in permutations(range(4)):
            Ax = bond.A[iln]
            rx = np.array( (bond.r[iln][p[0]],bond.r[iln][p[1]],bond.r[iln][p[2]],bond.r[iln][p[3]]) )
            nx = np.array( (bond.n[iln][p[0]],bond.n[iln][p[1]],bond.n[iln][p[2]],bond.n[iln][p[3]]) )
            line = TlineIDD(Ax, rx, nx, bond.R0, bond.dim, bond.R1T, bond.R2)
            lst.append(line)
###------------------------------------------------------------------------------

####-------------------------------------------------------------------------------
### converts a list of bonds into a list of lines
### permutations included
def PermutedLinesIDD(H4bonds):
    lst = nb.typed.List()
    for bnd in H4bonds:
        AddPermutedLinesIDD(lst, bnd)
    return lst
####-------------------------------------------------------------------------------


####-------------------------------------------------------------------------------
### main procedure creating "line-type" terms for 4-magnon Hamiltonian
### representing LR dipole-dipole exchange interaction
### must be created from the initial spin Hamiltonian without FM-rotation
def Create_IDD_lines(SH, R0, dim=2):
    BondList = H4magnons_bonds_IDD(SH, R0, dim)  
    LineList = PermutedLinesIDD(BondList)
    return LineList
####-------------------------------------------------------------------------------


#####################################################################################################
#### the following part is for the calculation of matrix elements
################################################################################################


@nb.jit(nopython=True)
def lineMelIDD(kx4, lamx4, Gix4, line,  Nat, roles):
    r"""
    The order is: k0,k1 - out; k2,k3 - in
    lamx4: lam0, lam1, lam2, lam3 - magnon branches
    Gix4: Gi[k0], Gi[k1], Gi[-k3], Gi[-k4]
    all the matricies Gi should be modified according to ut.MogGi
    """
    Amat = line.A

    kef = -roles[0]*line.r[0]*kx4[0] - roles[1]*line.r[1]* kx4[1] - roles[2]*line.r[2]*kx4[2] - roles[3]*line.r[3]*kx4[3]
    Jmat = dd.longrangeDDmatr(kef, line.R0, line.dim, g1=1, g2=1)  
    Jeff = line.R1T@Jmat@line.R2
    A = np.sum(Amat*Jeff)
    
    pG = 1.0
    for ima in range(4):
        if roles[ima] == -1:
            noper = lamx4[ima]
        else:
            noper = 2*Nat -1 - lamx4[ima]
        pG *= Gix4[ima][line.n[ima], noper]

    return A*pG

@nb.jit(nopython=True)
def fullMelIDD(kx4, lamx4, Gix4, lineList, Nat, roles):
    r"""
    lamx4: lam0, lam1, lam2, lam3 - magnon branches
    Gix4: Gi[k0], Gi[k1], Gi[-k3], Gi[-k4]
    all the matricies Gi should be modified according to ut.MogGi
    """
    M = 0.0
    for l in lineList:
        M += lineMelIDD(kx4, lamx4, Gix4, l, Nat, roles)
    return M

