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
utilitary functions for spin-phonon interaction
"""

import numpy as np
import numpy.linalg
import scipy as sp
import numbers

import numba as nb
from numba.experimental import jitclass

import MagnoFallas.OldRadtools as rad
from MagnoFallas.Utils import util2 as ut2



### phonon-x-constnt = 10^10 [\hbar^2 / 2 * amu  * 1meV ]^{1/2}    (result is in angstr)
phonon_x_constant = 1.4457107723636267

ex = np.array((1.,0,0))
ey = np.array((0,1.,0))
ez = np.array((0,0,1.))



############# base class describing the modification of exchange interaction in spin notations ######################

ModificationJlist =  [
    ('i1', nb.int32),  ##### participating spins (numbers inside unit cell)
    ('i2', nb.int32),  
    ('cvecJ', nb.int32[::1]),  ### vector between unit cells of participating atoms
    ('n', nb.int32),     #### displaced atom (number in unit cell of phonon calculation)
    ('edis', nb.float64[::1]),  ###unit vector of the displacement
    ('cvecN', nb.int32[::1]),  
    ('dJ', nb.complex128[:,::1])  ### derivation of the exchange interaction matrix
]

@jitclass(ModificationJlist)
class TderivJ(object):
    def __init__(self, i1, i2, cvecJ, n, edis, cvecN, dJ):
        self.i1 = i1
        self.i2 = i2
        self.cvecJ = cvecJ
        self.n = n
        self.edis = edis
        self.cvecN = cvecN
        self.dJ = dJ

def EmptydJList():
    i1 = 0 ; i2 = 0
    cvec = np.array((0,0,0), dtype = np.int32)
    n= 0 
    dJ = np.eye(3, dtype=np.complex128)
    test0 = TderivJ(i1,i2,cvec,n,ex,cvec,dJ)
    lis = nb.typed.List()
    lis.append(test0)
    lis.clear()
    return lis

########################################################################################################################


##calculates real position of the atoms if relative ones are given inside spin Hamiltonian SH
# def realposition(SH, at):
#     vec = np.zeros(3)
#     vec += at.position[0]*SH.a1
#     vec += at.position[1]*SH.a2
#     vec += at.position[2]*SH.a3
#     return vec

### reade TB2J results, assigns spins to magnetic atoms
### and calculates the real positions
def SHreadTB2J(str, sp0=3/2):
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


###------ procedures to relate TB2J and Phonopy

#### searches in the phon.primitive an atom with symbol Sy and position pos
def findatom_Phonopy(phon, Sy, pos, d_max=0.5, dim=2):
    N = -10000
    Nuc = len(phon.primitive.symbols)
    for i in range(Nuc):
        if phon.primitive.symbols[i] == Sy:
            pos1 = phon.primitive.positions[i]
            d = ut2.find_distance(pos,pos1,phon.primitive.cell, ignoreZ = (dim==2))
            #print(d)
            if d<d_max:
                #print(pos,pos1)
                N = i
    return N

#### relate SH from TB2J to Phonopy object phonon
def relate_TB2J_Phonopy(SH, phonon, dim, name_map = None):
    r"""
    relate SH from TB2J to Phonopy object phonon
    dim - system dymension
    name_map - allows to map the ion names from TB2J to phonopy
    """
    phDict = {}
    for at in SH.atoms:
        Sy0 = ut2.removeDigits(at.name)
        if name_map is None:
            Sy = Sy0
        else:
            Sy = name_map[Sy0]
        pos1 = ut2.realposition(SH, at)
        #print(Sy, pos1)
        N1 = findatom_Phonopy(phonon, Sy, pos1, d_max=0.5, dim=dim)
        phDict[at.name] = N1
    liAt = []
    liMag = []
    for at in SH.atoms:    
        liAt.append(phDict[at.name])
    for at in SH.magnetic_atoms:    
        liMag.append(phDict[at.name])
    return liMag, liAt








