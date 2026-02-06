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
Module containing Boltzmann-related physics for 4-magnon processes
Not supposed to interact with user
"""



import numpy as np
import numpy.linalg
import numba as nb


import MagnoFallas.OldRadtools as rad

from MagnoFallas.Interface import PseudoRad as prad
from MagnoFallas.Utils import util2 as ut2

from MagnoFallas.Quantum.process4M import quantum4M as quant
from MagnoFallas.Quantum.process4M import quantum4M_IDD as quant4dd
from MagnoFallas.Boltzmann.process4M import ScatteringList4M as slist

from MagnoFallas.Interface import PseudoRad as prad






def ScatToMel(sc, pSH, Hlines, pos, LR=False, LRlines=None):
    r"""
    Initiates matrix element of the scatterer sc
    pSH - pseudo-rad-tools spin Hamiltonian
    Hlines - "lines" of quantum 4-magnon Hamiltonian
    pos - "real" positions of atoms [in A]
    """
    NMat = pSH.Nat
    kx4 = (sc.k0,sc.k1, sc.k2real,sc.k3real)
    lamx4 = (sc.lk0,sc.lk1, sc.lk2,sc.lk3)
    
    o0,G0,Gi0_0 = prad.omega(pSH, -sc.roles[0]*sc.k0)
    o1,G1,Gi1_0 = prad.omega(pSH, -sc.roles[1]*sc.k1)
    o2,G2,Gi2_0 = prad.omega(pSH, -sc.roles[2]*sc.k2real)
    o3,G3,Gi3_0 = prad.omega(pSH, -sc.roles[3]*sc.k3real)
    Gi0 = ut2.ModGi(Gi0_0, -sc.roles[0]*sc.k0, pos) [0]
    Gi1 = ut2.ModGi(Gi1_0, -sc.roles[1]*sc.k1, pos)[0]
    Gi2 = ut2.ModGi(Gi2_0, -sc.roles[2]*sc.k2real, pos)[0]
    Gi3 = ut2.ModGi(Gi3_0, -sc.roles[3]*sc.k3real, pos)[0]
    Gix4 = (Gi0,Gi1,Gi2,Gi3)

    mel = quant.fullMel(kx4, lamx4, Gix4, Hlines, NMat, sc.roles)

    if LR:
        if LRlines is None:
            print('Incorrect activation of LR dipole-dipole interaction')
            mel2 = 0.0
        else:
            mel2 = quant4dd.fullMelIDD(kx4, lamx4, Gix4, LRlines, NMat, sc.roles)
        mel = mel + mel2
    
    sc.Mel = mel
    return mel

def InitiateMel_List(lst, pSH, Hlines, pos, LR=False, LRlines=None):
    for sc in lst:
        ScatToMel(sc, pSH, Hlines, pos, LR=LR, LRlines=LRlines)
        

def InitiateMel_SetList(setlst, pSH, Hlines, pos, LR=False, LRlines=None):
    for k,lst in setlst.items():
        InitiateMel_List(lst, pSH, Hlines, pos, LR=LR, LRlines=LRlines)


#### note: Gilbert relaxation requires roles[0]=-1
#### T should be in meV
##### without any pre-factors like pi
@nb.jit(nopython=True)
def alphaTermSC(sc, Tmev):
    T = Tmev
    pre = sc.DeltaInt*np.abs(sc.Mel)*np.abs(sc.Mel)
    Den = 1.0
    Den *=  ( np.exp(sc.e1/T) -1  )
    Den *=  ( np.exp(sc.e2/T) -1  )
    Den *=  ( np.exp(sc.e3/T) -1  )
    
    C1 = 1.0
    C2 = 1.0
    if sc.roles[1] == 1:
        C1 *= np.exp(sc.e1/T)
    else:
        C2 *= np.exp(sc.e1/T)

    if sc.roles[2] == 1:
        C1 *= np.exp(sc.e2/T)
    else:
        C2 *= np.exp(sc.e2/T)

    if sc.roles[3] == 1:
        C1 *= np.exp(sc.e3/T)
    else:
        C2 *= np.exp(sc.e3/T)

    AlpTerm = pre*(C1-C2)/Den
    AlpTerm /= sc.e0
    return AlpTerm


#### version corresponding to e0 << T
@nb.jit(nopython=True)
def alphaTermSC_0(sc, Tmev):
    T = Tmev
    pre = sc.DeltaInt*np.abs(sc.Mel)*np.abs(sc.Mel)
    Den = 1.0
    Den *=  ( np.exp(sc.e1/T) -1  )
    Den *=  ( np.exp(sc.e2/T) -1  )
    Den *=  ( np.exp(sc.e3/T) -1  )
    
    C1 = 1.0
    if sc.roles[1] == 1:
        C1 *= np.exp(sc.e1/T)
    if sc.roles[2] == 1:
        C1 *= np.exp(sc.e2/T)
    if sc.roles[3] == 1:
        C1 *= np.exp(sc.e3/T)

    AlpTerm = pre*(C1)/Den
    AlpTerm /= T
    return AlpTerm

def alphaList(slist, TK, gK, regime=1):
    res = 0.0
    T = TK*ut2.K_to_mev
    for sc in slist:
        if regime==1:
            res += alphaTermSC(sc, T)
        elif regime==0:
            res += alphaTermSC_0(sc, T)

    res *= np.pi  ##### (2pi/hbar) from golden rule
                ####### note, there is a double counting because the states 2 and 3 are interchangable
    res *= gK
    #### note: factors related to the integration like the area of unit cell
    ##### and (1/(2 pi)^{dim}) should be already included into gK
    return res






