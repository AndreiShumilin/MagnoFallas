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
the module combines the scattering events created by "ScatteringList2M1Ph"
and the quantum mechanics  by  "Quantum2M1Ph"
The idea is to (1) calculate the matrix elements corresponding to each scattering events
and (2) use them to calculate damping at given T
the elements do not depend on T, making the calculating of many different T fast
"""



import numpy as np
import numpy.linalg
import numba as nb

from MagnoFallas.Interface import PseudoRad as prad
from MagnoFallas.Utils import util2 as ut2

import MagnoFallas.OldRadtools as rad

from MagnoFallas.Quantum.process2M1Ph import Quantum2M1Ph as quant
from MagnoFallas.Quantum.process2M1Ph import Quantum2M1Ph_LRDD as quantLR
from MagnoFallas.Boltzmann.process2M1Ph import ScatteringList2M1Ph as slist




__all__ = ['ScatToMel','initiateMel','alphaTerm2M1Ph', 'alpha2MqPh']

##### calculates the matrix element for the scattering event
def ScatToMel(scat, Hlines, pSH, pos, Ephon, LR=False, LRlines=None, Vuc = None):
    NMat = pSH.Nat
    
    kx2 = (scat.k0, scat.k1real)
    lamx2 = (scat.lk0, scat.lk1)
    
    o0,G0,Gi0_0 = prad.omega(pSH, -scat.roles[0]*scat.k0)
    o1,G1,Gi1_0 = prad.omega(pSH, -scat.roles[1]*scat.k1real)
    Gi0 = ut2.ModGi(Gi0_0, -scat.roles[0]*scat.k0, pos) [0]
    Gi1 = ut2.ModGi(Gi1_0, -scat.roles[1]*scat.k1real, pos)[0]
    Gix2 = (Gi0,Gi1)
    
    ph_Psi = Ephon.eVector(scat.qreal, scat.qBra).copy()
    mel = quant.FullMel(Hlines, kx2, lamx2, Gix2, NMat, scat.qreal, scat.ePh, ph_Psi, Ephon.masses, scat.roles )
    if LR:
        mel2 = quantLR.LRMel(LRlines, kx2, lamx2, Gix2, NMat, scat.qreal, scat.ePh, ph_Psi, Ephon.masses, scat.roles, Vuc )
    else:
        mel2 = 0
    return mel + mel2

#### initiates matrix elements in a full list
def initiateMel(scatList, Hlines, SH, pSH, Ephon, LR=False, LRlines=None, Vuc = None):
    r"""
    calculates Matrix elements for the scattering events in the list scatList
    Hlines - lines of 2M1Ph Hamiltonian,
    SH - spin Hamiltonian
    pSH - pseudorad spin Hamiltonian
    Ephon - extended phonon object according to UtilPhonopy.py module
    LR - whether to include long-range dipole-dipole interaction
    LRlines - Hamiltonian lines related to such an interaction
    """
    pos = np.array([at.position for at in SH.magnetic_atoms])
    for sc in scatList:
        mel = ScatToMel(sc, Hlines, pSH, pos, Ephon, LR=LR, LRlines= LRlines, Vuc=Vuc)
        sc.Mel = mel


#### calculates the contribution to Gilber alpha from a single scattering event
#### 2M1Ph (both options)
@nb.jit(nopython=True)
def alphaTerm2M1Ph(sc, Tmev, cellV):

    if sc.ePh < ut2.Global_zero:
        return 0.0          ###### prevents nan-contribution from q=0 event (formaly it fulfills the energy conservation law)

    dim = sc.dim
    T = Tmev
    
    t1 = np.abs(sc.Mel)*np.abs(sc.Mel) 

    Nmag1 = 1/(np.exp(sc.e1/T)-1)
    Mphon = 1/(np.exp(sc.ePh/T)-1)

    Nmin = 0
    Mpl = 0
    if sc.roles[1] == -1:
        Nmin += Nmag1
    else:
        Mpl += Nmag1

    if sc.roles[2] == -1:
        Nmin += Mphon
    else:
        Mpl += Mphon
    
    t1 *= ( (Nmin-Mpl)/sc.e0 )
    
    t1 *= sc.DeltaInt
    t1 *= 2*np.pi*cellV/ ( (2*np.pi)**dim  )
    return t1

#### calculates the  2M1Ph (magnon-conserving) contribution to Gilber alpha from a list of events
@nb.jit(nopython=True)
def alpha2M1Ph(slist, TK, cell):

    if len(slist) == 0:
        return 0.0

    regime2D = (slist[0].dim == 2)

    cellV = ut2.cellVolume(cell, regime2D=regime2D)
    
    res = 0.0
    Tmev = TK*ut2.K_to_mev
    for sc in slist:
        res += alphaTerm2M1Ph(sc, Tmev, cellV)
    return res












