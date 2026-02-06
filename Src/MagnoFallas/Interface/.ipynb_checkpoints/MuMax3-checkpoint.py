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
import copy


from MagnoFallas.Utils import util2 as ut2
import MagnoFallas.OldRadtools as rad

const_Aex_to_Si = ut2.mev_to_J * 1e10
const_muB_Si = 9.2740100657e-24
const_mu0_Si = 1.25663706127e-6
const_hbar_Si = 1.054571817e-34
const_hbar_mev = const_hbar_Si/ut2.mev_to_J

def AExTensor(SH, dim=2, zef=10, units='Si'):
    r"""
    calculates Tensor of the exchange stiffness in the units of J/m [compatible with MuMax3]
    SH - spin Hamiltonian
    dim - dimension. If set to 2, the effective thickness zef is requires to make the system compatible with 
    MuMax3 (wich is based on 3D)
    Units: 'Si' - compatible with MuMax3, 'Int' - internal untis, Aex will be in meV/A
    """
    Vcell = ut2.cellVolume(SH.cell, regime2D=(dim==2))
    if dim==2:
        Vcell *= zef
    SH1 = copy.deepcopy(SH)
    SH1.notation = (True, False, -1)
    Aex = np.zeros((3,3))
    for at1,at2, dv, Jrad in SH1:
        J = Jrad.iso
        J *= at1.spin_vector@at2.spin_vector
        r1 = ut2.realposition(SH1, at1)
        r2 = ut2.realposition(SH1, at2)
        rdv = dv[0]*SH1.a1 + dv[1]*SH1.a2 + dv[2]*SH1.a3
        dr = (r2 + rdv) - r1
        for alp in range(3):
            for bet in range(3):
                rr = dr[alp]*dr[bet]
                Aex[alp, bet] += J*(rr/2)/Vcell
    if units=='Si':
        Aex *= const_Aex_to_Si
    return Aex

def Msat(SH, dim=2, zef=10, gs=None, units='Si'):
    r"""
    Calculates the magnetization [Si units compatible with MuMax3]
    SH - spin Hamiltonian
    dim - dimension. If set to 2, the effective thickness zef is requires to make the system compatible with 
    MuMax3 (wich is based on 3D)
    Units: 'Si' - compatible with MuMax3, 'Int' - internal untis, Aex will be in meV/A
    """
    Vcell = ut2.cellVolume(SH.cell, regime2D=(dim==2))
    if dim==2:
        Vcell *= zef

    Nat = len(SH.magnetic_atoms)
    if gs is None:
        gs = np.zeros(Nat) + 2.0

    M1 = np.zeros(3)
    for iat,at in enumerate(SH.magnetic_atoms):
        M1 += at.spin_vector*gs[iat]
    M1 /= Vcell
    aM1 = np.linalg.norm(M1)
    if units == 'Si':
        aM1 *= const_muB_Si * 1e30
    return aM1

def MagnonEfromA(k, A, M0, g=2, units='Si'):
    r"""
    Function for tests,calculates the magnon energy [meV] from
    exchange stiffness tensor A and magnetization M0
    k - wavevector in [1/A]
    """
    if units=='Si':
        k1 = k*1e10
        const = g*const_muB_Si
        constE = 1/ut2.mev_to_J
    elif units=='Int':
        const = g
        k1 = k
        constE = 1
    else:
        k1 = k
        const = 1
        constE = 1
    E = const*(2/M0)*(k1@A@k1)
    return E*constE
