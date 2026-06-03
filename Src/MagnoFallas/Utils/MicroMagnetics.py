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
Module to calculate "macroscopic" micromagnetic marameters from Spin Hamiltonian
all results are (usually) in units natural for MagnoFallas
"""

import numpy as np
import copy

from . import util2 as ut2
import MagnoFallas.OldRadtools as rad

def DMItensor(SH, dim=2, zef=None):
    r"""
    Calculates DMI tensor
    D_{ijk}  ---->   m_j  (d/d x_i) m_k
    result in meV / A^2
    """
    Vcell = ut2.cellVolume(SH.cell, regime2D=(dim==2))
    if dim==2:
        if zef is None:
            zef = SH.cell[2,2]
        Vcell *= zef
        
    SH2 = copy.deepcopy(SH)
    SH2.notation = (True, True, 1)
    D = np.zeros((3,3,3), dtype=np.float64)
    for a1, a2, v, J in SH2:
        r1 = ut2.realposition2(a1.position, SH2.cell)
        r2 = ut2.realposition2(a2.position, SH2.cell)
        rv = v[0]*SH2.cell[0] + v[1]*SH2.cell[1] + v[2]*SH2.cell[2]
        dr = r2 + rv - r1
        Jm = J.matrix
        #print(rv, Jm)
        for i in range(dim):
            for j in range(3):
                for k in range(3):
                    D[i,j,k] += dr[i]*(Jm[j,k] - Jm[k,j])
    D /= Vcell
    return D                    


def AniTensor(SH, dim=2, zef=None):
    r"""
    calculates Anisotropy tensor
    result in meV / A^3
    """
    Vcell = ut2.cellVolume(SH.cell, regime2D=(dim==2))
    if dim==2:
        if zef is None:
            zef = SH.cell[2,2]
        Vcell *= zef
        
    SH2 = copy.deepcopy(SH)
    SH2.notation = (True, True, 1) 
    At = np.zeros((3,3))
    for a1, a2, v, J in SH2:
        Jm = J.matrix
        Jiso = J.iso
        for i in range(3):
            for j in range(3):
                A1 = (Jm[i,j] + Jm[j,i])/2
                if i==j:
                    A1 -= Jiso
                At[i,j] += A1

    At /= Vcell
    return At

def AniParameters(SH, dim=2, zef=None):
    r"""
    calculates two anisotropy constants [in meV / A^3]
    and the corresponding axes
    """
    At = AniTensor(SH, dim=dim, zef=zef)
    ee, evec = np.linalg.eigh(At)
    A1 = ee[2]-ee[0]
    A2 = ee[1]-ee[0]
    ax1 = evec[2] / np.linalg.norm(evec[2])
    ax2 = evec[1] / np.linalg.norm(evec[1])
    return A1, A2, ax1, ax2


def AExTensor(SH, dim=2, zef=None, units='Int'):
    r"""
    calculates Tensor of the exchange stiffness in the units of J/m [compatible with MuMax3]
    SH - spin Hamiltonian
    dim - dimension. If set to 2, the effective thickness zef is used to make the system compatible with 
    MuMax3 (wich is based on 3D)
    zef --- effective thickness. If not set, is taken basing on the unit cell
    Units: 'Si' - compatible with MuMax3, 'Int' - internal untis, Aex will be in meV/A
    """
    Vcell = ut2.cellVolume(SH.cell, regime2D=(dim==2))
    if dim==2:
        if zef is None:
            zef = SH.cell[2,2]
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
        for alp in range(dim):
            for bet in range(dim):
                rr = dr[alp]*dr[bet]
                Aex[alp, bet] += J*(rr/2)/Vcell
    return Aex   


def Msat(SH, dim, zef=None, gs=None, units='Int'):
    r"""
    Calculates the magnetization [Si units compatible with MuMax3]
    SH - spin Hamiltonian
    dim - dimension. If set to 2, the effective thickness zef is requires to make the system compatible with 
    MuMax3 (wich is based on 3D)
    gs - array of g-factors
    Units: 'Si' - compatible with MuMax3, 'Int' - internal untis, magnetization is in nu_B / A^3
    """
    Vcell = ut2.cellVolume(SH.cell, regime2D=(dim==2))
    if dim==2:
        if zef is None:
            zef = SH.cell[2,2]
        Vcell *= zef

    Nat = len(SH.magnetic_atoms)
    if gs is None:
        gs = np.zeros(Nat) + 2.0

    M1 = np.zeros(3)
    for iat,at in enumerate(SH.magnetic_atoms):
        M1 += at.spin_vector*gs[iat]
    M1 /= Vcell
    aM1 = np.linalg.norm(M1)

    return aM1

