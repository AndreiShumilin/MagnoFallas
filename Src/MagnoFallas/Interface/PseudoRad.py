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
This module is intended to make magnon spectrum calculations numba-compatible
First the TpradSH object is created by make_pSH() that reads the
important properties from magnon dispersion (alternative option: make_pSH(), starting from the spin Hamiltonian)
It is then used to calculated magnon frequencies/wavefunctions  by omega() procedure
"""


import numpy as np
import numpy.linalg
import scipy as sp
import matplotlib.pyplot as plt

import numba as nb
from numba.experimental import jitclass


import MagnoFallas.OldRadtools as rad
from MagnoFallas.Utils import DipoleDipole as dd
from MagnoFallas.Utils import util2 as ut2
from MagnoFallas.Quantum import quantum_util as qut

__all__ = ['TpradSH','make_pSH','J','A','B','H','nb_solve_via_colpa','omega']

###
Stabilizing_field_0 = 1e-6   ### field used to stabilizing spin-Hamiltonans
                             ### always directed along the spin magnetization
                             ### therefore, it is un-physically different for different sub-lattices
###

ex = np.array((1,0,0), dtype=np.float64)
ey = np.array((0,1,0), dtype=np.float64)
ez = np.array((0,0,1), dtype=np.float64)


Tesl_to_mev = 5.788e-2

zeros3c = np.zeros((1,1,1), dtype=np.complex128)

TpradSHlist =  [
    ('Nb', nb.int32),  
    ('Nat', nb.int32),
    ('S', nb.float64[:,::1]),
    ('g', nb.float64[::1]),
    ('B_mag', nb.float64),
    ('B_stab', nb.float64),
    ('Jij', nb.complex128[:,:,:]),
    ('i', nb.int32[::1]),  
    ('j', nb.int32[::1]),  
    ('dij', nb.float64[:,::1]),
    ('u', nb.complex128[:,::1]),
    ('v', nb.complex128[:,::1]),
    ('C', nb.complex128[:,::1]),
    ##### information for long-range dipole-dipole interaction
    ##### must be provided if long-range dipole-dipole interaction is important
    ('LR', nb.boolean),     ### wether to include long-range dipole-dipole interaction
    ('dim', nb.int32),      ### dimension, valid numbers 2 or 3
    ('R0', nb.float64),     ###  cutof distance of short/long range dipole-dipole interaction
    ('Vcell',nb.float64),   ###  area/volume of the unit cell
    #-------------------------------------------------#
    ('Rmatr', nb.complex128[:,:,:])  ## rotation matricies for all the spins
]

@jitclass(TpradSHlist)
class TpradSH(object):
    def __init__(self, Jij, i, j, dij, Nat, S, u, v, B, signs,
                 LR=False, dim=0, R0=0.0, Vcell=1.0, Rmatr = zeros3c,
                 stabField = Stabilizing_field_0):
        self.Jij = Jij
        self.i = i
        self.j = j
        self.dij = dij
        self.Nb = len(i)
        self.Nat = Nat
        self.S = S
        self.u = u
        self.v = v
        self.C = np.zeros((Nat,Nat), dtype=np.complex128)

        NS = S.shape[0]
        self.g = np.zeros(NS)
        for i in range(NS):
            self.g[i] = 2.0*signs[i]
                            #+ 0.5   ###### To CHANGE
        self.B_mag = B
        self.B_stab = stabField
        
        self.LR = LR
        self.dim = dim
        self.R0 = R0
        self.Vcell = Vcell
        self.Rmatr = Rmatr
        

def make_pSH(Magn, B=0.0, spinproj=None, LR=False, dim=2, SH=None, R0=0):
    r"""
    Constructs numba-compatible spin-Hamiltonian from magnon dispersion
    Important!!!: avoid for Ferrimagnetic/AFM materials
                    Use make_pSH2 instaead
    """

    
    Nat = Magn.N
    Jij = np.asarray(Magn.J_matrices, dtype=np.complex128)
    ai = np.asarray(Magn.indices_i, dtype=np.int32)
    aj = np.asarray(Magn.indices_j, dtype=np.int32)
    dij = np.asarray(Magn.dis_vectors, dtype=np.float64)
    S1 = np.asarray(Magn.S, dtype=np.float64)
    u1 = np.asarray(Magn.u, dtype=np.complex128)
    v1 = np.asarray(Magn.v, dtype=np.complex128)
    
    signs = np.zeros(Nat, dtype=np.float64)
    if not (spinproj is None):
        for i in range(Nat):
            signs[i] = np.sign(spinproj[i], dtype=np.float64)
    else:
        signs += 1.0

    if LR:
        Vcell = ut2.cellVolume(SH.cell, regime2D=(dim==2))
        Rmatr = np.zeros((Nat,3,3), dtype=np.complex128)
        for iat,at in enumerate(SH.magnetic_atoms):
            Rm = qut.vec_to_RM(at.spin_vector)
            Rmatr[iat] = Rm
        pSH = TpradSH(Jij, ai, aj, dij, Nat, S1, u1, v1, B, signs,   LR,dim,R0,Vcell,Rmatr)
    else:
        pSH = TpradSH(Jij, ai, aj, dij, Nat, S1, u1, v1, B, signs)
    pSH.C = prad_init_C(pSH)
    return pSH

#####  must be made from non-rotated Hamiltonian to correctly introduce 
#####  g-factors to the pseudo-Rad Hamiltonian
#####  note: magnetic field is considered to be applied along mean spin polarization
#####  in AFMs it is applied alonf the direction of a random sub-lattice
#####  it would also work incorrectly for non-colinear magnets
def make_pSH2(SH, B=0.0, LR=False, dim=2, R0=0, stabField = Stabilizing_field_0):
    r"""
    Constructs numba-compatible spin-Hamiltonian from the initial spin-hamiltonian SH
    B  external magnetic field [T]
    LR - to include long-range dipole-dipole interaction
    dim - system dimension (important for LR dipole-dipole)
    R0 - cutoff parameter for long-range (LR)  dipole-dipole
    stabField - additional "fictional" field helping for convergance of the matricies at Gamma point. Should be 
                a small positive value

    Note: to correctly use the LR dipole-dipole interaction, the short-range dipole-dipole interaction 
         must be already added to SH with the same cutoff distance R0
    """

    
    SHferro = qut.make_FerroHam(SH)
    Magn = rad.MagnonDispersion(SHferro)
    Nat = Magn.N
    Jij = np.asarray(Magn.J_matrices, dtype=np.complex128)
    ai = np.asarray(Magn.indices_i, dtype=np.int32)
    aj = np.asarray(Magn.indices_j, dtype=np.int32)
    dij = np.asarray(Magn.dis_vectors, dtype=np.float64)
    S1 = np.asarray(Magn.S, dtype=np.float64)
    u1 = np.asarray(Magn.u, dtype=np.complex128)
    v1 = np.asarray(Magn.v, dtype=np.complex128)

    av_spin = np.zeros(3)
    for at in SH.magnetic_atoms:
        av_spin += at.spin_vector
    av_spin /= Nat
    if np.linalg.norm(av_spin) < ut2.Global_zero:
        av_spin = SH.magnetic_atoms[0].spin_vector
        ### it means that we deal with ideally compensated AFM
    
    signs = np.zeros(Nat, dtype=np.float64)
    for i in range(Nat):
        sp1 = SH.magnetic_atoms[i].spin_vector
        signs[i] = np.sign( sp1@av_spin, dtype=np.float64  )
    
    if LR:
        Vcell = ut2.cellVolume(SH.cell, regime2D=(dim==2))
        Rmatr = np.zeros((Nat,3,3), dtype=np.complex128)
        for iat,at in enumerate(SH.magnetic_atoms):
            Rm = qut.vec_to_RM(at.spin_vector)
            Rmatr[iat] = Rm
        pSH = TpradSH(Jij, ai, aj, dij, Nat, S1, u1, v1, B, signs,   LR,dim,R0,Vcell,Rmatr, stabField=stabField)
    else:
        pSH = TpradSH(Jij, ai, aj, dij, Nat, S1, u1, v1, B, signs, stabField=stabField)
    pSH.C = prad_init_C(pSH)
    return pSH





@nb.jit(nopython=True)
def J(pSH, k):
    r"""
    numba-compatible clone of Magnon_dispersion.J
    """
    # Initialize matrix
    result = np.zeros((pSH.Nat, pSH.Nat, 3, 3), dtype=np.complex128)
    # Compute J(k)
    for index in range(pSH.Nb):
        i = pSH.i[index]
        j = pSH.j[index]
        result[i][j] += pSH.Jij[index] * np.exp(
            -1j * (k @ pSH.dij[index])
        )
        
    if pSH.LR:
        g = 2    ###### !!!!! for the time being it's fixed
        Jmat0 = dd.longrangeDDmatr(k, pSH.R0, pSH.dim, g1=2, g2=2)
        #### we calculate Jmat0 only once with effective g-factors g1 = g2 = 2
        #### g-factors of real spins can be added later
        Jmat0 *= (-1.0)*(0.5)      #### general translation between notations
        Jmat0 /= pSH.Vcell
        for i in range(pSH.Nat):
            for j in range(pSH.Nat):
                R1 = pSH.Rmatr[i].copy()
                R2 = pSH.Rmatr[j].copy()
                J1 = np.transpose(R1) @ Jmat0 @ R2
                J1 *= np.abs(pSH.g[i])*np.abs(pSH.g[j])/4
                result[i][j] += J1

    return result

@nb.jit(nopython=True)
def A(pSH, k):
    r"""
    numba-compatible clone of Magnon_dispersion.A
    """
    # Initialize matrix
    result = np.zeros((pSH.Nat, pSH.Nat), dtype=np.complex128)    
    # Compute A(k)
    J1 = J(pSH,-k)
    for i in range(len(J1)):
        for j in range(len(J1[i])):
            result[i][j] += (
                np.sqrt(np.linalg.norm(pSH.S[i]) * np.linalg.norm(pSH.S[j]))
                / 2
                * (pSH.u[i] @ J1[i][j] @ np.conjugate(pSH.u[j]))
            )
    return result
    
@nb.jit(nopython=True)
def B(pSH, k):
    r"""
     numba-compatible clone of Magnon_dispersion.A
    """
    # Initialize matrix
    result = np.zeros((pSH.Nat, pSH.Nat), dtype=np.complex128)  
    # Compute B(k)
    J1 = J(pSH,-k)
    for i in range(len(J1)):
        for j in range(len(J1[i])):
            result[i][j] += (
                np.sqrt(np.linalg.norm(pSH.S[i]) * np.linalg.norm(pSH.S[j]))
                / 2
                * (pSH.u[i] @ J1[i][j] @ pSH.u[j])
            )
    return result

@nb.jit(nopython=True)
def prad_init_C(pSH):
    r"""
     numba-compatible clone of Magnon_dispersion.C
    """
    res = np.zeros((pSH.Nat, pSH.Nat), dtype=np.complex128)
    # Compute C matrix, note: sum over l is hidden here
    J1 = J(pSH,np.zeros(3, dtype=np.float64))
    for i in range(len(J1)):
        for j in range(len(J1[i])):
            res[i][i] += (
                np.linalg.norm(pSH.S[j]) * pSH.v[i] @ J1[i][j] @ pSH.v[j]
            )
            
    for i in range(len(J1)):
        ### g[i] should be negative for the secons dublattice
        res[i][i] += -(0.5)*Tesl_to_mev*pSH.g[i]*pSH.B_mag  ##check the signs
                        #### ??? 0.5 to make it compatible with future multiplication my 2 in H()
                        #### should be checked by calculating magnon gap in Heisenberg FM
        res[i][i] += -Tesl_to_mev*pSH.B_stab  ##check the signs
    return res

@nb.jit(nopython=True)
def H(pSH, k):
    # Compute h matrix
    left = np.concatenate(
        (2 * A(pSH,k) - 2 * pSH.C, 2 * np.conjugate(B(pSH,k)).T), axis=0
    )
    right = np.concatenate(
        (2 * B(pSH,k), 2 * np.conjugate(A(pSH,-k)) - 2 * pSH.C), axis=0
    )
    h = np.concatenate((left, right), axis=1)
    return h


@nb.jit(nopython=True)
def nb_solve_via_colpa(D):
    r"""
    numba-compatible clone of solve-via-colpa
    However, it also returns the inverse G
    """
    N = len(D) // 2
    g = np.diag(np.concatenate((np.ones(N, dtype=np.complex128), -np.ones(N, dtype=np.complex128))))


    K = np.conjugate(np.linalg.cholesky(D)).T
    L, U = np.linalg.eig(K @ g @ np.conjugate(K).T)

    # Sort with respect to L, in descending order
    U = np.concatenate((L[:, None], U.T), axis=1).T
    U = U[:, np.argsort(U[0])]
    L = U[0, ::-1].copy()
    U = U[1:, ::-1].copy()

    E = g @ L

    G_minus_one = np.linalg.inv(K) @ U @ np.sqrt(np.diag(E))

    # Compute G from G^-1 following Colpa, see equation (3.7) for details
    #G = np.conjugate(G_minus_one).T
    #G[:N, N:] *= -1
    #G[N:, :N] *= -1
    G = np.linalg.inv(G_minus_one)

    return E, G, G_minus_one



@nb.jit(nopython=True)
def omega(pSH, k, option=0):
    r"""
    numba-compatible version of Magnon_dispersion.omega
    does not contain try-except, instead an "option" should be introduced externally 
    Always returns G and G-inverse
    """
    # Diagonalize h matrix via Colpa
    h = H(pSH, k)
    if (option==0):
        omegas, G, Ginv = nb_solve_via_colpa(h)
    elif (option==1):
        omegas, G, Ginv = nb_solve_via_colpa(h + np.diag(1e-8 * np.ones(2 * pSH.Nat)))
    elif (option==2):
        omegas, G, Ginv = nb_solve_via_colpa(-h)
        omegas *= -1
        G *= -1
        Ginv *= -1
    elif (option==3):
        omegas, G, Ginv = nb_solve_via_colpa(-h - np.diag(1e-8 * np.ones(2 * pSH.Nat)) )
        omegas *= -1
        G *= -1
        Ginv *= -1
    else:
        omegas = np.zeros(2 * pSH.Nat, dtype=np.complex128)
        G = np.zeros((2 * pSH.Nat,2 * pSH.Nat), dtype=np.complex128)
        Ginv = np.zeros((2 * pSH.Nat,2 * pSH.Nat), dtype=np.complex128)

    omegas = omegas.real
    for i in range(len(omegas)):
        if omegas[i] < ut2.Global_zero:
            omegas[i] = 0.0
    
    return omegas, G, Ginv



@nb.jit(nopython=True)
def omega0(pSH, k, option=0):
    r"""
    numba-compatible version of the calculations of Magnon-dispersion
    """
    # Diagonalize h matrix via Colpa
    h = H(pSH, k)
    if (option==0):
        omegas, G, Ginv = nb_solve_via_colpa(h)
    elif (option==1):
        omegas, G, Ginv = nb_solve_via_colpa(h + np.diag(1e-8 * np.ones(2 * pSH.Nat)))
    elif (option==2):
        omegas, G, Ginv = nb_solve_via_colpa(-h)
        omegas *= -1
        G *= -1
        Ginv *= -1
    elif (option==3):
        omegas, G, Ginv = nb_solve_via_colpa(-h - np.diag(1e-8 * np.ones(2 * pSH.Nat)) )
        omegas *= -1
        G *= -1
        Ginv *= -1
    else:
        omegas = np.zeros(2 * pSH.Nat, dtype=np.complex128)
        G = np.zeros((2 * pSH.Nat,2 * pSH.Nat), dtype=np.complex128)
        Ginv = np.zeros((2 * pSH.Nat,2 * pSH.Nat), dtype=np.complex128)

    omegas = omegas.real
    for i in range(len(omegas)):
        if omegas[i] < ut2.Global_zero:
            omegas[i] = 0.0

    omegas0 = omegas[:pSH.Nat]
    
    return omegas0






