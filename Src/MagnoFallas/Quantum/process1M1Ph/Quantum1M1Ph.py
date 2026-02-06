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

import MagnoFallas.OldRadtools as rad
from MagnoFallas.Utils import util2 as ut2
from MagnoFallas.SpinPhonon import SPhUtil as sphut
from MagnoFallas.Interface import PseudoRad as prad


### phonon-x-constnt = 10^10 [\hbar^2 / 2 * amu  * 1meV ]^{1/2}    (result is in angstr)
#phonon_x_constant = 1.4457107723636267



Tline1M1Ph_lst = [       ### 2 magnon 1 phonon
    ('A', nb.complex128),
    ('r', nb.float64[::1]),
    ('nu', nb.int32),  
    ('jdis', nb.int32),       ### number of displaced atom
    ('Rdis', nb.float64[::1]),   #### vector to UC of displaced atom from UC of i1  (in real coordinated [A])
    ('edis', nb.float64[::1])    ### 0-1-2 = x-y-z direction
]


### class for a term of 1-magnon 1 phonon Hamiltonian
### A-amplitude, n-operator numbers, r-atomic positions
@jitclass(Tline1M1Ph_lst)
class Tline1M1Ph(object):
    def __init__(self, A,r,nu, jdis, Rdis, edis):
        self.A = A
        self.r = r
        self.nu = nu
        self.jdis = jdis
        self.Rdis = Rdis
        self.edis = edis


def Empty1M1PhLineList():
    ex1 = np.array((1.,0.,0.), dtype = np.float64)
    A = 1j
    r = ex1
    nu = 1
    jdis = 1
    Rdis = np.array((1.,0.,0.), dtype = np.float64)
    edis = ex1
    
    test0 = Tline1M1Ph(A,r,nu,jdis,Rdis,edis)
    
    lis = nb.typed.List()
    lis.append(test0)
    lis.clear()
    return lis


def dJ_to_Lines_1M1Ph(lis0, dJinst, SH, Amin=1e-14):
        at1 = SH.magnetic_atoms[dJinst.i1]
        at2 = SH.magnetic_atoms[dJinst.i2]
        r1 = at1.position
        drcell = dJinst.cvecJ[0]*SH.a1 + dJinst.cvecJ[1]*SH.a2 + dJinst.cvecJ[2]*SH.a3
        r2 = at2.position + drcell
        ####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ty1 = dJinst.i1
        ty2 = dJinst.i2
        S1 = at1.spin
        S2 = at2.spin
        Nat = len(SH.magnetic_atoms) 

        jdis = dJinst.n 
        Rdis = dJinst.cvecN[0]*SH.a1 + dJinst.cvecN[1]*SH.a2 + dJinst.cvecN[2]*SH.a3
        edis = dJinst.edis 
    
        J1 = dJinst.dJ

        A = S2*np.sqrt(S1/2)*(J1[0,2] - 1j*J1[1,2])
        r = r1
        lam = ty1 
        line = Tline1M1Ph(A, r, lam, jdis, Rdis, edis)
        if np.abs(A)>Amin:
            lis0.append(line)

        A = S2*np.sqrt(S1/2)*(J1[0,2] + 1j*J1[1,2])
        r = r1
        lam = ty1 + Nat
        line = Tline1M1Ph(A, r, lam, jdis, Rdis, edis)
        if np.abs(A)>Amin:
            lis0.append(line)


        A = S1*np.sqrt(S2/2)*(J1[2,0] - 1j*J1[2,1])
        r = r2
        lam = ty2 
        line = Tline1M1Ph(A, r, lam, jdis, Rdis, edis)
        if np.abs(A)>Amin:
            lis0.append(line)

        A = S1*np.sqrt(S2/2)*(J1[2,0] + 1j*J1[2,1])
        r = r2
        lam = ty2 + Nat
        line = Tline1M1Ph(A, r, lam, jdis, Rdis, edis)
        if np.abs(A)>Amin:
            lis0.append(line)


def Full_dHS_toLines(listdJ, SH):
    lList0 = Empty1M1PhLineList()
    for dJ in listdJ:
        dJ_to_Lines_1M1Ph(lList0, dJ, SH)
    return lList0

#######--------------------------------------------------------

### matrix element corresponding to s b^+
### (anihilation magnon operator s + creation phonon operator b^+)
@nb.njit
def Wmel_s_bP(lines1M1P, k, lam, Gmin1, ph_EdAll, ph_masses, ePh):
    ### lines 1M1P - 1magnon-1phonon Hamiltonian
    ### k - wavevector
    ### lam - magnon branch
    ### Gmin1 - inversed magnon matrix (at k)
    ### ph_EdAll  --- phonon self-vector. Must be calculated at -k
    ### ph_masses - array of atom masses (in AMU)
    ### enPh - phonon energy (in meV)


    W = 0.0
    for lin in lines1M1P:
        cons1 = sphut.phonon_x_constant/(np.sqrt(ph_masses[lin.jdis]*ePh))
        A = lin.A
        G = Gmin1[lin.nu, lam]
        vecPh = ph_EdAll[lin.jdis]
        edis = (1+0j)*lin.edis
        Cph = np.dot(vecPh, edis)
        ex = k@(lin.r - lin.Rdis)
        ex = np.exp(1j*ex)
        W += A*G*Cph*cons1*ex
    return W
    
    


### matrix element corresponding to s^+ b
### The methods do not agree in phase
### presumably because of different (not tunes) phases of +k and -k phonons
@nb.njit
def Wmel_sP_b(lines1M1P, k, lam, Gmin1, ph_EdAll, ph_masses, ePh):
    ### lines 1M1P - 1magnon-1phonon Hamiltonian
    ### k - wavevector
    ### lam - magnon branch
    ### Gmin1 (-k) - inversed magnon matrix
    ### ph_EdAll  --- now calculated at k
    ### ph_masses - array of atom masses (in AMU)
    ### enPh - phonon energy (in meV)

    W = 0.0
    M = Gmin1.shape[0] // 2
    for lin in lines1M1P:
        cons1 = sphut.phonon_x_constant/(np.sqrt(ph_masses[lin.jdis]*ePh))
        A = lin.A
        G = Gmin1[lin.nu, 2*M - 1 - lam]
        vecPh = ph_EdAll[lin.jdis]
        #Cph = np.conjugate(np.dot(vecPh,lin.edis))
        edis = (1+0j)*lin.edis
        Cph = np.dot(vecPh, edis)
        ex =  (-k)@(lin.r - lin.Rdis)
        ex = np.exp(1j *ex)
        W += A*G*Cph*cons1*ex
    return W




def JointHamiltonian(k, pSH, Ephon, lines1M1Ph, pos):
    o1,G1,Gi1_0 = prad.omega(pSH, k)
    Gi1 = ut2.ModGi(Gi1_0, k, pos)[0]
    nMagn = len(o1)//2

    Nphon = Ephon.Nband
    ePHmev, PsiPhfull = Ephon.fullSolve(-k)
    for i in range(3):
        ePHmev[i] = Ephon.energy(-k,i)   #### to get the sound-velocity based correstions

    Nj = nMagn + Nphon
    Hjoint = np.zeros((Nj,Nj), dtype=np.complex128)

    for i in range(nMagn):
        Hjoint[i,i] = o1[i]
    for i in range(Nphon):
        Hjoint[i+nMagn,i+nMagn] = ePHmev[i]

    for iM in range(nMagn):
        for iP in range(Nphon):
            iPH = iP + nMagn
            w = Wmel_s_bP(lines1M1Ph, k, iM, Gi1, PsiPhfull[iP], Ephon.masses, ePHmev[iP])
            Hjoint[iPH,iM] = w
            Hjoint[iM,iPH] = np.conjugate(w)
            
    return Hjoint

