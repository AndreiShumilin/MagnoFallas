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
import copy

import numba as nb
from numba.experimental import jitclass
import MagnoFallas.OldRadtools as rad


from MagnoFallas.SpinPhonon import SPhUtil as sphut
from MagnoFallas.Quantum import quantum_util as qut



### phonon-x-constnt = 10^10 [\hbar^2 / 2 * amu  * 1meV ]^{1/2}    (result is in angstr)
phonon_x_constant = 1.4457107723636267


############# class describing lines in magnon Hamiltonian, phonons are still displacements ######################

Tline2M1Ph_lst = [       ### 2 magnon 1 phonon
    ('A', nb.complex128),
    ('r', nb.float64[:,::1]),
    ('nu', nb.int32[::1]),  
    ('jdis', nb.int32),       ### number of displaced atom
    ('Rdis', nb.float64[::1]),   #### vector to UC of displaced atom from UC of i1  (in real coordinated [A])
    ('edis', nb.float64[::1])    ### 0-1-2 = x-y-z direction
]



### class for a term of 2-magnon 1 phonon Hamiltonian
### A-amplitude, n-operator numbers, r-atomic positions
@jitclass(Tline2M1Ph_lst)
class Tline2M1Ph(object):
    def __init__(self, A,r,nu, jdis, Rdis, edis):
        self.A = A
        self.r = np.asarray(r, dtype=np.float64)
        self.nu = np.asarray(nu, dtype=np.int32)
        self.jdis = jdis
        self.Rdis = np.asarray(Rdis, dtype=np.float64)
        self.edis = np.asarray(edis, dtype=np.float64)


def Empty2M1PhLineList():
    ex1 = np.array((1.,0.,0.), dtype = np.float64)
    A = 1j
    r = np.array([ex1,ex1])
    nu = np.array([1,1], dtype = np.int32)
    jdis = 1
    Rdis = np.array((1,0,0), dtype = np.float64)
    edis = ex1
    
    test0 = Tline2M1Ph(A,r,nu,jdis,Rdis,edis)
    
    lis = nb.typed.List()
    lis.append(test0)
    lis.clear()
    return lis



def dJ_to_Lines_2M1Ph(lis0, dJinst, SH, RmDict):
    ### RmDict - dictionary of rotation matricies 
    ### corresponding to the Ferromagnetic Hamiltonian transformation
        at1 = SH.magnetic_atoms[dJinst.i1]
        at2 = SH.magnetic_atoms[dJinst.i2]
        Rm1 = RmDict[at1.name]
        Rm2 = RmDict[at2.name]
    
    
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
        #Rdis = dJinst.cvecN
        Rdis = dJinst.cvecN[0]*SH.a1 + dJinst.cvecN[1]*SH.a2 + dJinst.cvecN[2]*SH.a3
        edis = dJinst.edis 
    
        J1_0 = dJinst.dJ
        J1 = np.transpose(Rm1) @ J1_0 @ Rm2

        A = 0.5*np.sqrt(S1*S2)*(J1[0,0] - J1[1,1] -1j*(J1[0,1] + J1[1,0]) )
        rs = np.array( [r1, r2]  )   
        ts = np.array( [ty1, ty2] )   
        line = Tline2M1Ph(A, rs, ts, jdis, Rdis, edis)
        lis0.append(line)

        A = 0.5*np.sqrt(S1*S2)*(J1[0,0] + J1[1,1] +1j*(J1[0,1] - J1[1,0]) )
        rs = np.array( [r1, r2]  )   
        ts = np.array( [ty1, ty2 + Nat] )   
        line = Tline2M1Ph(A, rs, ts, jdis, Rdis, edis)
        lis0.append(line)

        A = 0.5*np.sqrt(S1*S2)*(J1[0,0] + J1[1,1] -1j*(J1[0,1] - J1[1,0]) )
        rs = np.array( [r1, r2]  )   
        ts = np.array( [ty1+Nat, ty2] )   
        line = Tline2M1Ph(A, rs, ts, jdis, Rdis, edis)
        lis0.append(line)

        A = 0.5*np.sqrt(S1*S2)*(J1[0,0] - J1[1,1] +1j*(J1[0,1] + J1[1,0]) )
        rs = np.array( [r1, r2]  )   
        ts = np.array( [ty1 + Nat, ty2 + Nat] )   
        line = Tline2M1Ph(A, rs, ts, jdis, Rdis, edis)
        lis0.append(line)

        A = -S2*J1[2,2]
        rs = np.array( [r1, r1]  )   
        ts = np.array( [ty1+Nat, ty1] )   
        line = Tline2M1Ph(A, rs, ts, jdis, Rdis, edis)
        lis0.append(line)

        A = -S1*J1[2,2]
        rs = np.array( [r2, r2]  )   
        ts = np.array( [ty2+Nat, ty2] )   
        line = Tline2M1Ph(A, rs, ts, jdis, Rdis, edis)
        lis0.append(line)

def PermutatedList(list0):
   list1 = Empty2M1PhLineList() 
   for el in list0:
       A = el.A
       r = el.r
       nu = el.nu
       jdis = el.jdis
       Rdis = el.Rdis
       edis = el.edis
       r2 = np.roll(r,1, axis=0)
       nu2 = np.roll(nu,1)
       line1 = Tline2M1Ph(A, r, nu, jdis, Rdis, edis)
       line2 = Tline2M1Ph(A, r2, nu2, jdis, Rdis, edis)
       list1.append(line1)
       list1.append(line2)
   return list1


def Full_dHS_toMagn(listdJ, SH0):
    r"""
    Calculation of quantum "magnon-phonon" Hamiltonian
    from the initial spin Hamiltonian SH0
    and list of the modifications of exchange interaction matricies listdJ

    both are expected to be in the "laboratory" coordinate frame
    the rotation to "ferromagnetic" frame is performed within this module
    """
    SH = copy.deepcopy(SH0)
    SH.notation = (False, False, -1)
    RotDict = qut.FerroHam_rotation_dict(SH)
   
    lList0 = Empty2M1PhLineList()
    for dJ in listdJ:
        dJ_to_Lines_2M1Ph(lList0, dJ, SH, RotDict)
    lList1 = PermutatedList(lList0)
    return lList1    



##------------  calculating of matrix elements -------------------

@nb.jit(nopython=True)
def lineMel(line, kx2, lamx2, Gix2, NMat, ph_q, ph_Om, ph_EdAll, ph_masses, roles ):
    A = line.A
    ex = -roles[0]*kx2[0]@line.r[0] - roles[1]*kx2[1]@line.r[1] - roles[2]*ph_q@line.Rdis
    ex = np.exp(1j*ex)
    vecPh = ph_EdAll[line.jdis]
    
    ePh = np.dot(vecPh, (1+0j)*line.edis)
    CPh = sphut.phonon_x_constant/np.sqrt(2*ph_Om*ph_masses[line.jdis])

    CMagn = 1.0
    for ima in range(2):
        if roles[ima] == -1:
            noper = lamx2[ima]
        else:
            noper = 2*NMat -1 - lamx2[ima]
        CMagn *= Gix2[ima][line.nu[ima], noper]

    res = A * ex * ePh * CPh *CMagn
    return res



@nb.jit(nopython=True)
def FullMel (linelist, kx2, lamx2, Gix2, NMat, ph_q, ph_Om, ph_EdAll, ph_masses, roles ):
    Mel = 0. + 0j
    for line in linelist:
        Mel += lineMel(line, kx2, lamx2, Gix2, NMat, ph_q, ph_Om, ph_EdAll, ph_masses, roles )
    return Mel


#########################################################################################

