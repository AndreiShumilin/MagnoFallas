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
import scipy as sp
import numba as nb
from numba.experimental import jitclass

from MagnoFallas.Math import Dipole_Integrations_3D as dd3D
from MagnoFallas.Math import Dipole_Integrations_3D_deriv as dd3Dd
from MagnoFallas.Utils import util2 as ut2
from MagnoFallas.Utils import DipoleDipole as dd

from MagnoFallas.SpinPhonon import SPhUtil as sphut

from MagnoFallas.Quantum import quantum_util as qut

#from MagnoFallas.Interface import UtilPhonopy as utph


eye1c = np.eye(3, dtype = np.complex128)



Tline2M1Ph_lst = [       ### 2 magnon 1 phonon
    ('A', nb.complex128[:,:]),
    ('r', nb.float64[::1]),      ### True (1) means typ2 (to be integrated), False (0) means central atom of typ1
    ('Rph', nb.float64),         ###  displaced atom: True (1) :to be integrated, False (0): means central atom of typ1
    ('nu', nb.int32[::1]),       ### types of magnon operators  
    ('jdis', nb.int32),          ### number of displaced atom
    ('edis', nb.float64[::1]),    ### 0-1-2 = x-y-z direction
    #####################################
    ('R0', nb.float64),           ### cutoff distance for LR DD
    ('dim', nb.int32),
    ####################################
   ('R1T', nb.complex128[:,::1]),    ##### matrices required for spin rotation
   ('R2', nb.complex128[:,::1])     ##### Jeff = R1T @ J @ R2
]


### class for a term of 2-magnon 1 phonon Hamiltonian
### A-amplitude, nu-operator numbers, r-atomic positions
@jitclass(Tline2M1Ph_lst)
class Tline2M1PhLR(object):
    def __init__(self, A,r,nu, Rph, edis, R0, dim, R1T=eye1c, R2=eye1c):
        self.A = np.asarray(A, dtype=np.complex128)
        self.r = np.asarray(r, dtype=np.float64)
        self.nu = np.asarray(nu, dtype=np.int32)
        self.Rph = Rph
        self.edis = np.asarray(edis, dtype=np.float64)

        self.R0 = R0
        self.dim = dim

        self.R1T = np.asarray(R1T, dtype=np.complex128)
        self.R2 = np.asarray(R2, dtype=np.complex128)

def lineMelDD(line, kx2, lamx2, Gix2, NMat, ph_q, ph_Om, ph_EdAll, ph_masses, roles, Vuc ):

    
    Amat = line.A
    double_counting_factor = 0.5
    
    kef = -roles[0]*line.r[0]*kx2[0] - roles[1]*line.r[1]* kx2[1] - roles[2]*line.Rph*ph_q 
    Dx, Dy, Dz = dd.longrangeDDderivs(kef, line.R0, line.dim)    #, g1=1, g2=1) 
    ### integrates the derivatives of dipole-dipole matricies over r>R0. 1/Vuc is the "density of spins" of any single type
    Dxeff = line.R1T@Dx@line.R2 /Vuc
    Dyeff = line.R1T@Dy@line.R2 /Vuc
    Dzeff = line.R1T@Dz@line.R2 /Vuc

    vecPh = ph_EdAll[line.jdis]
    Jeff = vecPh[0]*Dxeff  +  vecPh[1]*Dyeff + vecPh[2]*Dzeff
    A = np.sum(Amat*Jeff) * double_counting_factor

    ePh = np.dot(vecPh, (1+0j)*line.edis)
    CPh = sphut.phonon_x_constant/np.sqrt(2*ph_Om*ph_masses[line.jdis])

    CMagn = 1.0
    for ima in range(2):
        if roles[ima] == -1:
            noper = lamx2[ima]
        else:
            noper = 2*NMat -1 - lamx2[ima]
        CMagn *= Gix2[ima][line.nu[ima], noper]

    res = A * ePh * CPh *CMagn
    return res


def LRMel(Llines, kx2, lamx2, Gix2, NMat, ph_q, ph_Om, ph_EdAll, ph_masses, roles, Vuc ):
    res = 0.0
    if Vuc is None:
        return 0.0
    for ln in Llines:
        res += lineMelDD(ln, kx2, lamx2, Gix2, NMat, ph_q, ph_Om, ph_EdAll, ph_masses, roles, Vuc )
    return res


def PermutatedList(list0):
    list1 = nb.typed.List()
    for el in list0:
        A = el.A
        r = el.r
        nu = el.nu
        r2 = np.roll(r,1, axis=0)
        nu2 = np.roll(nu,1)
        
        Rph = el.Rph
        edis = el.edis
        
        R0 = el.R0
        dim = el.dim
        R1T = el.R1T
        R2 = el.R2
        
        line1 = Tline2M1PhLR(A, r, nu,   Rph, edis, R0, dim, R1T, R2)
        line2 = Tline2M1PhLR(A, r2, nu2, Rph, edis, R0, dim, R1T, R2)
        list1.append(line1)
        list1.append(line2)
    return list1



def LR_to_Lines_2M1Ph(SH, R0, liMag, dim):
    lis0 = nb.typed.List()
    RotDict = qut.FerroHam_rotation_dict(SH)
    for i1,at1 in enumerate(SH.magnetic_atoms): 
        for i2,at2 in enumerate(SH.magnetic_atoms): 
            Rm1 = RotDict[at1.name]
            Rm2 = RotDict[at2.name].copy().astype(np.complex128)
            Rm1T = np.transpose(Rm1).copy().astype(np.complex128)
            
            ty1 = i1
            ty2 = i2
            S1 = at1.spin
            S2 = at2.spin
            Nat = len(SH.magnetic_atoms) 
    
            jdis1 = liMag[i1]
            Rdis = 1

            for edis in (ut2.ex, ut2.ey, ut2.ez):
                #A = 0.5*np.sqrt(S1*S2)*(J1[0,0] - J1[1,1] -1j*(J1[0,1] + J1[1,0]) )
                As = 0.5*np.sqrt(S1*S2)
                Amat = np.zeros((3,3), dtype = np.complex128)
                Amat[0,0] = As
                Amat[1,1] = -As
                Amat[0,1] = -1j*As
                Amat[1,0] = -1j*As
                rs = np.array( [0, 1], dtype=np.float64  )       #np.array( [r1, r2]  )   
                ts = np.array( [ty1, ty2] , dtype=np.int32) 
                line = Tline2M1PhLR(Amat, rs, ts, 1, edis,   R0, dim, Rm1T,Rm2)
                lis0.append(line)
                line = Tline2M1PhLR(-Amat, rs, ts, 0, edis,   R0, dim, Rm1T,Rm2)    
                lis0.append(line)
        
                As = 0.5*np.sqrt(S1*S2)   ####*(J1[0,0] + J1[1,1] +1j*(J1[0,1] - J1[1,0]) )
                Amat = np.zeros((3,3), dtype = np.complex128)
                Amat[0,0] = As
                Amat[1,1] = As
                Amat[0,1] = 1j*As
                Amat[1,0] = -1j*As
                rs = np.array( [0, 1], dtype=np.float64  )     #np.array( [r1, r2]  )   
                ts = np.array( [ty1, ty2 + Nat] )   
                line = Tline2M1PhLR(Amat, rs, ts, 1, edis,   R0, dim, Rm1T,Rm2)
                lis0.append(line)
                line = Tline2M1PhLR(-Amat, rs, ts, 0, edis,   R0, dim, Rm1T,Rm2)    
                lis0.append(line)
        
                As = 0.5*np.sqrt(S1*S2)   #*(J1[0,0] + J1[1,1] -1j*(J1[0,1] - J1[1,0]) )
                Amat = np.zeros((3,3), dtype = np.complex128)
                Amat[0,0] = As
                Amat[1,1] = As
                Amat[0,1] = -1j*As
                Amat[1,0] = 1j*As
                rs = np.array( [0, 1], dtype=np.float64  )   #np.array( [r1, r2]  )   
                ts = np.array( [ty1+Nat, ty2] )   
                line = Tline2M1PhLR(Amat, rs, ts, 1, edis,   R0, dim, Rm1T,Rm2)
                lis0.append(line)
                line = Tline2M1PhLR(-Amat, rs, ts, 0, edis,   R0, dim, Rm1T,Rm2)    
                lis0.append(line)
        
                As = 0.5*np.sqrt(S1*S2)    #*(J1[0,0] - J1[1,1] +1j*(J1[0,1] + J1[1,0]) )
                Amat = np.zeros((3,3), dtype = np.complex128)
                Amat[0,0] = As
                Amat[1,1] = -As
                Amat[0,1] = 1j*As
                Amat[1,0] = 1j*As
                rs = np.array( [0, 1], dtype=np.float64  )   #np.array( [r1, r2]  )   
                ts = np.array( [ty1 + Nat, ty2 + Nat] )   
                line = Tline2M1PhLR(Amat, rs, ts, 1, edis,   R0, dim, Rm1T,Rm2)
                lis0.append(line)
                line = Tline2M1PhLR(-Amat, rs, ts, 0, edis,   R0, dim, Rm1T,Rm2)     
                lis0.append(line)
        
                As = -S2   #*J1[2,2]
                Amat = np.zeros((3,3), dtype = np.complex128)
                Amat[2,2] = As
                rs = np.array( [0, 0], dtype=np.float64  )   #rs = np.array( [r1, r1]  )   
                ts = np.array( [ty1+Nat, ty1] )   
                line = Tline2M1PhLR(Amat, rs, ts, 1, edis,   R0, dim, Rm1T,Rm2)
                lis0.append(line)
                line = Tline2M1PhLR(-Amat, rs, ts, 0, edis,   R0, dim, Rm1T,Rm2)     
                lis0.append(line)
        
                A = -S1   #*J1[2,2]
                Amat = np.zeros((3,3), dtype = np.complex128)
                Amat[2,2] = As
                rs = np.array( [1, 1], dtype=np.float64  )   #rs = np.array( [r2, r2]  )   
                ts = np.array( [ty2+Nat, ty2] )   
                line = Tline2M1PhLR(Amat, rs, ts, 1, edis,   R0, dim, Rm1T,Rm2)
                lis0.append(line)
                line = Tline2M1PhLR(-Amat, rs, ts, 0, edis,   R0, dim, Rm1T,Rm2)    
                lis0.append(line)
    lis1 = PermutatedList(lis0)
    return lis1




