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
The toy model of a material with a single atom in a square/qubic unit cell (which is magnetic)
all 3 magnon branches have acoustic dispersion E=cp with a single sound velocity c
The model contains spin-Hamiltoinan (normal anf pseudo-rad) and the information for phonons
that should be compatible with information for real materials
the magnon-phonon damping can also be calculated analitically 
"""

## last change  29.09.2005

import numpy as np

import numba as nb

from MagnoFallas.Interface import PseudoRad as prad
from MagnoFallas.SpinPhonon import SPhUtil as sph
from MagnoFallas.Utils import util2 as ut2
import MagnoFallas.OldRadtools as rad


#from MagnoFallas.Interface import UtilPhonopy as utph

__all__ = ['ToyPhonons','ToyHam','ToyModel']


@nb.jit(nopython=True)
def Jint2D(ka, Nphi=150):
    phis0 = np.linspace(0,2*np.pi, Nphi+1)
    phis = phis0[:Nphi]
    sin2 = (np.sin(ka*np.cos(phis)) )**2
    dPhi = 2*np.pi/Nphi
    return np.sum(sin2)*dPhi

@nb.jit(nopython=True)
def Jint3D(ka):
    if ka == 0:
        return 0.0

    C1 = 2*np.pi*(ka - np.sin(ka)*np.cos(ka))
    return C1/ka

class ToyPhonons:
    def  __init__(self, c, M, cell):
        ## c in meV * A
        self.masses = np.array([M,])
        self.c = c
        self.NAcc = 1
        self.Nband = 3
        self.NAcc = 3 
        self.cell = cell
        
        self.a1 = self.cell[0,...]
        self.a2 = self.cell[1,...]
        self.a3 = self.cell[2,...]
        
        self.b1 = np.cross(self.a2,self.a3)/ np.dot(self.a1, np.cross(self.a2,self.a3))
        self.b2 = np.cross(self.a3,self.a1)/ np.dot(self.a2, np.cross(self.a3,self.a1))
        self.b3 = np.cross(self.a1,self.a2)/ np.dot(self.a3, np.cross(self.a1,self.a2))

        self.bmatr = np.array([self.b1,self.b2,self.b3])
        
    def energy(self, qinp, qBra, cmin=None, isreal=True):
        qabs = np.sqrt( np.sum(qinp*qinp) )
        e1 = self.c*qabs    
        return e1  

    def energies_all(self, qinp, cmins=None, isreal=True ):
        qabs = np.sqrt( np.sum(qinp*qinp) )
        e1 = self.c*qabs    
        ens = np.array( (e1,e1,e1) )
        return ens
    

    def eVector(self, qinp, qBra, isreal=True):
        emat = np.eye(3, dtype=np.complex128)
        if qBra < 3:
            v1 = emat[qBra,...]
        else:
            v1 = np.zeros(3, dtype=np.complex128)
        res = np.array([v1,])
        return(res)

    def relate(self, SH, dim=3, Nmap=None):
        liMag = [0,]
        liAt  = [0,]
        return liMag, liAt

def ToyHam(J, kappa, a, S, dim=2):
    H = rad.SpinHamiltonian()
    H.notation = (False, False, -1)
    H.cell = a*np.eye(3)

    At1 = rad.Atom("Fake1", spin=S, position=(0, 0, 0))
    Jmatr = J*np.eye(3)
    Jmatr[2,2] += kappa*J

    H.add_bond(At1, At1, (1,0,0), matrix=Jmatr)    
    H.add_bond(At1, At1, (0,1,0), matrix=Jmatr)  
    if dim == 3:
        H.add_bond(At1, At1, (0,0,1), matrix=Jmatr)    
    return H



class ToyModel:
    def  __init__(self, J, kappa, a=1, dim=2, c=1, S=3/2, M=1, Jprime = None):
        self.J = J
        if Jprime is None:
            self.J1 = -J/a
        else:
            self.J1 = -Jprime
        self.kappa = kappa
        self.a = a
        self.dim = dim
        self.S = S
        self.c = c
        self.M = M

        self.Delta = 2*self.dim * self.J * self.S * self.kappa

        self.SH = ToyHam(self.J, self.kappa, self.a, self.S, dim=self.dim)
        self.Magn = rad.MagnonDispersion(self.SH)
        #self.PSH = prad.make_pSH(self.Magn, B=0.0)
        self.PSH = prad.make_pSH2(self.SH, B=0.0, dim=dim)

        self.Ephon = ToyPhonons(self.c, self.M, self.SH.cell)
        self.SpinPhon = self.MakeDisplacements()

    
    def MakeDisplacements(self):
        e0 = np.array((0,0,0), dtype=np.int32)
        ex = np.array((1,0,0), dtype=np.int32)
        ey = np.array((0,1,0), dtype=np.int32)
        ez = np.array((0,0,1), dtype=np.int32)
        
        dJm1 = self.J1*np.eye(3, dtype=np.complex128)
        dJm1[2,2] += self.kappa*self.J1
        
        Lis = sph.EmptydJList()
        term1a = sph.TderivJ(0, 0, ex, 0, 1.0*ex, ex, dJm1)
        term1b = sph.TderivJ(0, 0, ex, 0, 1.0*ex, e0, -dJm1)
        Lis.append(term1a)
        Lis.append(term1b)

        term2a = sph.TderivJ(0, 0, ey, 0, 1.0*ey, ey, dJm1)
        term2b = sph.TderivJ(0, 0, ey, 0, 1.0*ey, e0, -dJm1)
        Lis.append(term2a)
        Lis.append(term2b)

        if self.dim == 3:
            term3a = sph.TderivJ(0, 0, ez, 0, 1.0*ez, ez, dJm1)
            term3b = sph.TderivJ(0, 0, ez, 0, 1.0*ez, e0, -dJm1)
            Lis.append(term3a)
            Lis.append(term3b)
        return Lis
    


    def Alpha_2M1Ph(self, TK):
        dimConst = (sph.phonon_x_constant)**2
    
        Delt = 2*self.dim * self.J * self.S * self.kappa
        
        T = TK*ut2.K_to_mev
        a = self.a
        c = self.c
        d = self.dim
        M = self.M
        kap = self.kappa
    
        k1 = c/(self.J *self.S * a * a)
    
        eph = c*k1
        e1 = eph + Delt
            
        X1 = (k1*a*a) / (Delt * 2 * np.pi)
        if d==3:
            X1 *= (k1*a)/(2*np.pi)
        
        if d==3:
            Jterm = 3*Jint3D(k1*a)
        else:
            Jterm = 2*Jint2D(k1*a)
        
        X2 = 2*(kap*self.S*self.J1)**2
        X2 /= (M*c*c*k1)

        X3 = - 1/(np.exp(e1/T)-1) + 1/(np.exp(eph/T)-1)
        
        return X1*X2*X3*Jterm*dimConst

    def Energy0(self, BT):
        Bmev = BT*ut2.Tesl_to_mev
        Delta0 =  2*self.dim * np.abs(self.J) * self.S * self.kappa
        return Bmev + Delta0

    def alpha_4M(self, TK, BT=0):
        if self.dim==2:
            return self.alpha_4M_2D(TK, BT=BT)
        elif self.dim==3:
            return self.alpha_4M_3D(TK, BT=BT)
        else:
            return 0.0

    def alpha_4M_2D(self, TK, BT=0):
        Bmev = BT*ut2.Tesl_to_mev

        Delta0 =  2*self.dim * np.abs(self.J) * self.S * self.kappa
        eps1 = np.abs(self.J) * self.S * (self.a**2)
        Delta = Delta0 + Bmev
        T = TK*ut2.K_to_mev

        C1 = 1.35 * (T**2) *(self.a**4) * (Delta0**2)
        Z1 = (2*np.pi * self.S * eps1)**2  * Delta**2
        
        return C1/Z1
    
    def alpha_4M_3D(self, TK, BT=0):
        Bmev = BT*ut2.Tesl_to_mev

        Delta0 =  2*self.dim * np.abs(self.J) * self.S * self.kappa
        eps1 = np.abs(self.J) * self.S * (self.a**2)
        Delta = Delta0 + Bmev
        T = TK*ut2.K_to_mev

        C1 = (Delta0**2)  *  (T**2) * (self.a**6)
        Z1 = (2*np.pi * self.S * eps1)**3  * Delta
        C2 = 0.82 - 2.94*Delta/T
        
        return C1*C2/Z1

    def Kpath(self):
        b1,b2, b3 = self.SH.b1, self.SH.b2, self.SH.b3
        
        pG = np.zeros(3)
        pX = b1/2
        pM = b1/2 + b2/2
        pR = b1/2 + b2/2 + b3/2

        if self.dim==3:
            Points = [pG, pX, pM, pG, pR, pX]
            Labels = [r'$\Gamma$', 'X', 'M', r'$\Gamma$', 'R', 'X']
        else:
            Points = [pG, pX, pM, pG]
            Labels = [r'$\Gamma$', 'X', 'M', r'$\Gamma$']
        kpoi, xx, Xmarks = ut2.Kpath(Points)
        return kpoi, xx, Xmarks, Labels


