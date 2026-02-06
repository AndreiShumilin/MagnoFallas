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
Antiferromagnetic vestion of the toy model
the unit cell is doubled to include second magnetic atom
"""

## last change  29.09.2005

import numpy as np
import numba as nb

import MagnoFallas.OldRadtools as rad
from MagnoFallas.Interface import PseudoRad as prad
from MagnoFallas.SpinPhonon import SPhUtil as sph
from MagnoFallas.Utils import util2 as ut2



__all__ = ['ToyPhonons','ToyHam','ToyModel']


class ToyPhononsAFM:
    def  __init__(self, c, M, cell):
        ## c in meV * A
        self.masses = np.array([M,M])
        self.c = c
        self.NAcc = 1
        self.Nband = 3
        self.NAcc = 3 
        self.cell = cell
        
        self.a1 = -self.cell[0,...]
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
        res = np.array([v1,v1])
        return(res)

    def relate(self, SH, dim=3, Nmap=None):
        liMag = [0,1]
        liAt  = [0,1]
        return liMag, liAt



def ToyHamAFM(J, kappa, a, S, dim=2):
    H = rad.SpinHamiltonian()
    H.notation = (False, False, -1)
    cell0 = np.array(  ((1.0,1.0,0.0),(1.0,-1.0,0.0),(1.0,0.0,1.0))  )
    if dim==2:
        cell0[2] = np.array((0,0,1.0))
    H.cell = a*cell0

    spVector = np.array((0,0,S))

    At1 = rad.Atom("Fake1", spin=spVector, position=(0, 0, 0))
    At2 = rad.Atom("Fake2", spin=-spVector, position=(1/2, 1/2, 0.0))
    Jmatr = J*np.eye(3)
    Jmatr[2,2] += kappa*J

    H.add_bond(At1, At2, (0,0,0),  matrix=-Jmatr) ### x real direction   
    H.add_bond(At2, At1, (1,0,0), matrix=-Jmatr)  ## y real direction  
    H.add_bond(At2, At1, (0,1,0),  matrix=-Jmatr)   ## y real direction  
    H.add_bond(At2, At1, (1,1,0), matrix=-Jmatr)   ## x real direction 
    if dim == 3:
        H.add_bond(At1, At2, (-1,-1,1), matrix=-Jmatr)   ## z real direction
        H.add_bond(At1, At2, (0,0,-1), matrix=-Jmatr)    ## z real direction
    
    return H



class ToyModelAFM:
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

        self.SH = ToyHamAFM(self.J, self.kappa, self.a, self.S, dim=self.dim)
        self.Magn = rad.MagnonDispersion(self.SH)
        #self.PSH = prad.make_pSH(self.Magn, B=0.0)
        self.PSH = prad.make_pSH2(self.SH, B=0.0, dim=dim)

        self.Ephon = ToyPhononsAFM(self.c, self.M, self.SH.cell)
        self.SpinPhon = self.MakeDisplacements()

    
    def MakeDisplacements(self):
        e0 = np.array((0,0,0), dtype=np.int32)
        ex = np.array((1,0,0), dtype=np.int32)
        ey = np.array((0,1,0), dtype=np.int32)
        ez = np.array((0,0,1), dtype=np.int32)
        
        dJm1 = -self.J1*np.eye(3, dtype=np.complex128)
        dJm1[2,2] += -self.kappa*self.J1
        
        Lis = sph.EmptydJList()
        term1xa = sph.TderivJ(0, 1, e0, 1, 1.0*ex, e0, dJm1)
        term1xb = sph.TderivJ(0, 1, e0, 0, 1.0*ex, e0, -dJm1)
        Lis.append(term1xa)
        Lis.append(term1xb)
        
        term1ya = sph.TderivJ(1, 0, ex, 0, 1.0*ey, ex, dJm1)
        term1yb = sph.TderivJ(1, 0, ex, 1, 1.0*ey, e0, -dJm1)
        Lis.append(term1ya)
        Lis.append(term1yb)

        term2ya = sph.TderivJ(1, 0, ey, 0, 1.0*ey, ey, dJm1)
        term2yb = sph.TderivJ(1, 0, ey, 1, 1.0*ey, e0, -dJm1)
        Lis.append(term2ya)
        Lis.append(term2yb)

        term2xa = sph.TderivJ(1, 0, ex+ey, 0, 1.0*ex, ex+ey, dJm1)
        term2xb = sph.TderivJ(1, 0, ex+ey, 1, 1.0*ex, e0, -dJm1)
        Lis.append(term2xa)
        Lis.append(term2xb)

        if self.dim == 3:
            termz1a = sph.TderivJ(0, 1, ez-ex-ey, 1, 1.0*ez, ez-ex-ey, dJm1)
            termz1b = sph.TderivJ(0, 1, ez-ex-ey, 0, 1.0*ez, e0, -dJm1)
            Lis.append(termz1a)
            Lis.append(termz1b)

            termz2a = sph.TderivJ(0, 1, -ez, 1, 1.0*ez, -ez, dJm1)
            termz2b = sph.TderivJ(0, 1, -ez, 0, 1.0*ez, e0, -dJm1)
            Lis.append(termz2a)
            Lis.append(termz2b)
            
        return Lis
        
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





