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
import matplotlib.pyplot as plt

from MagnoFallas.Utils import util2 as ut2
from MagnoFallas.Boltzmann.process4M import kinetics4M as kin
from MagnoFallas.Boltzmann.process4M import boltzman4M as bolt

class Stat4M:
    def __init__(self, bolt, T):
        ScDict = bolt.Sc_listSets
        self.lamAcu = bolt.lamAcu 
        self.Tmev = T*ut2.K_to_mev
        self.ScDict = ScDict
        self.conD = {}
        self.Kgr = bolt.KGr
        self.gGr = bolt.gGr

        self.Mult = np.pi*bolt.gGr
        if bolt.dim==2:
            self.Mult *= bolt.Scell/(4*np.pi*np.pi) 
        else:
            self.Mult *= bolt.Vcell/(8*np.pi*np.pi*np.pi) 
        
        for k, lstSc in self.ScDict.items():
            ContrLst = []
            for sc in lstSc:
                con1 = kin.alphaTermSC(sc, self.Tmev) * self.Mult
                ContrLst.append(con1)
            self.conD[k] = ContrLst
            
    def TestAcu(self):
        alp = 0
        alpAcu = 0
        for k, lstSc in self.ScDict.items():
            lstC = self.conD[k]
            for isc,sc in enumerate(lstSc):
                con1 = lstC[isc]
                alp+=con1
                if ((sc.lk0 == self.lamAcu) and (sc.lk1 == self.lamAcu) and (sc.lk2 == self.lamAcu) and (sc.lk3 == self.lamAcu)):
                    alpAcu += con1
        return alp, alpAcu

    def typicalK(self, lam=None):
        norm = 0
        resK = 0
        for k, lstSc in self.ScDict.items():
            lstC = self.conD[k]
            for isc,sc in enumerate(lstSc):
                con1 = lstC[isc]
                if lam is None:
                    good=True
                else:
                    good = (sc.lk1 == lam)
                if good:
                    norm+=con1
                    ak = np.linalg.norm(sc.k1)
                    resK += ak*con1
        if norm==0:
            resK = 0
        else:
            resK /= norm
        return resK

    def Z0slice3D(self, Nk=None, k3regime=False, kmax=None, kzcut=None):
        if Nk is None:
            Nk=self.Kgr.shape[0]
        if kzcut is None:
            kzcut = self.Kgr[0,0,1][2] - self.Kgr[0,0,0][2]
        if kmax is None:
            kmax = np.max(self.Kgr)

        kar = np.linspace(-kmax, kmax, Nk+1)
        dk = kar[1]-kar[0]
        kx = kar[:Nk] + (dk/2)
        
        res = np.zeros((Nk,Nk))
        for k, lstSc in self.ScDict.items():
            lstC = self.conD[k]
            for isc,sc in enumerate(lstSc):
                con1 = lstC[isc]
                if k3regime: 
                    kt = sc.k3real
                else:
                    kt = sc.k1
                if np.abs(kt[2])<kzcut:
                    ix = int( Nk*(kt[0]+kmax)/(2*kmax) )
                    iy = int(Nk*(kt[1]+kmax)/(2*kmax)  )
                    good = (ix>0) and (ix<Nk)
                    good = good and (iy>0) and (iy<Nk)
                    if good:
                        res[ix,iy] += con1
        return kx, res
                        


        
            
    
        
    
    