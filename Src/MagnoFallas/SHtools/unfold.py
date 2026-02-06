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
A module for unfolding magnon bands calculated in supercells
"""

import numpy as np
import numpy.linalg
import numba as nb

import MagnoFallas.OldRadtools as rad
from MagnoFallas.Interface import PseudoRad as prad


__all__ = ['Unfold']


def findNum(H, name):
    for iat, at in enumerate(H.magnetic_atoms):
        if at.name == name:
            return True, iat
    return False, -1000


@nb.jit(nopython=True)
def calculateMel(k, G0, GF, Amps, rs, Nums, N0, NS ):
    Mels = np.zeros((NS, N0), dtype=np.complex128)
    for ibF in range(NS):
        for ib0 in range(N0):
            M1 = 0.0
            for iln in range(NS):
                PsiF = GF[ibF,iln]
                Psi0 = G0[ib0, Nums[iln]]
                Psi0 *= np.exp(-1j*(k@rs[iln]))
                Psi0 *= Amps[iln]
                M1 += PsiF*np.conjugate(Psi0)
            Mels[ibF,ib0] = M1
    return Mels

@nb.jit(nopython=True)
def Tmatrix1(Nom, NbF, ioms):
    M = np.zeros((Nom,NbF))
    for ib in range(NbF):
        M[ioms[ib],ib] = 1
    return M    


class Unfold:
    r"""
    Main class for unfolding
    """
    def __init__(self, H0, Hsuper, relationDict, prad_regime=True, pradB=1e-5):
        r"""
        H0 - unit cell spin Hamiltonian
        Hsuper - supercell spin Hamiltonian
        relationDict - dictionary that relates stoms in unit- and super-cells
        prad_regime - wether to use numba-accelerated procedures
        pradB - magnetic field used to stabilized pseudo-rad-tools magnons
        """
        self.H0 = H0
        self.HS = Hsuper
        self.Di = relationDict
        self.N0 = len(self.H0.magnetic_atoms)
        self.NS = len(self.HS.magnetic_atoms)

        self.Magn0 = rad.MagnonDispersion(self.H0)
        self.MagnS = rad.MagnonDispersion(self.HS)
        self.prad = prad_regime
        if prad_regime:
            self.pSH0 = prad.make_pSH(self.Magn0, B=pradB)
            self.pSHS = prad.make_pSH(self.MagnS, B=pradB)
        self.Initiate()

    def Initiate(self):
        self.Amps = np.zeros(self.NS, dtype=np.float64)
        self.rs = np.zeros((self.NS,3), dtype=np.float64)
        self.Nums = np.zeros(self.NS, dtype=np.int32)
        for k,v in self.Di.items():
            namS = k
            nam0, Rv = v
            Fou0, n0 = findNum(self.H0, nam0)
            FouS, nS = findNum(self.HS, namS)
            if (Fou0 and FouS):
                self.Amps[nS] = 1/np.sqrt(self.NS)
                self.rs[nS] = Rv[0]*self.H0.a1 + Rv[1]*self.H0.a2 + Rv[2]*self.H0.a3
                self.Nums[nS] = n0

    def Mels0(self, k):
        omF, GF = self.MagnS.omega(k, return_G=True)
        om0, G0 = self.Magn0.omega(k, return_G=True)
        Mels = np.zeros((self.NS, self.N0), dtype=np.complex128)
        for ibF in range(self.NS):
            for ib0 in range(self.N0):
                M1 = 0.0
                for iln in range(self.NS):
                    PsiF = GF[ibF,iln]
                    Psi0 = G0[ib0,self.Nums[iln]]
                    Psi0 *= np.exp(-1j*(k@self.rs[iln]))
                    Psi0 *= self.Amps[iln]
                    M1 += PsiF*np.conjugate(Psi0)
                Mels[ibF,ib0] = M1
        return Mels

    def Mels(self, k):
        if self.prad:
            omF, GF, iGH = prad.omega(self.pSHS, k)
            om0, G0, iG0 = prad.omega(self.pSH0, k)
        else:
            omF, GF = self.MagnS.omega(k, return_G=True)
            om0, G0 = self.Magn0.omega(k, return_G=True)
        Mels = calculateMel(k, G0, GF, self.Amps, self.rs, self.Nums, self.N0, self.NS )
        return Mels


    def Map0(self, kpoi, OmMax, Nom=64, OmMin=0.0):
        Nk = len(kpoi)
        maps = [np.zeros((Nk, Nom)) for i in range(self.N0)]
        mapTot = np.zeros((Nk, Nom))
        for ik,k in enumerate(kpoi):
            omsF = self.MagnS.omega(k)
            Mels2 = self.Mels(k)
            Mels2 = np.abs(Mels2*Mels2)*self.N0
            for ibF in range(self.NS):
                om1 = omsF[ibF]
                iom = int( Nom*(om1-OmMin) / (OmMax-OmMin)   )
                if iom<0:
                    iom = 0
                if iom >= Nom:
                    iom = Nom-1 
                for ib0 in range(self.N0):
                    maps[ib0][ik,iom] += Mels2[ibF,ib0]    
                    mapTot[ik,iom] += Mels2[ibF,ib0]    
        return mapTot, maps

    def Map(self, kpoi, OmMax, Nom=64, OmMin=0.0):
        r"""
        main procedure to calculate "unfilding maps"
        kpoi - list of k-points
        OmMax - maximum magnon energy [meV]
        Nom - number steps in energy
        OmMax - minimum magnon energy [meV]
        """
        Nk = len(kpoi)
        maps = [np.zeros((Nk, Nom)) for i in range(self.N0)]
        mapTot = np.zeros((Nk, Nom))
        for ik,k in enumerate(kpoi):
            if self.prad:
                omsF, GF, iGH = prad.omega(self.pSHS, k)
            else:
                omsF = self.MagnS.omega(k)
            Mels2 = self.Mels(k)
            Mels2 = np.abs(Mels2*Mels2)*self.N0
            ioms = ( Nom*(omsF-OmMin) / (OmMax-OmMin)   )
            ioms = ioms.astype(int)
            ioms = np.maximum(ioms,0)
            ioms = np.minimum(ioms,Nom-1)
            T = Tmatrix1(Nom, self.NS, ioms)
            for ib0 in range(self.N0):
                maps[ib0][ik,...] += T@Mels2[...,ib0]    
                mapTot[ik,...] += T@Mels2[...,ib0]    
            
        return mapTot, maps

    def findsimilarity(self, k, ib0):
        r"""
        finds an energy of the state in the folded spectrum
        which is the most similar to the state ib0 of the UC spectrum
        """
        if self.prad:
            omsF, GF, iGH = prad.omega(self.pSHS, k)
        else:
            omsF = self.MagnS.omega(k)
        Mels2 = self.Mels(k)
        Mels2 = np.abs(Mels2*Mels2)*self.N0
        M2x = Mels2[...,ib0]
        ib1 = np.argmax(M2x)
        return omsF[ib1]
        
            
            
