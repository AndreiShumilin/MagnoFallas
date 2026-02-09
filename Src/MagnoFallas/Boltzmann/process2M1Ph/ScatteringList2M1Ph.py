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
import pickle

import numba as nb
from numba.experimental import jitclass


import phonopy


import MagnoFallas.OldRadtools as rad

from MagnoFallas.Interface import UtilPhonopy as utph
from MagnoFallas.Interface import PseudoRad as prad

from MagnoFallas.Utils import util2 as ut2
from MagnoFallas.Math import SurfaceMath as smath
from MagnoFallas.Math import BulkMath as bmath
from MagnoFallas.Math import generalMath as gmath


__all__ = ['Tscat2M1Ph','EmptyScatList2M1Ph','deltaE2M1Ph','alphaKlist2M1Ph','findCenter2M1Ph','findDeltaInt2M1Ph']


zerovec = np.zeros(3, dtype = np.float64)
rolesMC = np.array((-1,1,-1), dtype=np.int32)    ### order: magnon0, magnon1, phonon
rolesMNC = np.array((-1,-1,1), dtype=np.int32)

argArray3D = np.array([bmath.nfTonxyz(i) for i in range(8)], dtype=np.int32)

######
##   Magnon-Conserving process: magnon with k0=0,e0 absorbs the phonon q and becomes a magnon with k1, e1
#####


Tscattering2M1Ph = [
    ('k0', nb.float64[::1]),              
    ('lk0', nb.int32),            
    ('k1corners', nb.float64[:,::1]),  #### order: BL - TL - BR - TR
    ('lk1', nb.int32),
    ('qBra', nb.int32),       ##branch index for phonon with q
    ('sum_k', nb.float64[::1]),  ## sum of all quasi-momenta of particles
    ('sum_e', nb.float64),  ### total energy of process (=0 in Boltzman)
###########################################################
    ('roles', nb.int32[::1]),   ### -1 if the particle disappears, +1 if it appears
                               ### order: magnon0, magnon1, phonon
    ('dim', nb.int32),    #### dimensionality, 2 or 3
###########################################################
    ('k1real', nb.float64[::1]),
    ('qreal', nb.float64[::1]),
    ('Exist', nb.int32),
    ('DeltaInt', nb.float64),
###########################################################
    ('e0', nb.float64),
    ('e1', nb.float64),
    ('ePh', nb.float64),
###########################################################
    ('Mel', nb.complex128),
###########################################################
    ('Status', nb.int32)
]


@jitclass(Tscattering2M1Ph)
class Tscat2M1Ph(object):
    def __init__(self, k0, lk0, k1corners, lk1, qBra, sum_k=zerovec, roles=rolesMC, dim=2, sum_e = 0.0 ):
        self.k0 = k0
        self.k1corners = k1corners
        
        self.sum_k = sum_k
        self.sum_e = sum_e
        self.dim = dim
        
        self.lk0=lk0
        self.lk1=lk1
        self.qBra=qBra
        self.roles=roles

        self.k1real = np.zeros(3)
        self.qreal = np.zeros(3)
        self.Exist = 0
        self.DeltaInt = -1000.0
        self.e0 = 0.0
        self.e1 = 0.0
        self.ePh = 0.0
    
        self.Mel = 0.0
        self.Status = 0



def EmptyScatList2M1Ph():
    r"""
    creates empty scattering list for 2M1Ph scatterings
    """
    k = np.array((1.0,0.0,0.0), dtype=np.float64)
    kCorn = np.array( (k,k,k,k) )
    sc1 = Tscat2M1Ph(k, 1, kCorn, 1, 1)
    lis = nb.typed.List()
    lis.append(sc1)
    lis.clear()
    return lis


def deltaE2M1Ph(e0, k1,lam1, pSH, q, qBra, Ephon, roles):
    r"""
    calculates the total energy "for the delta-function"
    """
    omg1 = prad.omega(pSH, k1)[0][lam1]
    omPh = Ephon.energy(q, qBra)
    deltaE = roles[0]*e0 + roles[1]*omg1 + roles[2]*omPh
    return deltaE


def alpha_Klist_2M1Ph(Kgr, EgrM, EgrPh, scat_list, dim, g=np.array((0,0,0)), lam0=1, ek0=0.0, excludeZero=True, roles=rolesMC):
    r"""
    Creates the list of scatterers. It is assumed that initial magnon has k=0
    Kgr - grid of k-vectors
    EgrM - grid of magnon energies
    EgrPh - grid of phonon energies
    scat_list - initial scattering list (new scatternigs would be added)
    dim - sistem dimension
    g - #not used# - reciprocal latice vector
    lam0 - acoustic magnon branch
    ek0 - energy of acoustic k=0 magnon (must be provided)
    excludezero - tracks that zero-frequency phonons are excluded (not to get infinities)
    roles = rolesMC or rolesNMC = magnon-conserving or non-conserving processes
    ////
    depending on the dimensionality one of the procedures 
    lpha_Klist_2M1Ph_2D
    lpha_Klist_2M1Ph_3D
    would be called
    """
    if dim == 2:
        slist = alpha_Klist_2M1Ph_2D(Kgr, EgrM, EgrPh, scat_list, g=g, lam0=lam0, ek0=ek0, excludeZero=excludeZero, roles=roles)
    elif dim == 3:
        slist = alpha_Klist_2M1Ph_3D(Kgr, EgrM, EgrPh, scat_list, g=g, lam0=lam0, ek0=ek0, excludeZero=excludeZero, roles=roles)
    else:
        slist = scat_list
    return slist


def alpha_Klist_2M1Ph_2D(Kgr, EgrM, EgrPh, scat_list, g=np.array((0,0,0)), lam0=1, ek0=0.0, excludeZero=True, roles=rolesMC):
    r"""
    2D version of alpha_Klist_2M1Ph_2D    
    """
    ### here we assume that the initial magnon has k=0
    ### g is not implemented
    k0 = np.zeros(3, dtype=np.float64)
    Nkx, Nky, Nkz, Nb = EgrM.shape
    Nbranch = EgrPh.shape[3]
    Ein = ek0
    ib = 0
    ibranch = 0

    def Check1(ix,iy):
        if not(excludeZero):
            return True
        else:
            si1 = np.sign( Kgr[ix,iy,0][0]*Kgr[ix+1,iy+1,0][0]  )
            si2 = np.sign( Kgr[ix,iy,0][1]*Kgr[ix+1,iy+1,0][1]  )
            return ( (si1>0) or (si2>0) )

    def dExy(ix,iy,iz):

        Emagn1 = EgrM[ix,iy,iz, ib]
        Ephon = EgrPh[ix,iy,iz, ibranch] 
        deltE = roles[0]*Ein + roles[1]*Emagn1 + roles[2]*Ephon
        return deltE
        
    for ib in range(Nb):
        for ibranch in range(Nbranch):
            for ix in range(Nkx-1):
                for iy in range(Nky-1):
                    e11 = dExy(ix,iy,0)
                    e12 = dExy(ix,iy+1,0)
                    e21 = dExy(ix+1,iy,0)
                    e22 = dExy(ix+1,iy+1,0)
                    if (not ut2.SameSign((e11,e12,e21,e22))) and (Check1(ix,iy)):
                        k1BL = Kgr[ix,iy,0]
                        k1BR = Kgr[ix+1,iy,0]
                        k1TL = Kgr[ix,iy+1,0]
                        k1TR = Kgr[ix+1,iy+1,0]
                        k1Corn = np.zeros((4,3), dtype=np.float64)
                        k1Corn[0,...] = k1BL
                        k1Corn[1,...] = k1TL
                        k1Corn[2,...] = k1BR
                        k1Corn[3,...] = k1TR
                        
                        newScat = Tscat2M1Ph(k0, lam0,  k1Corn, ib, ibranch, sum_k=zerovec, roles=roles)
                        scat_list.append(newScat)
    return scat_list


def alpha_Klist_2M1Ph_3D(Kgr, EgrM, EgrPh, scat_list, g=np.array((0,0,0)), lam0=1, ek0=0.0, excludeZero=True, roles=rolesMC):
    r"""
    2D version of alpha_Klist_2M1Ph_2D    
    """
    ### k0 must be =0 to use this simple 2-grid method
    ### g is not implemented
    k0 = np.zeros(3, dtype=np.float64)
    Nkx, Nky, Nkz, Nb = EgrM.shape
    Nbranch = EgrPh.shape[3]
    Ein = ek0
    ib = 0
    ibranch = 0

    ### it is sometimes important not to include k=0 phonons
    def CheckZero(ix,iy,iz):
        if not(excludeZero):
            return True
        else:
            si1 = np.sign( Kgr[ix,iy,iz][0]*Kgr[ix+1,iy+1,iz+1][0]  )
            si2 = np.sign( Kgr[ix,iy,iz][1]*Kgr[ix+1,iy+1,iz+1][1]  )
            si3 = np.sign( Kgr[ix,iy,iz][2]*Kgr[ix+1,iy+1,iz+1][2]  )
            return ( (si1>0) or (si2>0) or (si3>0) )

    def dExyz(ix,iy,iz):
        Emagn1 = EgrM[ix,iy,iz, ib]
        Ephon = EgrPh[ix,iy,iz, ibranch] 
        deltE = roles[0]*Ein + roles[1]*Emagn1 + roles[2]*Ephon
        return deltE
        
    for ib in range(Nb):
        for ibranch in range(Nbranch):
            for ix in range(Nkx-1):
                for iy in range(Nky-1):
                    for iz in range(Nkz-1):
                        co1 = np.array((ix,iy,iz), dtype=np.int32)
                        evec = np.array([dExyz(*(co1 + argArray3D[ii])) for ii in range(8)])
                        if (not ut2.SameSign((evec))) and (CheckZero(ix,iy,iz)):
                            k1Corn = np.zeros((8,3), dtype=np.float64)
                            for ii in range(8):
                                k1Corn[ii] = Kgr[ix+argArray3D[ii,0], iy+argArray3D[ii,1], iz+argArray3D[ii,2]]
                            newScat = Tscat2M1Ph(k0, lam0,  k1Corn, ib, ibranch, sum_k=zerovec, roles=roles, dim=3)
                            scat_list.append(newScat)
    return scat_list





def findCenter2M1Ph(Scat, pSH, Ephon):
    r"""
    finds a real k1 and q vectors corresponding to a scattering event
    Scat - scattering event
    pSH - numba-compatible spin Hamiltonian
    Ephon - phonons acording to UtilPhonopy
    """
    roles = Scat.roles
    gph = Scat.sum_k
    Scat.e0 = prad.omega(pSH, Scat.k0)[0][Scat.lk0]
    k0 = Scat.k0
    Ein = -Scat.e0*roles[0]

    def dEv(k1):
        q = Scat.sum_k -Scat.roles[2]*( Scat.roles[1]*k1 + Scat.roles[0]*Scat.k0)
        de1 = deltaE2M1Ph(Ein, k1, Scat.lk1, pSH, q, Scat.qBra, Ephon, Scat.roles)
        return de1

    Exist, k1real = gmath.FindZeroPoint(dEv, Scat.k1corners) 
    
    if Exist:
        qreal = Scat.sum_k -Scat.roles[2]*( Scat.roles[1]*k1real + Scat.roles[0]*Scat.k0)
        Scat.Exist = 1
        Scat.k1real = k1real.copy()
        Scat.qreal = qreal.copy()
        Scat.e1=prad.omega(pSH, k1real)[0][Scat.lk1]
        Scat.ePh= Ephon.energy(qreal, Scat.qBra)    #prad.omega(pSH, k2real)[0][Scat.lk2]
    else:
        k1real = np.array((0.0,0.0,0.0), dtype=np.float64)
        Scat.k1real = k1real.copy()
        Scat.qreal = k1real.copy()
        print('Ahtung: scattering is not there!')
    return Exist, k1real


def findDeltaInt2M1Ph(Scat, pSH, Ephon):
    r"""
    finds the integral of the delta-function for a scattering event
    
    Scat - scattering event
    pSH - numba-compatible spin Hamiltonian
    Ephon - phonons acording to UtilPhonopy

    calls one of the procedures
    findDeltaInt2M1Ph_2D
    findDeltaInt2M1Ph_3D
    """
    if Scat.dim == 3:
        return findDeltaInt2M1Ph_3D(Scat, pSH, Ephon)
    else:
        return findDeltaInt2M1Ph_2D(Scat, pSH, Ephon)



#### 2D procedure
def findDeltaInt2M1Ph_2D(Scat, pSH, Ephon):
    r"""
    2D version of findDeltaInt2M1Ph
    """
    Ein = -Scat.e0*Scat.roles[0]
    vals = np.zeros(4)
    for i in range(4):
        k1 = Scat.k1corners[i]
        q = Scat.sum_k -Scat.roles[2]*( Scat.roles[1]*k1 + Scat.roles[0]*Scat.k0)        
        vals[i] = deltaE2M1Ph(Ein, k1, Scat.lk1, pSH, q, Scat.qBra, Ephon, Scat.roles)
    
    res = smath.bilinDintV(Scat.k1corners, (vals[0], vals[1], vals[2], vals[3])  )
    Scat.DeltaInt = res
    return res



def findDeltaInt2M1Ph_3D(Scat, pSH, Ephon):
    r"""
    3D version of findDeltaInt2M1Ph
    """
    Ein = -Scat.e0*Scat.roles[0]

    vals = np.zeros(8)
    for i in range(8):
        k1 = Scat.k1corners[i]
        q = Scat.sum_k - Scat.roles[2]*( Scat.roles[1]*k1 + Scat.roles[0]*Scat.k0)        
        vals[i] = deltaE2M1Ph(Ein, k1, Scat.lk1, pSH, q, Scat.qBra, Ephon, Scat.roles)
    dk1 = Scat.k1corners[4] - Scat.k1corners[0]
    dk2 = Scat.k1corners[2] - Scat.k1corners[0]
    dk3 = Scat.k1corners[1] - Scat.k1corners[0]
    res = bmath.DeltaIntCell(vals, dk1,dk2,dk3)
    Scat.DeltaInt = res
    return res



################# save/load system for scatterers #############################

### non-numba scatterer class
### can be automatically saved by  pickle library 
class Pscatter:
    def __init__(self, s):
        self.k0        = s.k0       
        self.k1corners = s.k1corners
        
        self.sum_k     = s.sum_k 
        self.sum_e     = s.sum_e 
        self.dim       = s.dim   
        
        self.lk0       = s.lk0   
        self.lk1       = s.lk1   
        self.qBra      = s.qBra  
        self.roles     = s.roles 

        self.k1real    = s.k1real  
        self.qreal     = s.qreal   
        self.Exist     = s.Exist   
        self.DeltaInt  = s.DeltaInt
        self.e0        = s.e0      
        self.e1        = s.e1      
        self.ePh       = s.ePh     
    
        self.Mel       = s.Mel   
        self.Status    = s.Status

def NumbaScatter(psc):
    sc = Tscat2M1Ph(psc.k0, psc.lk0, psc.k1corners, psc.lk1, psc.qBra, sum_k=psc.sum_k, roles=psc.roles, dim=psc.dim, sum_e = psc.sum_e )

    sc.k1real   = psc.k1real  
    sc.qreal    = psc.qreal   
    sc.Exist    = psc.Exist   
    sc.DeltaInt = psc.DeltaInt
    sc.e0       = psc.e0      
    sc.e1       = psc.e1      
    sc.ePh      = psc.ePh     

    sc.Mel    = psc.Mel   
    sc.Status = psc.Status
    return sc


def saveSlist(lis, file):
    plis = []
    for sc in lis:
        psc = Pscatter(sc)
        plis.append(psc)
    
    with open(file, 'wb') as fp:
        pickle.dump(plis, fp)
    return plis
    
def loadSlist(file):
    with (open(file, "rb")) as openfile:
        plis = pickle.load(openfile)            
    
    lis1 = EmptyScatList2M1Ph()
    for psc in plis:
        lis1.append( NumbaScatter(psc) )
    return lis1







