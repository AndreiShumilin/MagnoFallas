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
Module for working with scattering events of 4-magnon processes
Not supposed to interact with user
"""



import numpy as np
import numpy.linalg
import scipy as sp
import pickle

import numba as nb
from numba.experimental import jitclass

import MagnoFallas.OldRadtools as rad

from MagnoFallas.Interface import PseudoRad as prad

from MagnoFallas.Utils import util2 as ut2
from MagnoFallas.Utils import Grids

from MagnoFallas.Math import SurfaceMath as smath
from MagnoFallas.Math import BulkMath as bmath
from MagnoFallas.Math import generalMath as gmath



__all__ = ['Tscat','SingleKlist2D','rolesMC','rolesMNC']


zerovec = np.zeros(3, dtype = np.float64)
argArray3D = np.array([bmath.nfTonxyz(i) for i in range(8)], dtype=np.int32)

rolesMC = np.array((-1,-1,1,1), dtype=np.int32)    
rolesMNC = np.array((-1,-1,-1, 1), dtype=np.int32)

TscatteringLST = [
    ('k0', nb.float64[::1]),              
    ('lk0', nb.int32),
    ('k1', nb.float64[::1]),              
    ('lk1', nb.int32),         
    ('k2corners', nb.float64[:,::1]),  #### 2D order: BL - TL - BR - TR
    ('lk2', nb.int32),
    ('lk3', nb.int32),          ##band index for k3
    ('k3g', nb.float64[::1]),   ## k-lattice vector for k3   
    ('dim', nb.int32),    #### dimensionality, 2 or 3
###########################################################
    ('roles', nb.int32[::1]),   ### -1 if the magon disappears, +1 if it appears
###########################################################
    ('k2real', nb.float64[::1]),
    ('k3real', nb.float64[::1]),
    ('Exist', nb.int32),
    ('DeltaInt', nb.float64),
###########################################################
    ('e0', nb.float64),
    ('e1', nb.float64),
    ('e2', nb.float64),
    ('e3', nb.float64),
###########################################################
    ('Mel', nb.complex128),
###########################################################
    ('Status', nb.int32)
]

@jitclass(TscatteringLST)
class Tscat(object):
    r"""
    Numba-compatible class containig information about single scattering event
    """
    def __init__(self, k0, lk0, k1, lk1, k2corners, lk2, lk3, k3g, roles, dim):
        self.k0 = k0
        self.k1 = k1
        self.k2corners = k2corners
        self.k3g = k3g
        self.dim = dim
        
        self.lk0=lk0
        self.lk1=lk1
        self.lk2=lk2
        self.lk3=lk3

        self.roles = roles

        self.k2real = np.zeros(3)
        self.k3real = np.zeros(3)
        self.Exist = 0
        self.DeltaInt = -1000.0
        self.e0 = 0.0
        self.e1 = 0.0
        self.e2 = 0.0
        self.e3 = 0.0
        self.Mel = 0.0
        self.Status = 0

def EmptyScatList():
    k = np.array((1.0,0.0,0.0), dtype=np.float64)
    kCorn = np.array( (k,k,k,k) )
    sc1 = Tscat(k, 1, k, 1, kCorn, 1, 1, k, rolesMC, 2)
    lis = nb.typed.List()
    lis.append(sc1)
    lis.clear()
    return lis


@nb.jit(nopython=True)
def deltaE4M(Ein, k1, k2, k3, lam1, lam2,lam3, pSH, roles):
    r"""
    Calculate energy-difference for scattering event (correct energy diference is zero)
    """
    omg1 = prad.omega(pSH, k1)[0]
    omg2 = prad.omega(pSH, k2)[0]
    omg3 = prad.omega(pSH, k3)[0]
    dE = Ein*roles[0] + omg1[lam1]*roles[1] + omg2[lam2]*roles[2] + omg3[lam3]*roles[3]
    return dE



@nb.jit(nopython=True)
def SingleKlist2D(lam0, k1, lam1, Kgr, pSH, scat_list, roles, gph=np.array((0.0,0.0,0.0)), k0 = zerovec,  ek0=0.0, ek1=None, Nbmax=None):
    r"""
    Calculates the list of scattering events for a single (vector) value of k1
    2D version of the procedure
    
    lam0, lam1 - magnon modes for k0 and k1
    Kgr - K-grid
    pSH - "pseudorad" spin Hamiltonia
    roles - correspond to MC/MNC process
    gph - reciprocal lattice vector, the summ of k1-k4 (currently not used)
    ek0 - energy of the relaxing k0 magnon (always should be set)
    ek1 - energy of k1-magnon (can be found automatically)
    Nbmax - number of magnon bands to consider
    """
    Nkx, Nky, Nkz, Ndim = Kgr.shape
    Nb = pSH.Nat
    if Nbmax is None:
        Nbmax = Nb

    if ek1==None:
        ek1=prad.omega(pSH,k1)[0][lam1]
    Ein1 = ek0*roles[0] + ek1*roles[1]
    lam2 = 0
    lam3 = 0

    def dExy(x,y):
        k2 = np.array((x,y,0.0))
        k3 = roles[3]*(gph - roles[0]*k0 - roles[1]*k1  -  roles[2]*k2)
        omg2 = prad.omega(pSH, k2)[0]
        omg3 = prad.omega(pSH, k3)[0]
        dE = Ein1 + omg2[lam2]*roles[2] + omg3[lam3]*roles[3]
        return dE

    for lam2_0 in range(Nbmax):
        for lam3_0 in range(Nbmax):
            lam2 = Nb-1-lam2_0
            lam3 = Nb-1-lam3_0
            for ix in range(Nkx-1):
                for iy in range(Nky-1):
                    x11, y11, z11 = Kgr[ix,iy,0]
                    x12, y12, z12 = Kgr[ix,iy+1,0]
                    x21, y21, z21 = Kgr[ix+1,iy,0]
                    x22, y22, z22 = Kgr[ix+1,iy+1,0]
                    e11 = dExy(x11,y11)
                    e12 = dExy(x12,y12)
                    e21 = dExy(x21,y21)
                    e22 = dExy(x22,y22)
                    evec = np.array((e11,e12,e21,e22))
                    if not ut2.SameSign(evec):
                        k2BL = Kgr[ix,iy,0]
                        k2BR = Kgr[ix+1,iy,0]
                        k2TL = Kgr[ix,iy+1,0]
                        k2TR = Kgr[ix+1,iy+1,0]
                        k2Corn = np.zeros((4,3), dtype=np.float64)
                        k2Corn[0,...] = k2BL
                        k2Corn[1,...] = k2TL
                        k2Corn[2,...] = k2BR
                        k2Corn[3,...] = k2TR
                        
                        newScat = Tscat(k0,lam0, k1, lam1, k2Corn, lam2,lam3, zerovec, roles, 2)
                               
                        scat_list.append(newScat)
    return scat_list
    


def Sets4MLists2D(Kgr_in, SH, pSH, lam0, lam1,  ek0, g_rcell=np.array((0,0,0)), roles=rolesMC, 
                  acuRegime=False, acuDelCor=False, NKXac=32, NKYac=32, Ecut=None, Nbmax=None):
    r"""
    Calculates all th cattering events. 
    2D version of the procedure
    
    Kgr_in - K-grid for the wavevectror k1
    SH - "rad tools" spin Hamiltonian
    pSH - "pseudo - rad - tools" spin Hamiltonian
    lam0, lam1 - magnon modes for k0 and k1
    ek0 - energy of relaxing k0-magnon
    g_rcell - reciprocal lattice vector related to momentum conservation (currently not used)
    roles - correspond to MC/MNC process
    acuRegime - wether "acoustic-type" k-grids should be used for k2/k3
    
    Ecut - energy cut-off
    Nbmax - number of magnon bands to consider
    """
    #### scuDelCor - substract one e0 from the maximum energy for acoustic grid. may be relevant for magnon-conserving processes.
    Nx, Ny, Nz, Ndim =  Kgr_in.shape
    Nb = pSH.Nat
    if Nbmax is None:
        Nbmax = Nb
        
    gph = g_rcell[0]*SH.b1 + g_rcell[1]*SH.b2 + g_rcell[2]*SH.b3
    sets = {}
    ksets = {}
    for ix in range(Nx):
        for iy in range(Ny):
            k = Kgr_in[ix,iy,0]
            ek = prad.omega(pSH, k)[0][lam1]

            To_calc = Ecut is None
            if not (Ecut is None):
                To_calc = (ek < Ecut)

            if To_calc:
                if acuRegime:
                    Ein = -roles[0]*ek0 - roles[1]*ek
                    if acuDelCor:     ###### sometimes relevant for contribution maps
                        Ein -= ek0
                    Gr_k, gk_k = Grids.acousticGrid(Ein, lam1, SH, pSH, Nx=NKXac, Ny=NKYac, multi=1.05)
                    # print(k)
                    # print('acoustic worked : ', np.max(np.abs(Gr_k[...,0])), np.max(np.abs(Gr_k[...,1])))
                    # print('---------------')
                else:
                    Gr_k = Kgr_in
                list0 = EmptyScatList()
                list1 = SingleKlist2D(lam0, k, lam1, Gr_k, pSH, list0, roles, gph=gph,   ek0=ek0, ek1=ek, Nbmax=Nbmax)
                sets[(ix,iy,lam1)] = list1
                ksets[(ix,iy,lam1)] = k
    return sets, ksets


################# calculates the energy difference for the scattering##################
##-------------- expected to be close to zero for real value of k2 vector ---------------
def ScatdE(Scat, k2tmp, pSH):
    r"""
    calculates the energy difference for the scatterin
    expected to be close to zero for real value of k2 vector
    """    
    roles = Scat.roles
    gph = Scat.k3g
    k3 = roles[3]*(gph - roles[0]*Scat.k0 - roles[1]*Scat.k1  -  roles[2]*k2tmp)
    omg2 = prad.omega(pSH, k2tmp)[0]
    omg3 = prad.omega(pSH, k3)[0]
    Ein1 = Scat.e0*roles[0] + Scat.e1*roles[1]
    dE = Ein1 + omg2[Scat.lk2]*roles[2] + omg3[Scat.lk3]*roles[3]
    return dE


########### finds a real value for k2 of the scattering event
def findCenter2D(Scat, pSH):
    r"""
    finds the values of k2/k3 fulfilling the energy-conservation law
    2D version
    """ 
    roles = Scat.roles
    gph = Scat.k3g
    Scat.e0 = prad.omega(pSH, Scat.k0)[0][Scat.lk0]
    Scat.e1 = prad.omega(pSH, Scat.k1)[0][Scat.lk1]
    k0 = Scat.k0
    k1 = Scat.k1

    Ein1 = Scat.e0*roles[0] + Scat.e1*roles[1]
    lam2 = Scat.lk2
    lam3 = Scat.lk3

    def dEv(k2):
        k3 = roles[3]*(gph - roles[0]*k0 - roles[1]*k1  -  roles[2]*k2)
        omg2 = prad.omega(pSH, k2)[0]
        omg3 = prad.omega(pSH, k3)[0]
        dE = Ein1 + omg2[lam2]*roles[2] + omg3[lam3]*roles[3]
        return dE

    Exist, k2real = gmath.FindZeroPoint(dEv, Scat.k2corners) 
    
    if Exist:
        k3real = roles[3]*(gph - roles[0]*k0 - roles[1]*k1  -  roles[2]*k2real)
        Scat.Exist = 1
        Scat.k2real = k2real.copy()
        Scat.k3real = k3real.copy()
        Scat.e2=prad.omega(pSH, k2real)[0][Scat.lk2]
        Scat.e3=prad.omega(pSH, k3real)[0][Scat.lk3]
    else:
        k2real = np.array((0.0,0.0,0.0))
        print('Ahtung: scattering is not there!')
        print(dEv(Scat.k2corners[0]), dEv(Scat.k2corners[1]), dEv(Scat.k2corners[2]), dEv(Scat.k2corners[3]))
    return Exist, k2real




def findDeltaInt(Scat, pSH):
    if Scat.dim == 2:
        k2Cor = Scat.k2corners
        vals = np.zeros(4)
        for i in range(4):
            vals[i] = ScatdE(Scat, k2Cor[i], pSH)
        
        res = smath.bilinDintV(k2Cor, (vals[0], vals[1], vals[2], vals[3])  )
        Scat.DeltaInt = res
        return res
    elif (Scat.dim ==3):
        k2Cor = Scat.k2corners
        vals = np.zeros(8)
        for i in range(8):
            vals[i] = ScatdE(Scat, k2Cor[i], pSH)
        dk1 = k2Cor[4] - k2Cor[0]
        dk2 = k2Cor[2] - k2Cor[0]
        dk3 = k2Cor[1] - k2Cor[0]
        res = bmath.DeltaIntCell(vals, dk1,dk2,dk3)
        Scat.DeltaInt = res
        return res
    else:
        return 0.0





####----------------------------------------------------------------------####
################ 3D versions of the procedures ###############################
####----------------------------------------------------------------------####

@nb.njit(parallel=True)
def SingleKlist3D(lam0, k1, lam1, Kgr, pSH, scat_list, roles, gph=np.array((0.0,0.0,0.0)), k0 = zerovec,  ek0=None, ek1=None, Nbmax=None):
    ## note 3D case requires different (3D) math for search of zero-point
    Nkx, Nky, Nkz, Ndim = Kgr.shape
    Nb = pSH.Nat
    if Nbmax is None:
        Nbmax = Nb

    if ek1==None:
        ek1=prad.omega(pSH,k1)[0][lam1]
    if ek0==None:
        ek0=prad.omega(pSH,k0)[0][lam0]
    
    Ein1 = ek0*roles[0] + ek1*roles[1]
    lam2 = 0
    lam3 = 0

    #----- temporary energy Grid reduces the ammount of calls to dEcal
    tEgr = np.zeros((Nkx,Nky,Nkz, Nb, Nb ), dtype=np.float64)
    for ix in range(Nkx):
        for iy in range(Nky):
            for iz in range(Nkz):
                tk2 = Kgr[ix,iy,iz]
                tk3 = roles[3]*(gph - roles[0]*k0 - roles[1]*k1  -  roles[2]*tk2)
                tomg2 = prad.omega(pSH, tk2)[0]
                tomg3 = prad.omega(pSH, tk3)[0]
                for l2_0 in range(Nbmax):
                    for l3_0 in range(Nbmax):
                        l2 = Nb - 1 - l2_0
                        l3 = Nb - 1 - l3_0
                        #tk2 = Kgr[ix,iy,iz]
                        #dE = dEcal(tk2, l2,l3)
                        dE = Ein1 + tomg2[l2]*roles[2] + tomg3[l3]*roles[3]
                        tEgr[ix,iy,iz,l2,l3] = dE

    for lam2_0 in range(Nbmax):
        for lam3_0 in range(Nbmax):
            lam2 = Nb - 1 - lam2_0
            lam3 = Nb - 1 - lam2_0
            for ix in range(Nkx-1):
                for iy in range(Nky-1):
                    for iz in range(Nkz-1):
                        evec = np.array([tEgr[ix+argArray3D[ii,0], iy+argArray3D[ii,1], iz+argArray3D[ii,2],lam2,lam3] for ii in range(8)], dtype=np.float64 )
                        if not ut2.SameSign(evec):
                            k2Corn = np.zeros((8,3), dtype=np.float64)
                            for ii in range(8):
                                k2Corn[ii] = Kgr[ix+argArray3D[ii,0], iy+argArray3D[ii,1], iz+argArray3D[ii,2]]
                                
                            newScat = Tscat(k0,lam0, k1, lam1, k2Corn, lam2,lam3, zerovec, roles, 3)
                            scat_list.append(newScat)
    return scat_list


def Sets4MLists3D(Kgr_in, SH, pSH, lam0, lam1,  ek0, g_rcell=np.array((0,0,0)), roles=rolesMC, 
                  acuRegime=False, acuDelCor=False, NKXac=8, NKYac=8, NKZac=8, Ecut=None, Nbmax=None, Log=None):
    #### acuDelCor - substract one e0 from the maximum energy for acoustic grid. may be relevant for magnon-conserving processes.
    Nx, Ny, Nz, Ndim =  Kgr_in.shape
    Nb = pSH.Nat
    if Nbmax is None:
        Nbmax = Nb

    if not Log is None:
        str1 = 'Starting the calculation of scattererers'
        Log.Twrite2(str1)
        lam1mod = Nb-1-lam1
        str1 = 'Number of band for magnon 1: ' + str(lam1mod)
        Log.write(str1)
        
    gph = g_rcell[0]*SH.b1 + g_rcell[1]*SH.b2 + g_rcell[2]*SH.b3
    sets = {}
    ksets = {}
    bvecs = (SH.b1, SH.b2, SH.b3)
    for ix in range(Nx):
        if not Log is None:
            proc = ix/Nx * 100
            proc_str = "{:.1f}".format(proc) + '%'
            str1 = proc_str + ' complete'
            Log.Twrite(str1)
        for iy in range(Ny):
            for iz in range(Nz):
                k = Kgr_in[ix,iy,iz]
                ek = prad.omega(pSH, k)[0][lam1]
    
                To_calc = (Ecut is None)
                if not (Ecut is None):
                    To_calc = (ek < Ecut)
    
                if To_calc:
                    if acuRegime:
                        Ein = -roles[0]*ek0 - roles[1]*ek
                        if acuDelCor:     ###### sometimes relevant for contribution maps
                            Ein -= ek0
                        Gr_k, gk_k = Grids.acousticGrid_p(Ein, lam1, bvecs, SH.cell, pSH, Nx=NKXac, Ny=NKYac, Nz=NKZac, regime2D=False, multi=1.05)
                    else:
                        Gr_k = Kgr_in
                        
                    list0 = EmptyScatList()
                    list1 = SingleKlist3D(lam0, k, lam1, Gr_k, pSH, list0, roles, gph=gph,   ek0=ek0, ek1=ek, Nbmax=Nbmax)
                    sets[(ix,iy,iz,lam1)] = list1
                    ksets[(ix,iy,iz,lam1)] = k
    return sets, ksets




#------------- finds a real value for k2 of the scattering event (3D version) -----------------
def findCenter3D(Scat, pSH):
    r"""
    finds the values of k2/k3 fulfilling the energy-conservation law
    3D version
    """ 

    roles = Scat.roles
    gph = Scat.k3g
    Scat.e0 = prad.omega(pSH, Scat.k0)[0][Scat.lk0]
    Scat.e1 = prad.omega(pSH, Scat.k1)[0][Scat.lk1]
    k0 = Scat.k0
    k1 = Scat.k1

    Ein1 = Scat.e0*roles[0] + Scat.e1*roles[1]
    lam2 = Scat.lk2
    lam3 = Scat.lk3

    def dEv(k2):
        k3 = roles[3]*(gph - roles[0]*k0 - roles[1]*k1  -  roles[2]*k2)
        omg2 = prad.omega(pSH, k2)[0]
        omg3 = prad.omega(pSH, k3)[0]
        dE = Ein1 + omg2[lam2]*roles[2] + omg3[lam3]*roles[3]
        return dE

    fval = np.zeros(8)
    for i in range(8):
        fval[i] = dEv(Scat.k2corners[i])
        
    v1, v2 = bmath.BestLine(fval, np.zeros(3), np.zeros(3)+1.0)
    iv1, iv2 = bmath.vecTonf(v1), bmath.vecTonf(v2)
    # k2_lbd = Scat.k2corners[0]
    # k2_rtu = Scat.k2corners[7]
    # k2t1 = k2_lbd + (k2_rtu-k2_lbd)*v1
    # k2t2 = k2_lbd + (k2_rtu-k2_lbd)*v2
    k2t1 = Scat.k2corners[iv1]
    k2t2 = Scat.k2corners[iv2]

    Exist, k2real = gmath.FindZeroPoint(dEv, (k2t1,k2t2)) 
   
    if Exist:
        k3real = roles[3]*(gph - roles[0]*k0 - roles[1]*k1  -  roles[2]*k2real)
        Scat.Exist = 1
        Scat.k2real = k2real.copy()
        Scat.k3real = k3real.copy()
        Scat.e2=prad.omega(pSH, k2real)[0][Scat.lk2]
        Scat.e3=prad.omega(pSH, k3real)[0][Scat.lk3]
    else:
        k2real = np.array((0.0,0.0,0.0))
        print('Ahtung: scattering is not there!')
        print(dEv(Scat.k2corners[0]), dEv(Scat.k2corners[1]), dEv(Scat.k2corners[2]), dEv(Scat.k2corners[3]), dEv(Scat.k2corners[4]), dEv(Scat.k2corners[5]),dEv(Scat.k2corners[6]), dEv(Scat.k2corners[7]))
    return Exist, k2real



################# save/load system for scatterers #############################


### non-numba scatterer class
### can be automatically saved by  pickle library 
class Pscatter:
    def __init__(self, s):
        self.k0 = s.k0
        self.lk0 = s.lk0
        self.k1 = s.k1
        self.lk1 = s.lk1
        self.k2corners = s.k2corners
        self.lk2 = s.lk2
        self.lk3 = s.lk3
        self.k3g = s.k3g
        self.dim = s.dim
        ################
        self.roles = s.roles
        ################
        self.k2real = s.k2real
        self.k3real = s.k3real
        self.Exist = s.Exist
        self.DeltaInt = s.DeltaInt
        ################
        self.e0 = s.e0
        self.e1 = s.e1
        self.e2 = s.e2
        self.e3 = s.e3
        #################
        self.Mel = s.Mel
        ##############
        self.Status = s.Status


def NumbaScatter(psc):
    sc = Tscat(psc.k0, psc.lk0, psc.k1, psc.lk1, psc.k2corners, psc.lk2, psc.lk3, psc.k3g, psc.roles, psc.dim) 
    
    sc.k2real = psc.k2real
    sc.k3real = psc.k3real
    sc.Exist = psc.Exist
    sc.DeltaInt = psc.DeltaInt
    #c############
    sc.e0 = psc.e0
    sc.e1 = psc.e1
    sc.e2 = psc.e2
    sc.e3 = psc.e3
    #c#############
    sc.Mel = psc.Mel
    #c##########
    sc.Status = psc.Status
    return sc

def saveSlist(inSlists, file):
    pSlists = {}
    for k, lis in inSlists.items():
        plis = []
        for sc in lis:
            psc = Pscatter(sc)
            plis.append(psc)
        pSlists[k] = plis
    with open(file, 'wb') as fp:
        pickle.dump(pSlists, fp)
    return pSlists
    
def loadSlist(file):
    inSlists = {}
    with (open(file, "rb")) as openfile:
        pSlists = pickle.load(openfile)            
    for klis, lis0 in pSlists.items():
        lis1 = EmptyScatList()
        for psc in lis0:
            lis1.append( NumbaScatter(psc) )
        inSlists[klis] = lis1
    return inSlists




