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
from numba import int32, float32, float64    # import the types
from numba.experimental import jitclass
from numba.typed import List
from scipy.spatial.transform import Rotation

from MagnoFallas.Interface import PseudoRad as prad
from MagnoFallas.Interface import UtilPhonopy as utph



@nb.jit(nopython=True)
def KGrid(Nkx, Nky, Nkz,  cell, rKXmax=1, rKYmax=1, rKZmax=1, regime2D=True):
    Kgrid1 = np.zeros((Nkx,Nky,Nkz,3), dtype=np.float64)
    a1 = cell[0,...]
    a2 = cell[1,...]
    a3 = cell[2,...]
    b1 = 2*np.pi*np.cross(a2, a3)/ np.dot(a1, np.cross(a2, a3))
    b2 = 2*np.pi*np.cross(a3, a1)/ np.dot(a2, np.cross(a3, a1))
    b3 = 2*np.pi*np.cross(a1, a2)/ np.dot(a3, np.cross(a1, a2))

    if regime2D:
        rKZmax1 = 0.0
    else:
        rKZmax1 = rKZmax
    
    akx = np.linspace(-rKXmax,rKXmax,Nkx )
    aky = np.linspace(-rKYmax,rKYmax,Nky )
    akz = np.linspace(-rKZmax1,rKZmax1,Nkz )
    for ix in range(Nkx):
        for iy in range(Nky):
            for iz in range(Nkz):
                Kgrid1[ix,iy,iz] = akx[ix]*(b1/2) + aky[iy]*(b2/2) + akz[iz]*(b3/2)
    ###gK = (cell[0,0]*rKXmax/np.pi)*(cell[1,1]*rKYmax/np.pi)/(Nkx*Nky)   ####???????  old version  ????????????????

    if regime2D:
        a12 = np.cross(a1,a2)
        Sa = np.sqrt(np.sum(a12*a12))
        b12 = np.cross(b1,b2)
        Sb = np.sqrt(np.sum(b12*b12))
        gK = Sa*Sb/(4*np.pi*np.pi*Nkx*Nky)
        gK *= rKXmax*rKYmax
    else:
        a12 = np.cross(a1,a2)
        Va = np.dot(a12,a3)
        b12 = np.cross(b1,b2)
        Vb = np.dot(b3,b12)
        gK = (Va*Vb)/((2*np.pi)**3 *Nkx *Nky *Nkz)
        gK *= rKXmax*rKYmax*rKZmax
    return Kgrid1, gK


@nb.jit(nopython=True)
def Egrid_prad(pSH, Kgr):
    ## pSH - pseudo rad-tools spin-Hamiltonian
    Nb = pSH.Nat
    Nkx, Nky, Nkz, Nd = Kgr.shape
    Egrid = np.zeros((Nkx,Nky, Nkz, Nb))
    for ix in range(Nkx):
        for iy in range(Nky):
            for iz in range(Nkz):
                k1 = Kgr[ix,iy,iz]
                omg, G, Gi = prad.omega(pSH, k1)
                Egrid[ix,iy,iz,...] = omg[:Nb]
    return Egrid



# @nb.jit(nopython=True)
# def Egrid2D_prad(pSH, Kgr):
#     ## pSH - pseudo rad-tools spin-Hamiltonian
#     Nb = pSH.Nat
#     Nkx, Nky, Nd = Kgr.shape
#     Egrid = np.zeros((Nkx,Nky,Nb))
#     for ix in range(Nkx):
#         for iy in range(Nky):
#             k1 = Kgr[ix,iy]
#             omg, G, Gi = prad.omega(pSH, k1)
#             Egrid[ix,iy,...] = omg[:Nb]
#     return Egrid


def Egrid_Ephonon(Ephon, Kgr, MaxBranch = None):
    ## pSH - pseudo rad-tools spin-Hamiltonian
    Nkx, Nky, Nkz, Nd = Kgr.shape
    if MaxBranch is None:
        Nbranch = Ephon.Nband
    else:
        Nbranch = MaxBranch
    Egrid = np.zeros((Nkx,Nky,Nkz, Nbranch))
    for ix in range(Nkx):
        for iy in range(Nky):
            for iz in range(Nkz):
                k1 = Kgr[ix,iy,iz]
                energies = Ephon.energies_all(k1, isreal=True)
                for ib in range(Nbranch):
                    #en = Ephon.energy(k1, ib, isreal=True)
                    Egrid[ix,iy,iz,ib] = energies[ib]
    return Egrid


# def Egrid2D_Ephonon(Ephon, Kgr):
#     ## pSH - pseudo rad-tools spin-Hamiltonian
#     Nkx, Nky, Nd = Kgr.shape
#     Nbranch = Ephon.Nband
#     Egrid = np.zeros((Nkx,Nky,Nbranch))
#     for ix in range(Nkx):
#         for iy in range(Nky):
#             k1 = Kgr[ix,iy]
#             for ib in range(Nbranch):
#                 en = Ephon.energy(k1, ib, isreal=True)
#                 Egrid[ix,iy,ib] = en
#     return Egrid





########## The following procedures are for automatic selection of (magnon) grid when only low energies and acostic magnons are of interest
@nb.jit(nopython=True)
def rel_kMAx(en, lam, b_vec0, pSH, tol=0.01):
    ### k1*b_vec that leads to energy en*multi  (at the branch lam)
    ## useful only for acoustic magnnos
    
    Cut = False   #### shows if we are going to "Cut", i.e. reduce reciprocal cell to efficiently include energies up to en

    k0 = 0.0
    k1 = 1.0
    b_vec = b_vec0/2
    kv1 = k1*b_vec
    kv0 = np.array((0.0,0.0,0.0), dtype=np.float64)
    e0 = prad.omega(pSH, kv0)[0][lam]
    e1 = prad.omega(pSH, kv1)[0][lam]
    if e1<=e0:
        return(Cut, k1)  ##definintely not acoustic magnons
    if e1<= (en + tol): 
        return(Cut, k1)  ##energy en is large enough, we have to include whole band
    if e0>= en: 
        return(Cut, k1)  ## energy is smaller that E(\Gamma-point), probably not acoustic magnons
    else:
        Cut=True
        eGues = e1
        kGues = k1
        while np.abs(eGues-en) > tol:
            k2 = (k0+k1)/2
            kv2 = k2*b_vec
            e2 = prad.omega(pSH, kv2)[0][lam]
            if e2>en:
                k1 = k2
                kv1 = kv2
                kGues = k1
                eGues = e2
            else:
                k0 = k2
                kv0 = kv2
                kGues = k0
                eGues = e2
        return Cut, kGues



def acousticGrid(en, lam, SH, pSH, Nx=32, Ny=32, Nz=1, regime2D=True, multi=1.0, tol=0.01):
    en1 = en*multi
    b1, b2, b3 = SH.b1, SH.b2, SH.b3
    sucx, rKXM = rel_kMAx(en1, lam, b1, pSH, tol=tol)
    sucy, rKYM = rel_kMAx(en1, lam, b2, pSH, tol=tol)
    if regime2D:
        sucz = True
        rKZM = 1.0
    else:
        sucz, rKZM = rel_kMAx(en1, lam, b3, pSH, tol=tol)
    
    Kgr, gK = KGrid(Nx, Ny, Nz,  SH.cell, rKXmax=rKXM, rKYmax=rKYM, rKZmax=rKZM, regime2D=regime2D)
    return Kgr, gK




@nb.jit(nopython=True)
def acousticGrid_p(en, lam, bvecs, cell, pSH, Nx=32, Ny=32, Nz=1, regime2D=True, multi=1.0, tol=0.01):
    en1 = en*multi
    b1, b2, b3 = bvecs
    sucx, rKXM = rel_kMAx(en1, lam, b1, pSH, tol=tol)
    sucy, rKYM = rel_kMAx(en1, lam, b2, pSH, tol=tol)
    if regime2D:
        sucz = True
        rKZM = 1.0
    else:
        sucz, rKZM = rel_kMAx(en1, lam, b3, pSH, tol=tol)
    
    Kgr, gK = KGrid(Nx, Ny, Nz,  cell, rKXmax=rKXM, rKYmax=rKYM, rKZmax=rKZM, regime2D=regime2D)
    return Kgr, gK




