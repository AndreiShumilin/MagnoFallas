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
A collection of utilities to work with a ground state of the system (for ex., to calculate 
anisotropy energy)
"""


import numpy as np
import scipy as sp

from MagnoFallas.Utils import util2 as ut2
from MagnoFallas.Utils import DipoleDipole as dd





DirectionList1 = [(1,0,0), (0,1,0),(0,0,1) ]

def onlyDDHamiltonian(SH, R0, dim, gStrategy="2", gClust=None, gValues=None ):
    r"""
    Suppresses exchange interaction and adds SRDD up to the cutoff distance R0
    used for calculation of shape anisotropy 

    dim - dimensionality of the sample

    gStrategy --- strategy for calculating g-factors, should be one of:
        "2" - all g-factors equal 2 (probably due to weak SOC)
        "Magn" - g-factors are calculated from Magnetization values in TB2J
        "Cluster" - also from TB2J, but each spin is associated with a cluster of atoms, the "maps" of the clausters should be provided
                     in gClust
        "Values" - user-provided values of g-factors. Must be in gValues 
    """
    SHt = ut2.cloneSH(SH, Multiply=ut2.Global_zero**2)
    SHdd = dd.AddSRDD(SHt, R0, dim=dim, gStrategy=gStrategy, gClust=gClust, gValues=gValues)
    return SHdd


def randomNormalDirection(vec1):
    r"""
    Generates some direction, which is perpendicular to vec1
    """
    evec1 = vec1/np.linalg.norm(vec1)
    if np.abs(1 - np.abs((evec1@ut2.ez)))<ut2.Global_zero**2:
        return(ut2.ex)
    else:
        v1 = np.cross(evec1, ut2.ez)
        ev1 = v1/np.linalg.norm(v1)
        return ev1

def vecs_to_RM(vec1, vec2):
    r"""
    for arbitrary vectors vec1 and vec2
    generates a transformation matrix which would rotate vec1 to the direction of vec2
    """
    evec1 = vec1/np.linalg.norm(vec1)
    evec2 = vec2/np.linalg.norm(vec2)
    Vcross = np.cross(evec1, evec2)
    aVcross = np.linalg.norm(Vcross)
    if aVcross > ut2.Global_zero:
        erot = Vcross/aVcross
        angle = np.arcsin(aVcross)
        if vec1@vec2<0:
            angle = np.pi-angle
        rv2 = erot*angle
        R = sp.spatial.transform.Rotation.from_rotvec(rv2)
        M = R.as_matrix()
        return M
    elif (evec1@evec2) > 0:
        return(np.eye(3))
    else:
        erot = randomNormalDirection(evec1)
        angle = np.pi
        rv = angle*erot
        R = sp.spatial.transform.Rotation.from_rotvec(rv)
        M = R.as_matrix()
        return M

def checkNotations(SH):
    dcou, snorm, factor = SH.notation
    good = (dcou) and (not snorm) and (factor==-1)
    return good
    

def TotalEnergy(SH, spins=None, simple=True, LR=False, R0 = -1.0, dim=2, gStrategy="2", gClust=None, gValues=None ):
    r"""
    Calculates the total energy of the spin Hamiltonian (SH) with some arbitrary direction of spins.
    Two versions are possible:
    simple = True -> only the general direction for spins is set in spins. It will be trated as the direction of the 
            subblattice of the first magnetic atoms
    simple = False -> all spin vectors must be set manually

    other parameters: 
    
    LR - allows LRDD, requires:
            R0 > 0 - cutoff distance
            dim=2 - LRDD contribution to the total energy is zero in 3D samples with spherical symmetry
    Note: SRDD shopuld be already included into SH0
    
    --- used only when LR=True ---
    gStrategy --- strategy for calculating g-factors, should be one of:
        "2" - all g-factors equal 2 (probably due to weak SOC)
        "Magn" - g-factors are calculated from Magnetization values in TB2J
        "Cluster" - also from TB2J, but each spin is associated with a cluster of atoms, the "maps" of the clausters should be provided
                     in gClust
        "Values" - user-provided values of g-factors. Must be in gValues 
    """
    if simple:
        return TotalEnergySimple(SH, evec=spins, LR=LR, R0 = R0, dim=dim, gStrategy=gStrategy, gClust=gClust, gValues=gValues)
    else:
        return TotalEnergyAllS(SH, spins, LR=LR, R0 = R0, dim=dim, gStrategy=gStrategy, gClust=gClust, gValues=gValues)
    


def TotalEnergySimple(SH0, evec=None, LR=False, R0 = -1.0, dim=2, gStrategy="2", gClust=None, gValues=None):
    r"""
    Calculates the total energy of the spin Hamiltonian (SH0) when rotating the spins so that subblatics of the first spin becomes 
    polarized along evec

    LR - allows LRDD, requires:
            R0 > 0 - cutoff distance
            dim=2 - LRDD contribution to the total energy is zero in 3D samples with spherical symmetry
    Note: SRDD shopuld be already included into SH0
    
    --- used only when LR=True ---
    gStrategy --- strategy for calculating g-factors, should be one of:
        "2" - all g-factors equal 2 (probably due to weak SOC)
        "Magn" - g-factors are calculated from Magnetization values in TB2J
        "Cluster" - also from TB2J, but each spin is associated with a cluster of atoms, the "maps" of the clausters should be provided
                     in gClust
        "Values" - user-provided values of g-factors. Must be in gValues 
    """
    if checkNotations(SH0):
        SH = SH0
    else:
        SH = ut2.cloneSH(SH0)
        SH.notation = (True, False, -1.0)
    
    mvec1 = SH.magnetic_atoms[0].spin_vector
    if evec is None:
        evec= mvec1.copy()
    RM = vecs_to_RM(mvec1, evec)

    TotEn = 0.0
    for at1,at2, v, Jrad in SH:
        #print(at1, at2, v)
        sv1_0 = at1.spin_vector
        sv2_0 = at2.spin_vector
        sv1 = RM@sv1_0
        sv2 = RM@sv2_0
        Jmat = Jrad.matrix
        TotEn += sv1 @ (Jmat@sv2)

    if (LR) and (dim==2) and (R0>0):
        Nat = len(SH.magnetic_atoms)
        ## note: at dim=3, Long range part is always zero
        gFactors = dd.get_gFactors(SH, gStrategy=gStrategy, gClust=gClust, gValues=gValues) 
        LRmat0 = np.real(dd.longrangeDDmatr(np.zeros(3), R0, 2, g1=2, g2=2))
        Scell = ut2.cellVolume(SH.cell, regime2D=True)
        for iat1, at1 in enumerate(SH.magnetic_atoms):
            for iat2, at2 in enumerate(SH.magnetic_atoms):
                sv1_0 = at1.spin_vector
                sv2_0 = at2.spin_vector
                sv1 = RM@sv1_0
                sv2 = RM@sv2_0
                ee1 = sv1 @ (LRmat0@sv2)
                ee1 /= Scell
                ee1 /= 2
                ee1 *=   gFactors[iat1]*gFactors[iat2]/4
                TotEn += ee1

    TotEn *= -1
    return TotEn
    


def TotalEnergyAllS(SH0, allspinvec, LR=False, R0 = -1.0, dim=2, gStrategy="2", gClust=None, gValues=None):
    r"""
    Calculates the total energy of the spin Hamiltonian (SH0) with arbitrary spin vecotrs defined in allspinvec

    LR - allows LRDD, requires:
            R0 > 0 - cutoff distance
            dim=2 - LRDD contribution to the total energy is zero in 3D samples with spherical symmetry
    Note: SRDD shopuld be already included into SH0
    
    --- used only when LR=True ---
    gStrategy --- strategy for calculating g-factors, should be one of:
        "2" - all g-factors equal 2 (probably due to weak SOC)
        "Magn" - g-factors are calculated from Magnetization values in TB2J
        "Cluster" - also from TB2J, but each spin is associated with a cluster of atoms, the "maps" of the clausters should be provided
                     in gClust
        "Values" - user-provided values of g-factors. Must be in gValues 
    """
    if checkNotations(SH0):
        SH = SH0
    else:
        SH = ut2.cloneSH(SH0)
        SH.notation = (True, False, -1.0)

    TotEn = 0.0

    atDi = {}
    for iat, at in enumerate(SH.magnetic_atoms):
        atDi[at.name] = iat

    for at1,at2, v, Jrad in SH:
        #print(at1, at2, v)
        iat1 = atDi[at1.name]
        iat2 = atDi[at2.name]
        sv1 = allspinvec[iat1]
        sv2 = allspinvec[iat2]
        Jmat = Jrad.matrix
        TotEn += sv1 @ (Jmat@sv2)

    if (LR) and (dim==2) and (R0>0):
        Nat = len(SH.magnetic_atoms)
        ## note: at dim=3, Long range part is always zero
        gFactors = dd.get_gFactors(SH, gStrategy=gStrategy, gClust=gClust, gValues=gValues) 
        LRmat0 = np.real(dd.longrangeDDmatr(np.zeros(3), R0, 2, g1=2, g2=2))
        Scell = ut2.cellVolume(SH.cell, regime2D=True)
        for iat1, at1 in enumerate(SH.magnetic_atoms):
            for iat2, at2 in enumerate(SH.magnetic_atoms):
                sv1 = allspinvec[iat1]
                sv2 = allspinvec[iat2]
                ee1 = sv1 @ (LRmat0@sv2)
                ee1 /= Scell
                ee1 /= 2
                ee1 *=   gFactors[iat1]*gFactors[iat2]/4
                TotEn += ee1

    TotEn *= -1
    return TotEn









def ShapeEnergy(SH, R0=25.0, LR=False, dim=2, gStrategy="Magn", gClust=None, gValues=None, 
                DirList = DirectionList1, file=None, Show = True):
    r"""
    Calculates the Shape anisotropy energy by rotating spins in the hamiltinan "SH" along the directions in "DirList"
    (default = Cartesian axes)

    dim - dimensionality of the sample
    file - allows to save results in a text file
    Show - prints the results
    
    R0 - cutoff between SRDD and LRDD
    LR - allows LRDD (long-ranged part of dipole-dipole interaction)

    gStrategy, gClust, gValues - related to the identification of g-factors
    
    gStrategy --- strategy for calculating g-factors, should be one of:
        "2" - all g-factors equal 2 (probably due to weak SOC)
        "Magn" - g-factors are calculated from Magnetization values in TB2J
        "Cluster" - also from TB2J, but each spin is associated with a cluster of atoms, the "maps" of the clausters should be provided
                     in gClust
        "Values" - user-provided values of g-factors. Must be in gValues 
    """
    Hdd = onlyDDHamiltonian(SH, R0, dim, gStrategy=gStrategy, gClust=gClust, gValues=gValues )
    Hdd.notation = (True,False,-1)
    Elist = []
    for evec in DirList:
        Et = TotalEnergy(Hdd, evec, LR=LR, dim=dim,  R0 = R0 ,gStrategy=gStrategy, gClust=gClust, gValues=gValues)
        if Show:
            print(evec, ' : ', Et)
        Elist.append(Et)
    if not file is None:
        ND = len(DirList)
        SList = [ list(DirList[i])+[Elist[i]] for i in range(ND) ]
        SList = np.array(SList)
        np.savetxt(file, SList)
    return Elist



def SOCEnergy(SH, DirList = DirectionList1, file=None, Show = True, dirRef=ut2.ez):
    r"""
    Calculates the SOC anisotropy energy by rotating spins in the hamiltinan "SH" along the directions in "DirList"
    (default = Cartesian axes)

    file - allows to save results in a text file
    Show - prints the results
    dirRef - "reference" direction, will correspond to SOC energy = 0
    """
    H2 = ut2.cloneSH(SH)
    H2.notation = (True,False,-1)
    Eref = TotalEnergy(H2, dirRef)
    Elist = []
    for evec in DirList:
        Et = TotalEnergy(H2, evec) - Eref
        if Show:
            print(evec, ' : ', Et)
        Elist.append(Et)
    if not file is None:
        ND = len(DirList)
        SList = [ list(DirList[i])+[Elist[i]] for i in range(ND) ]
        SList = np.array(SList)
        np.savetxt(file, SList)
    return Elist




###-----------------------------------------------------------------------------------------------------------
def TotEnMinimize(SH, spins, simple=True):
    r"""
    Tries to minimize the total energy starting from spin directions written in "spins"
    based on the spin Hamiltonian SH
    !!! Note: the algorithm will stuck in meta-stable and saddle configuration, in particular no energy extremumus would be optimized !!!
    
    two vestions: 
    simple=True  --> spins: single direction related to the 1-st sub-lattice
    simple=False --> spins: all vectors of all the magnetic atoms (in the corresponding order)
    """
    if simple:
        return TotEnMinimizeSimple(SH, spins)
    else:
        return TotEnMinimizeFull(SH, spins)


###----------------------------------------------------------------------------------------------------------
def TotEnMinimizeSimple(SH, spin0):

    espin = np.asarray(spin0)
    espin /= np.linalg.norm(espin)

    En0 = TotalEnergy(SH, espin, simple=True)
    phi0 = np.arctan2(espin[1], espin[0])
    tet0 = np.arccos(espin[2])
    angles0 = np.array( (phi0, tet0) )

    bndsD = np.zeros(2)
    bndsU = np.array((np.pi, 2*np.pi))
    bnds = [(bndsD[i], bndsU[i]) for i in range(2)]
    
    def minfuu(angles):
        phi, tet = angles
        ev = np.array((np.cos(phi)*np.sin(tet), np.sin(phi)*np.sin(tet), np.cos(tet) ))
        En = TotalEnergy(SH, ev, simple=True)
        return En - En0

    opt1 = sp.optimize.minimize(minfuu, angles0, bounds=bnds, method='L-BFGS-B')
    angles1 = opt1['x']
    phi, tet = angles1
    evec1 = np.array((np.cos(phi)*np.sin(tet), np.sin(phi)*np.sin(tet), np.cos(tet) ))
    Etot = TotalEnergy(SH, evec1, simple=True)
    return evec1, Etot 


def TotEnMinimizeFull(SH, spins0):

    spins1 = np.asarray(spins0)
    Ns = len(spins1)

    En0 = TotalEnergy(SH, spins0, simple=False)
    
    AS = np.array([np.linalg.norm(sp) for sp in spins1])
    
    phis0 = []
    tet0 = []
    for isp,spi in enumerate(spins1):
        spN = spi/AS[isp]
        phi = np.arctan2(spN[1], spN[0])
        tet = np.arccos(spN[2])
        phis0.append(phi)
        tet0.append(tet)
    angles0 = np.array(phis0+tet0)

    bndsD = np.zeros(2*Ns)
    bndsU = np.zeros(2*Ns) + np.pi
    for i in range(Ns):
        bndsU[i] += np.pi
    bnds = [(bndsD[i], bndsU[i]) for i in range(2*Ns)]
    
    def angles2spins(angles):
        phis = angles[:Ns]
        tets = angles[Ns:]
        spinsA = np.array([ (AS[i]*np.cos(phis[i])*np.sin(tets[i]),
                             AS[i]*np.sin(phis[i])*np.sin(tets[i]),
                                                    AS[i]*np.cos(tets[i])) for i in range(Ns) ])
        return spinsA 
    
    
    def minfuu(angles):
        spins2 = angles2spins(angles)
        En = TotalEnergy(SH, spins2, simple=False)
        return En - En0

    opt1 = sp.optimize.minimize(minfuu, angles0, bounds=bnds, method='L-BFGS-B')
    ang1 = opt1['x']
    spinsF = angles2spins(ang1)
    Etot = TotalEnergy(SH, spinsF, simple=False)
    return spinsF, Etot 

###----------------------------------------------------------------------------------------------------------

