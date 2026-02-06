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
from copy import deepcopy
import copy

from MagnoFallas.Interface import PseudoRad as prad
import MagnoFallas.OldRadtools as rad

from MagnoFallas.Boltzmann.process2M1Ph.ScatteringList2M1Ph import rolesMC as rolesMC_2P1M
from MagnoFallas.Boltzmann.process2M1Ph.ScatteringList2M1Ph import rolesMNC as rolesMNC_2P1M

from MagnoFallas.Boltzmann.process4M.ScatteringList4M import rolesMC as rolesMC_4M
from MagnoFallas.Boltzmann.process4M.ScatteringList4M import rolesMNC as rolesMNC_4M

### phonon-dependent
# try:
#     from MagnoFallas.Interface import UtilPhonopy as utph
#     util_has_phonons = True
# except ImportError:
#     util_has_phonons = False



__all__ = ['K_to_mev','gaus_delta','Boze','quasimom','matconj','KGrid','KGrid2D','Egrid_prad','Egrid2D_prad','Egrid_Ephonon','Egrid2D_Ephonon','ModGi', 'SameSign','quasimom','rel_kMAx','acousticGrid']


K_to_mev = 0.0862
eV_to_J = 1.602176634e-19
mev_to_J = 1.602176634e-22
Tesl_to_mev = 5.788e-2

### internal velocity is in meV*A
### we need it in m/s
### Vel_Const = meV/[hbar * 10^10]
Vel_Const = 151.9267445016502


Global_zero = 1e-6


ex = np.array((1.0,0.0,0.0), dtype = np.float64)
ey = np.array((0.0,1.0,0.0), dtype = np.float64)
ez = np.array((0.0,0.0,1.0), dtype = np.float64)
ebasis = np.array((ex,ey,ez), dtype = np.float64)


rolesDict={'PhononMC' : rolesMC_2P1M,
           'PhononNMC' : rolesMNC_2P1M,
           'MagnonMC' : rolesMC_4M,
           'MagnonNMC' : rolesMNC_4M}


@nb.njit
def gaus_delta(e, de = 0.1):
    return np.exp(-e*e/(2*de*de))/ (np.sqrt(2*np.pi) * de )


@nb.njit
def Boze(x,T=1.0):
    return 1/(np.exp(x/T)-1)


@nb.njit
def matconj(A):
    return np.transpose(np.conjugate(A))

@nb.njit
def cellVolume(cell, regime2D=True):
    a1 = cell[0,...]
    a2 = cell[1,...]
    a3 = cell[2,...]

    if regime2D:
        a12 = np.cross(a1,a2)
        cellV = np.sqrt(np.sum(a12*a12))
    else:
        a12 = np.cross(a1,a2)
        cellV = np.dot(a12,a3)
    return np.abs(cellV)

@nb.jit(nopython=True)
def SameSign2D(a,b,c,d):
    if ((a>0) and (b>0) and (c>0) and (d>0)):
        return True
    elif ((a<0) and (b<0) and (c<0) and (d<0)):
        return True
    else:
        return False

@nb.jit(nopython=True)
def SameSign(arr):
    N = len(arr)
    cP = 0
    cM = 0
    for i in range(N):
        if arr[i] > 0:
            cP += 1
        elif arr[i] < 0:
            cM += 1
    if (cP == N):
        return True
    elif (cM == N):
        return True
    else:
        return False




@nb.njit
def quasimom(k, rec_cell):
    k_rel = np.linalg.solve(rec_cell, k)
    k_rel_1 = (k_rel +0.5) % 1 - 0.5
    k1 = rec_cell @ k_rel_1
    return k1


@nb.njit
def ModGi(G,k,positions):
    r"""
    The modification of the inverse matrix G of rad-tools (= parameter G here) to absorb real positions of atoms into definition of 
    on-atom magnon operators

    The initial rad-tools (including pseudo-) transforms the Hamiltonian from the basis where wavevector k is applied only to the 
    unit cell coordinates. ex1 includes exp(-ik r_n) to each operator related to site n
    the second exponent ex2 adds some phases to the final operators of diagonalized Hamiltonian
    ex2 should not affect Boltzmann-level expressions
    """
    Nat = len(positions)
    ex1 = [k@(positions[i]) for i in range(Nat)]
    ex1 = np.array(ex1+ex1)
    ex1 = np.diag(np.exp(-1j*ex1))
    G1 = ex1@G
    ex1c = np.conjugate(ex1)

    phases = np.zeros(2*Nat)
    for i in range(Nat):
        phases[i] = np.angle(G1[0,i])
    for i in range(Nat,2*Nat):
        phases[i] = np.angle(G1[Nat,i])
    ex2 = np.diag(np.exp(-1j*phases))
    G2 = G1@ex2
    return G2, ex1c

##########################################################################################

@nb.njit
def find_distance(r1,r2,cell, ignoreZ = False):
    dr = r1-r2
    if ignoreZ:
        dr = dr*np.array((1,1,0))
    drmod = quasimom(dr, cell)
    return np.linalg.norm(drmod)

@nb.njit
def find_atom_number(positions, coo, cell, dmax=0.5, dr = np.array((0,0,0))):
    n = -1000
    coo1 = coo+dr
    for iat,pos in enumerate(positions):
        d1 = find_distance(coo1, pos, cell)    
        if d1<dmax:
            n = iat
    return n


def TotEn(SH):
    En = 0.0
    for at1,at2, v, Jrad in SH:
        sv1 = at1.spin_vector
        sv2 = at2.spin_vector
        Jmat = Jrad.matrix
        En += sv1 @ (Jmat@sv2)
    return En


##-------------------------------------------------------------------------------------------------------


def realposition(SH, at):
    r"""
    calculates real position of the atom "at" if relative ones are given inside spin Hamiltonian "SH"
    """
    vec = np.zeros(3)
    vec += at.position[0]*SH.a1
    vec += at.position[1]*SH.a2
    vec += at.position[2]*SH.a3
    return vec


def GetRealPositions(SH):
    r"""
    all "real" positions (in angstr) of magnetic atoms in a spin Hamiltoinan SH
    """
    realpos = []
    for at in SH.magnetic_atoms:
        realpos.append( realposition(SH, at)  )
    return realpos


@nb.njit
def realposition2(pos, cell):
    r"""
    Calculates real (Cartesian) from the relative postion "pos" and unti cell "cell"
    """
    vec = np.zeros(3)
    vec += pos[0]*cell[0]
    vec += pos[1]*cell[1]
    vec += pos[2]*cell[2]
    return vec

@nb.njit
def PutIntoCell(cell, co):
    r"""
    puts "cartesian" coordinates "co" into unit cell "cell"
    """
    pos = np.linalg.solve(np.transpose(cell), co)
    pos2 = np.array(( pos[0]%1, pos[1]%1, pos[2]%1  ))
    co2 = realposition2(pos2, cell)
    return co2

@nb.njit
def PutIntoCell2(cell, co):
    r"""
    puts "cartesian" coordinates "co" into unit cell "cell"
    returns new coordinates and (and) displacement vector in untic cell vectors
    """
    pos = np.linalg.solve(np.transpose(cell), co)
    pos2 = np.array(( pos[0]%1, pos[1]%1, pos[2]%1  ))
    dv = np.array(( pos[0]//1, pos[1]//1, pos[2]//1  ), dtype = np.int32)
    co2 = realposition2(pos2, cell)
    return co2, dv

@nb.njit
def RelativePosition(cell, co):
    r"""
    calculates the relative position in the "cell" from Cartesian coordinates "co"
    """
    pos = np.linalg.solve(np.transpose(cell), co)
    return pos


##-------------------------------------------------------------------------------------------------------


def removeDigits(s):
    r"""
    removes all numbers from a string
    """
    result = ''.join([i for i in s if not i.isdigit()])
    return result

@nb.njit
def EqualVectors(v1, v2, prec=1e-3):
    r"""
    chacks the equivalence of two vecotrs with a precission prec
    """
    val = np.linalg.norm(v1-v2)
    return val<prec

##------------procedures to calculate group velocities from rad.MagnonDispersion------------------

def groupV(Mag, k, ib, dk=1e-2):
    r"""
    calculates vector group velocity [m/s] at wavevector k and band ib
    uses rad - MagnonDispersion Mag
    dk - step for calculation of numerical derivative dE/dk
    """
    v = np.zeros(3)
    for alp in range(3):
        om1 = Mag.omega(k - dk*ebasis[alp]/2)[ib]
        om2 = Mag.omega(k + dk*ebasis[alp]/2)[ib]
        v[alp] = (om2-om1)/dk
    v = v*Vel_Const
    return v

def AgroupV(Mag, k, ib, dk=1e-2):
    r"""
    calculates absolute value of group velocity  [m/s] at wavevector k and band ib
    uses rad - MagnonDispersion Mag
    dk - step for calculation of numerical derivative dE/dk
    """
    v = groupV(Mag, k, ib, dk=dk)
    return np.linalg.norm(v)


def groupV_P(pSH, k, ib, dk=1e-2):
    r"""
    calculates vector group velocity  [m/s] at wavevector k and band ib
    uses pseudorad Hamiltonian pSH
    dk - step for calculation of numerical derivative dE/dk
    """
    v = np.zeros(3)
    for alp in range(3):
        om1 = prad.omega(pSH, k - dk*ebasis[alp]/2)[0][ib]
        om2 = prad.omega(pSH, k + dk*ebasis[alp]/2)[0][ib]
        v[alp] = (om2-om1)/dk
    v = v*Vel_Const
    return v

def AgroupV_P(pSH, k, ib, dk=1e-2):
    r"""
    calculates absolute value of group velocity [m/s] at wavevector k and band ib
    uses rad - MagnonDispersion Mag
    dk - step for calculation of numerical derivative dE/dk
    """
    v = groupV_P(pSH, k, ib, dk=dk)
    return np.linalg.norm(v)

##-------------------------------------------------------------------------------------------------------

def Kpath(Points, Nst = 30):
    r"""
    generates Kpathe based on the set of Points
    """
    Npoi0 = len(Points)

    kpoi = [Points[0],]
    Xmarks = [0.,]
    xx = [0,]
    cnt = 0
    len1 = 0.0
    for i in range(Npoi0-1):
        for i1 in range(Nst):
            cnt += 1
            k1 = ((Nst-i1-1)/Nst) * Points[i] + ((i1+1)/Nst)* Points[i+1]
            dk = k1-kpoi[-1]
            adk = np.linalg.norm(dk)
            len1 += adk
            kpoi.append(k1)
            xx.append(len1)
        Xmarks.append(len1)
    return kpoi, xx, Xmarks



##-------------------------------------------------------------------------------------------------------

def normaldirections(v, dmin=0.01):
    r"""
    finds the directions perpendicular to a vector v
    """
    ex = np.array((1,0,0))
    ez = np.array((0,0,1))
    v1 = np.cross(v,ez)
    if np.linalg.norm(v1) < dmin:
        v1 = np.cross(v,ex)
    ev1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(v,ev1)
    ev2 = v2 / np.linalg.norm(v2)
    return ev1,ev2


##-------------------------------------------------------------------------------------------------------


def RemoveAnisotropy(SH):
    r"""
    removes all anisotropy from spin Hamiltonian
    """
    SH1 = copy.deepcopy(SH)
    
    not1 = SH1.notation
    SH1.notation = (False, False, -1)
    for at1,at2, evec, J in SH1:
        J1 = J.iso
        radJ1 = rad.ExchangeParameter(iso=J1)
        SH1[at1, at2, evec] = radJ1
    SH1.notations = not1
    return SH1






###### calculate the lists l_i1, l_i2, l_cvec from spin Hamiltonian and real positions 
##### the lists describe the possible bonds
###### with the maximum distance equal to disMax
def formLists(SH0, pos, disMax = 5):
    l_i1 = []
    l_i2 = []
    l_cvec = []
    numD = {}
    for iat, at in enumerate(SH0.magnetic_atoms):
        numD[at.name] = iat
    for at1,at2, cvec, Jint in SH0:
        i1 = numD[at1.name]
        i2 = numD[at2.name]
        vec = cvec[0]*SH0.a1 + cvec[1]*SH0.a2 + cvec[2]*SH0.a3
        vec += pos[i2]-pos[i1]
        dis = np.linalg.norm(vec)
        if dis < disMax:
            l_i1.append(i1)
            l_i2.append(i2)
            l_cvec.append(cvec)
    return l_i1, l_i2, l_cvec



def cloneSH(SH0, cell1):
    r"""
    clones spin Hamiltonian with new cell
    uses standardize=False to prevent spontaneous rotations
    """
    SH = rad.SpinHamiltonian(cell=cell1, standardize=False)
    SH.notation = SH0.notation

    for at in SH0.magnetic_atoms:
        at1 = copy.deepcopy(at)
        SH.add_atom(at1)

    for at1,at2,cvec, J in SH0:
        Jmat = J.matrix
        Jnew = rad.ExchangeParameter(matrix=Jmat)
        SH[at1.name, at2.name,cvec] = Jnew
    return SH
