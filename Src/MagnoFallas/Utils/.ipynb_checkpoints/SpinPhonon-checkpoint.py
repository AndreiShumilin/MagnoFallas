###### out-dated file
###### to be removed from the project
###### spin-phonon interactions should be in a separate folder.


import numpy as np
import numpy.linalg
import scipy as sp
import numbers

import numba as nb
from numba.experimental import jitclass

import radtools as rad

__all__ = ['TderivJ','EmptydJList','realposition','SHread','linefit','Estimate0','Tline2M1Ph','Empty2M1PhLineList','dJ_to_Lines_2M1Ph','Full_dHS_toMagn']

### phonon-x-constnt = 10^10 [\hbar^2 / 2 * amu  * 1meV ]^{1/2}    (result is in angstr)
##phonon_x_constant = 1.4457107723636267

ex = np.array((1.,0,0))
ey = np.array((0,1.,0))
ez = np.array((0,0,1.))





ModificationJlist =  [
    ('i1', nb.int32),  ##### participating spins (numbers inside unit cell)
    ('i2', nb.int32),  
    ('cvecJ', nb.int32[::1]),  ### vector between unit cells of participating atoms
    ('n', nb.int32),     #### displaced atom (number in unit cell of phonon calculation)
    ('edis', nb.float64[::1]),  ###unit vector of the displacement
    ('cvecN', nb.int32[::1]),  
    ('dJ', nb.complex128[:,::1])  ### derivation of the exchange interaction matrix
]


############# class describing the modification of exchange interaction in spin notations ######################

@jitclass(ModificationJlist)
class TderivJ(object):
    def __init__(self, i1, i2, cvecJ, n, edis, cvecN, dJ):
        self.i1 = i1
        self.i2 = i2
        self.cvecJ = cvecJ
        self.n = n
        self.edis = edis
        self.cvecN = cvecN
        self.dJ = dJ


def EmptydJList():
    i1 = 0 ; i2 = 0
    cvec = np.array((0,0,0), dtype = np.int32)
    n= 0 
    dJ = np.eye(3, dtype=np.complex128)
    test0 = TderivJ(i1,i2,cvec,n,ex,cvec,dJ)
    lis = nb.typed.List()
    lis.append(test0)
    lis.clear()
    return lis


def realposition(SH, at):
    vec = np.zeros(3)
    vec += at.position[0]*SH.a1
    vec += at.position[1]*SH.a2
    vec += at.position[2]*SH.a3
    return vec

def SHread(str, sp0=3/2):
    SH = rad.load_tb2j_model(str, quiet=True, standardize=False)
    realpos = []

    is1S = isinstance(sp0, numbers.Number)
    
    for i,at in enumerate(SH.magnetic_atoms):
        if is1S:
            SH.magnetic_atoms[i].spin = sp0
        else:
            SH.magnetic_atoms[i].spin = sp0[i]
        realpos.append( realposition(SH, at)  )
    realpos = np.array( realpos )
    SH.notation = (False,False,-1)
    return SH, realpos

### a version which removes all anisotropy
def SHread_iso(str, sp0=3/2):
    realpos = []
    SH = rad.load_tb2j_model(str, quiet=True)
    is1S = isinstance(sp0, numbers.Number)
    for i,at in enumerate(SH.magnetic_atoms):
        if is1S:
            SH.magnetic_atoms[i].spin = sp0
        else:
            SH.magnetic_atoms[i].spin = sp0[i]
        realpos.append( realposition(SH, at)  )
            
    for at1, at2, v, J in SH:
        iso = J.iso
        Jiso = rad.ExchangeParameter(iso=iso)
        SH[at1,at2,v] = Jiso
    SH.notation = (False,False,-1)
    realpos = np.array( realpos )
    return SH, realpos




def linefit(dx,dA):   ####to be improved
    we = dx*dx*dx*dx
    R = np.sum(we*(dA/dx))/np.sum(we)
    return R


### n enumerated the displaced atom
def Estimate0(n, edis, dir0, dirlist, fil = 'exchange.out', lst=None):
    if lst==None:
        lst = EmptydJList()
    SH0, pos0 = SHread(dir0 + fil)

    numD = {}
    for iat, at in enumerate(SH0.magnetic_atoms):
        numD[at.name] = iat
    
    xL = []
    shL = []
    for d in dirlist:
        SH, pos = SHread(d + fil)
        dx = (pos[n]-pos0[n])@edis
        xL.append(dx)
        shL.append(SH)
    xL = np.array(xL)
    
    for bnd in SH0:
        At1 = bnd[0]
        At2 = bnd[1]
        vecJ = bnd[2]
        i1 = numD[At1.name]
        i2 = numD[At2.name]
        vecJv = np.array(vecJ, dtype = np.int32)
        Jmat0 = bnd[3].matrix

        dJmat = np.zeros((3,3),dtype=np.complex128)
        for j1 in range(3):
            for j2 in range(3):
                a = []
                for SH in shL:
                    J1 = SH[At1.name,At2.name,vecJ].matrix
                    dj = J1[j1,j2] - Jmat0[j1,j2]
                    a.append(dj)
                a = np.array(a)
                dJmat[j1,j2] = linefit(xL,a)
        ###
        vecN = np.array((0,0,0), dtype = np.int32)
        dJline = TderivJ(i1,i2, vecJv, n, edis, vecN, dJmat)
        lst.append(dJline)
    
    return lst
#########################################################################################




############# class describing lines in magnon Hamiltonian, phonons are still displacements ######################

Tline2M1Ph_lst = [       ### 2 magnon 1 phonon
    ('A', nb.complex128),
    ('r', nb.float64[:,::1]),
    ('nu', nb.int32[::1]),  
    ('jdis', nb.int32),       ### number of displaced atom
    ('Rdis', nb.float64[::1]),   #### vector to UC of displaced atom from UC of i1  (in real coordinated [A])
    ('edis', nb.float64[::1])    ### 0-1-2 = x-y-z direction
]











