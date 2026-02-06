## last modified 01.07.2025
### A strategy that assumes that each type of exchange interactions depends only on the distance between corresponding atoms

import numpy as np
import numpy.linalg
import scipy as sp
import numbers

import numba as nb
from numba.experimental import jitclass

import radtools as rad

from SpinPhonon import SPhUtil as sphut



#### base function: determine the exchange interaction based on spin hamiltonian and distances
#### and than calculate the dJ/dr lines of spin-displacement Hamiltonian
def Estimate_auto_dist(SH, pos, dir0, dirlist, mag_correspond_list, fil = 'exchange.out', lst=None, sp0=3/2, PCCmin=0.45, disMax = 5):
    l_i1, l_i2, l_cvec = formLists(SH, pos, disMax = disMax)
    HlineLst = Estimate_bonds_dist(l_i1, l_i2, l_cvec, dir0, dirlist, mag_correspond_list, fil = fil, lst=lst, sp0=sp0, PCCmin=PCCmin)
    return HlineLst


###### calculate the lists l_i1, l_i2, l_cvec from spin Hamiltonian and real positions 
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
        vec = cvec[0]*SH.a1 + cvec[1]*SH.a2 + cvec[2]*SH.a3
        vec += pos[i2]-pos[i1]
        dis = np.linalg.norm(vec)
        if dis < disMax:
            l_i1.append(i1)
            l_i2.append(i2)
            l_cvec.append(cvec)
    return l_i1, l_i2, l_cvec


###### Gets list of dJ/dr based on the lists "l_i1, l_i2, l_cvecJ" describing the desired exchange interactions
def Estimate_bonds_dist(l_i1, l_i2, l_cvecJ, dir0, dirlist, mag_correspond_list, fil = 'exchange.out', lst=None, sp0=3/2, PCCmin=0.45):
    ###    l_i1, l_i2, l_cvecJ : lists describing bonds to study i1,i2 - numbers in magnetic_atoms, cvec - distance in unit cells
    ###    dir0 - directory of the unperturbed Hamiltonian
    ###    dirlist - listy of folders with results to be used
    ###    mag_correspond_lis - correspondence list between magnetic_atoms and results of phonopy
    ###    fil - name of output files of TB2J (should always be the same)
    ###    lst - list of already calculated terms (usually should be left "None")
    ###    sp0 - spins of magnetic atoms (single value or list)
    ###    PCCmin - minimum value of the correlation criterium  (between 0 and 1) to accept the estimated linear dependence
    
    if lst==None:
        lst = sphut.EmptydJList()
    SH0, pos0 = sphut.SHread(dir0 + fil, sp0=sp0)

    numD = {}
    for iat, at in enumerate(SH0.magnetic_atoms):
        numD[at.name] = iat
    

    #preliminary calculations with zero-Hamiltonian
    Ncalc = len(l_i1)
    Ndir = len(dirlist)
    mat0 = SH0.magnetic_atoms
    J0matr = [SH0[mat0[l_i1[i]],mat0[l_i2[i]], l_cvecJ[i]].matrix  for i in range(Ncalc)   ]
    evec0 = np.zeros((Ncalc,3))
    vecs0 = np.zeros((Ncalc,3))
    for i in range(Ncalc):
        po0 = pos0[l_i1[i]]
        po1 = pos0[l_i2[i]]
        cvec = l_cvecJ[i]
        vec = cvec[0]*SH0.a1 + cvec[1]*SH0.a2 + cvec[2]*SH0.a3
        vec += po1-po0
        vecs0[i] = vec
        vecN = vec / np.linalg.norm(vec)
        evec0[i] = vecN

    dJmatr = [[] for i in range(Ncalc)]
    d_dis = np.zeros((Ncalc,Ndir))

    #### collect the data from files
    for id,d in enumerate(dirlist):
        SH, pos = sphut.SHread(d + fil, sp0=sp0)
        mat = SH.magnetic_atoms
        for i in range(Ncalc):
            Jm1 = SH[mat[l_i1[i]],mat[l_i2[i]], l_cvecJ[i]].matrix
            dJm = Jm1 - J0matr[i]
            dJmatr[i].append(dJm)
            
            po0 = pos[l_i1[i]]
            po1 = pos[l_i2[i]]
            cvec = l_cvecJ[i]
            vec = cvec[0]*SH.a1 + cvec[1]*SH.a2 + cvec[2]*SH.a3
            vec += po1-po0
            vec -= vecs0[i]
            ddist = vec@evec0[i]
            d_dis[i,id] = ddist
            
    #linear regression
    JlinM = np.zeros((Ncalc, 3, 3 ))
    dJmatr = np.array(dJmatr)
    for ica in range(Ncalc):
        arX = d_dis[ica]
        arY0 = dJmatr[ica]
        for i1 in range(3):
            for i2 in range(3):
                arY = arY0[...,i1,i2]
                regres = sp.stats.linregress(arX, arY)
                pcc = regres.rvalue
                if np.abs(pcc) > PCCmin:
                    JlinM[ica,i1,i2] = regres.slope    

    cvec0 = np.array((0,0,0), dtype = np.int32)
    for ica in range(Ncalc):
        i1 = l_i1[ica]
        i2 = l_i2[ica]
        n1 = mag_correspond_list[i1]
        n2 = mag_correspond_list[i2]
        J1 = JlinM[ica].copy()
        J1 = J1.astype(complex)
        cvec = l_cvecJ[ica]
        #print(J1.shape)
        #print(i1,i2, np.array(cvec), n2, evec0[ica].copy(), np.array(cvec), J1)
        Xlist.append( (i1,i2, np.array(cvec), n2, evec0[ica].copy(), np.array(cvec), J1)  )
        dJline1 = sphut.TderivJ(i1,i2, np.array(cvec), n2, evec0[ica].copy(), np.array(cvec), J1)
        dJline2 = sphut.TderivJ(i1,i2, np.array(cvec), n1, evec0[ica].copy(), np.array(cvec0), -J1)
        lst.append(dJline1)
        lst.append(dJline2)
        
    return lst

