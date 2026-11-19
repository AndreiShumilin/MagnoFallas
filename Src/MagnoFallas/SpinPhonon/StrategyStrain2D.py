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




r""" A strategy corresponding to strained 2D materials
"""


import numpy as np
import numpy.linalg
import scipy as sp
import numbers

import MagnoFallas.OldRadtools as rad
import MagnoFallas.Utils.util2 as ut2
from MagnoFallas.SpinPhonon import SPhUtil as sphut




def Estimate_from_2D_strain(SHlist, mag_correspond_list, disMax = 5, SH0=None, dJlst=None):
    ## main function of the module
    r"""
    Generates information on spin-displacement coupling presuming that a set of spin Hamiltonians in
    SHlist describes a 2D material with different strains in x- and y-direction
    Presumes that the exchange matrix between two magnetic atoms depends only on the relative displacement of these atoms
    in xy-plane

    SHlist: lest of spin Hamiltonians
    mag_correspond_list: correspondence between magnetic_atoms and atom enumeration in phonopy
    disMax: maximum distance to consider (in A)
    SH0 - spin Hamiltonian considered "not strained"
    dJlst - list where results are written (new one is created if "None")
    """
    
    if dJlst is None:
        dJlst = sphut.EmptydJList()

    Poslist = [ut2.GetRealPositions(SH) for SH in SHlist]
    
    if SH0 is None:
        SH0 = SHlist[0]
        pos0 = Poslist[0]
    else:
        pos0 = ut2.GetRealPositions(SH0)
    
    Itlist = formLists(SH0, pos0, disMax=disMax)

    for it in Itlist:
        strainXYfit_bond(it, SHlist, Poslist, mag_correspond_list, dJlst)

    return dJlst






def formLists(SH0, rpos, disMax = 5):
    r"""
    identifies a list of bonds to calculate based on 
    spin Hamiltonian SH0 
    and real positions rpos
    disMax - maximum distance (in A) to consider
    """
    list1 = []
    numD = {}
    for iat, at in enumerate(SH0.magnetic_atoms):
        numD[at.name] = iat
    for at1,at2, cvec, Jint in SH0:
        i1 = numD[at1.name]
        i2 = numD[at2.name]
        vec = cvec[0]*SH0.a1 + cvec[1]*SH0.a2 + cvec[2]*SH0.a3
        vec += rpos[i2]-rpos[i1]
        dis = np.linalg.norm(vec)
        if dis < disMax:
            it1 = (i1,i2,cvec)
            list1.append(it1)
    return list1





def strainXYfit_bond(item, SHlist, Poslist, mag_correspond_list, dJlst):
    r"""
    calculates dJ-lines for a single exchange bond
    
    item: i1, i2, cvec - defines the bond to to lineary fit
    SHlist:  list of Spin Hamiltonians
    Poslist: list of real positions
    mag_correspond_list: "correspondence list" for magnetic atoms
    
    dJlst: list of dJ-lines where the result is written
    """
    i1,i2,cvec = item
    n1 = mag_correspond_list[i1]
    n2 = mag_correspond_list[i2]
    Nham = len(SHlist)
    
    xx = np.zeros(Nham)
    yy = np.zeros(Nham)
    JJ = []
    for ih in range(Nham):
        pos = Poslist[ih]
        SH =  SHlist[ih]
        at1 = SH.magnetic_atoms[i1]
        at2 = SH.magnetic_atoms[i2]
        if not (at1,at2,cvec) in SH:
            print('Problem!!!, element not in SH num. ', ih, ' : ', i1,i2,cvec)
        Jm = SH[at1,at2,cvec].matrix
        vec = cvec[0]*SH.a1 + cvec[1]*SH.a2 + cvec[2]*SH.a3
        vec += pos[i2] - pos[i1]
        xx[ih] = vec[0]
        yy[ih] = vec[1]
        JJ.append(Jm)
    JJ = np.array(JJ)
    fitA = np.array([np.ones_like(xx), xx, yy])
    fitA = np.transpose(fitA)

    acvec = np.array(cvec, dtype=np.int32)
    zvec = np.zeros(3, dtype=np.int32)
    dJx = np.zeros((3,3), dtype=np.float64)
    dJy = np.zeros((3,3), dtype=np.float64)
    for ii1 in range(3):
        for ii2 in range(3):
            arJ = JJ[...,ii1,ii2]
            sqfit = np.linalg.lstsq(fitA, arJ, rcond=None)
            cX = sqfit[0][1]
            cY = sqfit[0][2]
            dJx[ii1,ii2] = cX
            dJy[ii1,ii2] = cY
    # print(dJx)
    # print(dJy)
            
    dJlineX1 = sphut.TderivJ(i1,i2, acvec, n2, ut2.ex.copy(), acvec, dJx)
    dJlineX2 = sphut.TderivJ(i1,i2, acvec, n1, ut2.ex.copy(), zvec, -dJx)
    
    dJlineY1 = sphut.TderivJ(i1,i2, acvec, n2, ut2.ey.copy(), acvec, dJy)
    dJlineY2 = sphut.TderivJ(i1,i2, acvec, n1, ut2.ey.copy(), zvec, -dJy)
    # return JJ

    dJlst.append(dJlineX1)
    dJlst.append(dJlineX2)
    dJlst.append(dJlineY1)
    dJlst.append(dJlineY2)