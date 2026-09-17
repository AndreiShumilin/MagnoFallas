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
colection of procedures to work with distributions and statistics
"""


import numpy as np
import scipy as sp
import numba as nb



__all__ = ['RecalculateContributions2D', 'RecalculateDependence2D']



def RecalculateContributions2D(Contr1, axis1, SH, Nnew, norm=True):
    r"""
    Transforms 2D contributions (Contr1) to some quantity calculated in the "relative grid" defined in fractions of b1, b2
    into the real coordinates of A^{-1}
    requires old axes (axis1) corresponding to Contr1 (which should be in relative coordinates)
    and spin Hamiltonian (SH) to know the reciprocal vectors
    Nnew - the size of the new grid
    norm - True means that the contributions will be normalized.
    """
    b1 = SH.b1
    b2 = SH.b2
    Nx = len(axis1[0])
    Ny = len(axis1[1])
    corners = np.array([np.min(axis1[0])*b1 + np.min(axis1[1])*b2,
                        np.min(axis1[0])*b1 + np.max(axis1[1])*b2,
                        np.max(axis1[0])*b1 + np.min(axis1[1])*b2,
                        np.max(axis1[0])*b1 + np.max(axis1[1])*b2,
                       ])
    #print(corners)
    xm, xM = np.min(corners[...,0]), np.max(corners[...,0])
    ym, yM = np.min(corners[...,1]), np.max(corners[...,1])

    cellsize = np.linalg.norm(np.cross(b1, b2))
    cellsize *= (xM-xm)
    cellsize *= (yM-ym)
    cellsize /= Nx*Ny

    newCont = np.zeros((Nnew, Nnew))
    newContCou = np.zeros((Nnew, Nnew))
    dx = (xM-xm)/Nnew
    dy = (yM-ym)/Nnew
    
    newcellsize = dx*dy
    if norm:
        Ncontr = np.sum(Contr1)
    else:
        Ncont = 1.0
    
    newX = np.linspace(xm, xM, Nnew)
    newY = np.linspace(ym, yM, Nnew)
    newaxis = np.array((newX, newY))
    for ix in range(Nx):
        for iy in range(Ny):
            ix2 = int((axis1[0][ix] - xm)//dx)
            iy2 = int((axis1[1][iy] - ym)//dy)

            if (ix2>=0) and (ix2<Nnew) and (iy2>=0) and (iy2<Nnew):
                newCont[ix2,iy2] += Contr1[ix,iy]/(newcellsize * Ncontr  )
                newContCou[ix2,iy2] +=1

    ### we try to correct artefacts related to the inconsistency between "relative" and "real" k-grids
    ### we take into account that each point of real grid must have the same numbers of k1-vectors
    ### but in reality they dont
    AvCou = 0
    goodCells = 0
    for ix2 in range(Nnew):
        for iy2 in range(Nnew):
            if newContCou[ix2,iy2] > 0:
                goodCells += 1
                AvCou += newContCou[ix2,iy2]
    AvCou /= goodCells
    for ix2 in range(Nnew):
        for iy2 in range(Nnew):
            if newContCou[ix2,iy2] > 0:
                newCont[ix2,iy2] *= (AvCou/newContCou[ix2,iy2])
    
    return newaxis,newCont


def RecalculateDependence2D(Depend1, axis1, SH, Nnew):
    r"""
    Transforms 2D k-vector dependence (Depend1) calculated in the "relative coordinates", i.e. fractions of reciprocal vectors b1,b2
    into "real" coordinates in A^{-1}
    axis1 - axes (in relative coordinates) corresponding to Depend1
    SH - spin Hamiltonian (required only to know the reciprocal vectors)
    Nnex - size of the new k-grid in real coordinates
    """
    b1 = SH.b1
    b2 = SH.b2
    Nx = len(axis1[0])
    Ny = len(axis1[1])
    corners = np.array([np.min(axis1[0])*b1 + np.min(axis1[1])*b2,
                        np.min(axis1[0])*b1 + np.max(axis1[1])*b2,
                        np.max(axis1[0])*b1 + np.min(axis1[1])*b2,
                        np.max(axis1[0])*b1 + np.max(axis1[1])*b2,
                       ])
    #print(corners)
    xm, xM = np.min(corners[...,0]), np.max(corners[...,0])
    ym, yM = np.min(corners[...,1]), np.max(corners[...,1])

    cellsize = np.linalg.norm(np.cross(b1, b2))
    cellsize *= (xM-xm)
    cellsize *= (yM-ym)
    cellsize /= Nx*Ny

    newCont = np.zeros((Nnew, Nnew))
    newContCou = np.zeros((Nnew, Nnew))
    dx = (xM-xm)/Nnew
    dy = (yM-ym)/Nnew
    newX = np.linspace(xm, xM, Nnew)
    newY = np.linspace(ym, yM, Nnew)
    newaxis = np.array((newX, newY))
    for ix in range(Nx):
        for iy in range(Ny):
            ix2 = int((axis1[0][ix] - xm)//dx)
            iy2 = int((axis1[1][iy] - ym)//dy)

            if (ix2>=0) and (ix2<Nnew) and (iy2>=0) and (iy2<Nnew):
                newCont[ix2,iy2] += Depend1[ix,iy]
                newContCou[ix2,iy2] +=1

    for ix2 in range(Nnew):
        for iy2 in range(Nnew):
            if newContCou[ix2,iy2] > 0:
                newCont[ix2,iy2] /= newContCou[ix2,iy2]
    
    
    return newaxis,newCont

    