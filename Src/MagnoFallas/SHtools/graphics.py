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
import numba as nb
import matplotlib.pyplot as plt

import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap

import MagnoFallas.OldRadtools as rad
from MagnoFallas.SHtools import tools as tools
from MagnoFallas.Utils import util2 as ut2
from MagnoFallas.Interface import PseudoRad as prad



#### here are the two custom color maps: blue and red
#### the reason for both is to make zero-values transparent
cmap = pl.cm.Reds
R_cmap = cmap(np.arange(cmap.N))
R_cmap[:,-1] = np.linspace(0, 1, cmap.N)
R_cmap = ListedColormap(R_cmap)

cmap = pl.cm.Blues
B_cmap = cmap(np.arange(cmap.N))
B_cmap[:,-1] = np.linspace(0, 1, cmap.N)
B_cmap = ListedColormap(B_cmap)



@nb.njit
def TestCenter(k, b1, b2, Nli=3):
    r'''
    Checks if a point lies in 1-st Brillouin zone
    '''
    ak0 = np.linalg.norm(k)
    L1 = np.arange(-Nli,Nli+1)
    res = True
    for i in L1:
        for j in L1:
            if not((i==0) and (j==0)):
                k1 = k + i*b1 + j*b2
                ak1 = np.linalg.norm(k1)
                if ak1<ak0:
                    res = False
                    return res
    return res


def BrilMap(foo, SH, Np=100, Nli=3):
    r'''
    The tool to calculate a map of quantity foo inside the first Brillouin zone of 2D magnetic material
    foo - function of a wavevector (wavevector is a 1D array with size 3)
    SH - spin Hamiltonian
    Np - number of points per axis 
    Nli - maximum shift per axis (I am not sure if values >1 would be ever needed)
    points outside the 1-st brilluin zone would be included (the map is rectangular) but will have zero values
    '''
    ez1 = np.cross(SH.a1, SH.a2)
    ez1 /= np.linalg.norm(ez1)
    b1 = np.cross(ez1, SH.a2)
    b1 *= 2*np.pi/(b1@SH.a1)
    b2 = np.cross(ez1, SH.a1)
    b2 *= 2*np.pi/(b2@SH.a2)

    ex1 = ut2.ex - ez1*(ut2.ex@ez1)
    ex1 /= np.linalg.norm(ex1)
    ey1 = ut2.ey - ez1*(ut2.ey@ez1)
    ey1 /= np.linalg.norm(ey1)

    L1 = np.arange(-Nli,Nli+1)
    xx = np.array([i*b1@ex1/2  + j*b2@ex1/2 for i in L1 for j in L1])
    yy = np.array([i*b1@ey1/2  + j*b2@ey1/2 for i in L1 for j in L1])
    KXmin = np.min(xx)
    KXmax = np.max(xx)
    KYmin = np.min(yy)
    KYmax = np.max(yy)

    akx = np.linspace(KXmin, KXmax, Np)
    aky = np.linspace(KYmin, KYmax, Np)
    res = np.zeros((Np,Np))
    for i in range(Np):
        for j in range(Np):
            k = np.array((akx[i], aky[j],0))
            if TestCenter(k, b1, b2, Nli=Nli):
                f = foo(k)
                res[i,j] = f
    return res, akx, aky


def DrawMapSym(akx, aky, res, cmap='seismic'):
    r'''
    The tool helping to draw of a symmetric quantity (both negative and positive values are possible)
    '''
    res2 = np.transpose(res)
    Vm = np.max(np.abs(res2))
    plt.pcolormesh(akx,aky, res2, vmin=-Vm, vmax=Vm, cmap=cmap)
    ixmin = np.min(np.nonzero(res2)[1])
    ixmax = np.max(np.nonzero(res2)[1])
    iymin = np.min(np.nonzero(res2)[0])
    iymax = np.max(np.nonzero(res2)[0])
    plt.xlim(akx[ixmin], akx[ixmax])
    plt.ylim(aky[iymin], aky[iymax])


def DrawMapPos(akx, aky, res, cmap=B_cmap):
    r'''
    The tool helping to draw of a positive quantity
    '''
    res2 = np.transpose(res)
    Vm = np.max(np.abs(res2))
    plt.pcolormesh(akx,aky, res2, vmin=0, vmax=Vm, cmap=cmap)
    ixmin = np.min(np.nonzero(res2)[1])
    ixmax = np.max(np.nonzero(res2)[1])
    iymin = np.min(np.nonzero(res2)[0])
    iymax = np.max(np.nonzero(res2)[0])
    plt.xlim(akx[ixmin], akx[ixmax])
    plt.ylim(aky[iymin], aky[iymax])


def drawHam(SH, dim=2, realpos = True, cmapP = pl.cm.Blues, cmapN = pl.cm.Reds, cmapA = pl.cm.bwr, file=None,
            FullFig=True, ax0=None, minJ = 0, removeAxis=True):
    r'''
    The tool to graphically show a spin Hamiltonian
    #### SH - spin Hamiltonian
    #### dim - dimension, 2 or 3
    #### realpos - wether the atom position are real (in Angstr) of relative (fractions of cell vectors)
    #### cmapP, cmapN - colormaps for positive and negative bonds respectingly
    #### cmapA - colormap for magnetic atoms
    #### file - (string) if set, saces the figure to this file
    #### fullFigure - true if we just need a figure, False is part of larger "plot" code
    ####   (in the later case for a 3D Hamiltonian a valid axis ax0 should be provided)
    #### minJ - minimal value of the exchange interation to show
    ###  removeAxis - if True, remove the axes from the plot
    '''
    D=dim
    
    ### some work to make the code work for both relative and absolute positions
    Nat = len(SH.magnetic_atoms)
    Svec0 = SH.magnetic_atoms[0].spin_vector
    Aspins = np.array( [at.spin for at in SH.magnetic_atoms])
    MaxSpin = np.max(Aspins)
    
    if not realpos:
        pos = np.array([at.position for at in SH.magnetic_atoms])
    else:
        pos = np.array([ut2.realposition(SH, at) for at in SH.magnetic_atoms])

    xs = [pos1[0] for pos1 in pos]
    ys = [pos1[1] for pos1 in pos] 
    zs = [pos1[2] for pos1 in pos] 

    Js = np.array([J.iso for at1,at2,v,J in SH])
    Jmax = np.max(np.abs(Js))   ### we need maximum interaction to normalize the colors

    ## creating the figure
    if FullFig:
        if D==2:
            Fig = plt.figure(figsize=(4,4))
        elif D==3:
            Fig = plt.figure(figsize=(6,5))
            ax = Fig.add_subplot(111, projection='3d')
    else:
        ax = ax0

    ### part to draw a single lne
    ### code is long because of the choise 2D/3D
    ### and absolute/relative positions
    def line(at1,at2,val):
        if not realpos:
            rp1 = at1.position
            rp2 = at2.position
        else:
            rp1 = ut2.realposition(SH, at1)
            rp2 = ut2.realposition(SH, at2)
        xx = [rp1[0],rp2[0]]
        yy = [rp1[1],rp2[1]]
        zz = [rp1[2],rp2[2]]
        if val>0:
            col = cmapP(val/Jmax)
        else:
            col = cmapN(-val/Jmax)
            
        if D==2:
            plt.plot(xx,yy,color=col)
        elif D==3:
            ax.plot(xx,yy,zz, color=col)

    ### we draw lines for each bond
    for at1,at2,v,J in SH:
        if v==(0,0,0):
            val = J.iso
            if np.abs(val) > minJ:
                line(at1,at2,val)

    ### we put a marker for each atom 
    for iat,at in enumerate(SH.magnetic_atoms):
        x,y,z = xs[iat], ys[iat], zs[iat]
        xx = [x,]
        yy = [y,]
        zz = [z,]
        Svec1 = at.spin_vector
        ss = Svec0@Svec1 / (np.linalg.norm(Svec0) * np.linalg.norm(Svec1))
        ss *= np.abs(at.spin/MaxSpin)
        ss = (ss+1)/2
        col1 = cmapA(ss)
        if D==2:
            plt.scatter(xx,yy, marker='o', s=100, c=(col1,), edgecolor='k')
        elif D==3:
            ax.scatter(xx,yy, zz, marker='o', s=100, c=(col1,), edgecolor='k')

    if removeAxis:
        if D==3:
            ax.set_axis_off()
            # ax.grid(False)
            # ax.set_xticks([])
            # ax.set_yticks([])
            # ax.set_zticks([])
        elif D==2:
            plt.axis('off')

    if not file is None:
        plt.savefig(file, bbox_inches='tight')

    if FullFig:
        plt.show()




    
