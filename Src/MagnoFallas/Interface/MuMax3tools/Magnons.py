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
import matplotlib.pyplot as plt
import copy


import MagnoFallas as mfal
from MagnoFallas.Utils import util2 as ut2
from MagnoFallas.Utils import MicroMagnetics as micro

# from ..MuMax3 import Material2D, DMILines, SquareGeometry, Header
# from ..MuMax3 import AexTensorCutoff0, AniCutoff0, DMICutoff0
# from ..MuMax3 import AexTensorCutoffSi, AniCutoffSi, DMICutoffSi
# from ..MuMax3 import const_Aex_to_Si, const_DMI_to_Si, const_Ani_to_Si 
# from ..MuMax3 import const_muB_Si, const_mu0_Si, const_hbar_Si,  const_hbar_mev
from .constants import *
from .Material import *
from .MiscParts import *

__all__ = ['simuMagnons_2D', 'testProgMagnons']

####=============================================

def simuMagnons_2D(Nx, Ny, Bext=None, GilbertAlpha=0.001, tsim=0.5e-9, tsave=1e-12, amp=0.05, tsaveM = 0.25e-10,
                  fixdt=None):
    r"""
    Lines for the simulation of a sinlge skyrmion
    """
    Lines = []
    Lines.append('/// ---->  Simulation Block <------------: \n')
    Lines.append(f'/// single magnon mode: {Nx} x {Ny}\n')
    if not(Bext is None):
        l1 = len(np.asarray(Bext).shape)
        if l1 == 0:
            Lines.append(f'B_ext = vector(0, 0, {Bext})  \n')
        elif l1 == 1:
            Lines.append(f'B_ext = vector({Bext[0]}, {Bext[1]}, {Bext[2]} )  \n')
    Lines.append(f'alpha   = {GilbertAlpha} \n\n')

    Lines.append('NoDemagSpins=1 \n')
    Lines.append('SetPBC(1, 1, 1) \n\n')
    if not fixdt is None:
        Lines.append(f'fixdt = {fixdt}\n')
        

    Lines.append(f'kx := 2*Pi*{Nx} / (cx*gridNx) \n')
    Lines.append(f'ky := 2*Pi*{Ny} / (cy*gridNy) \n')
    Lines.append('print("kx=", kx, "  ky=", ky ) \n\n')

    Lines.append(f'amp := {amp} \n')
    Lines.append(f'amp2 := {np.sqrt(1-amp*amp)} \n\n')

    Lines.append('/// setting initial magnetization \n')
    Lines.append('for i:=0; i<gridNx; i++{  \n')
    Lines.append( '    for j:=0; j<gridNy; j++{  \n')
    Lines.append(f'           phi := 2*Pi*{Nx}*i/gridNx +  2*Pi*{Ny}*j/gridNy  \n')
    Lines.append( '           mVec := vector(amp*cos(phi), amp*sin(phi), amp2)\n')
    Lines.append( '           m.SetCell(i,j,0, mVec)\n')
    Lines.append( '    }\n')
    Lines.append('} \n\n')

    Lines.append('tableAdd(Crop(m, 0, 1, 0, 1, 0, 1))\n')
    Lines.append(f'tableautosave({tsave})\n')
    Lines.append(f'autosave(m, {tsaveM})\n')
    Lines.append(f'run({tsim})\n\n')

    Lines.append('/// ---->  end of the simulation block <------------: \n')
    Lines.append('\n\n\n')
    
    return Lines  




def testProgMagnons(SH, Nx, Ny, fileName='magnon.txt', 
               Lx=1000, Ly=1000, gridNx=100, gridNy=100, cz_nm = 1,
                 AexTensorCutoffSi=AexTensorCutoffSi, AniCutoffSi=AniCutoffSi, DMICutoffSi=DMICutoffSi,
                 Bext=None, GilbertAlpha=1e-3, tsim=3e-10,
                   tsave=1e-12, amp=0.05, tsaveM = 1e-11, fixdt=None):
    r"""
    Generates a MuMax3 code to test Nx x Ny magnon mode
    """

    L1 = Header()
    L2 = SquareGeometry(Lx=Lx, Ly=Ly, Nx=gridNx, Ny=gridNy, cz_nm = cz_nm)
    L3 = Material2D(SH, cz_nm = cz_nm, AexTensorCutoffSi=AexTensorCutoffSi, AniCutoffSi=AniCutoffSi, DMICutoffSi=DMICutoffSi)
    L4 = simuMagnons_2D(Nx, Ny, Bext=Bext, GilbertAlpha=GilbertAlpha, tsim=tsim, tsave=tsave, amp=amp, tsaveM = tsaveM, fixdt=fixdt)

    Lines = L1 + L2 + L3 + L4

    with open(fileName, "w") as f:
        for ln in Lines:
            f.write(ln)