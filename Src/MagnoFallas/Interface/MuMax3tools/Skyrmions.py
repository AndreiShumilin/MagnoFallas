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

#from ..MuMax3 import Material2D, DMILines, SquareGeometry, Header

from .constants import *
from .Material import *

# from ..MuMax3 import AexTensorCutoff0, AniCutoff0, DMICutoff0
# from ..MuMax3 import AexTensorCutoffSi, AniCutoffSi, DMICutoffSi
# from ..MuMax3 import const_Aex_to_Si, const_DMI_to_Si, const_Ani_to_Si 
# from ..MuMax3 import const_muB_Si, const_mu0_Si, const_hbar_Si,  const_hbar_mev

__all__ = ['simuSkyrmion_2D', 'prog_Skyrmion']

####=============================================

def simuSkyrmion_2D(Bext=None, GilbertAlpha=0.05, tsim=0.5e-9):
    r"""
    Lines for the simulation of a sinlge skyrmion
    """
    Lines = []
    Lines.append('/// ---->  Simulation Block <------------: \n')
    Lines.append('/// relaxation of a single skyrmion \n')
    if not(Bext is None):
        l1 = len(np.asarray(Bext).shape)
        if l1 == 0:
            Lines.append(f'B_ext = vector(0, 0, {Bext})  \n')
        elif l1 == 1:
            Lines.append(f'B_ext = vector({Bext[0]}, {Bext[1]}, {Bext[2]} )  \n')
    Lines.append(f'alpha   = {GilbertAlpha} \n')

    Lines.append('m = BlochSkyrmion(1, -1).scale(1,1,1) \n')
    Lines.append('saveas(m, "Initial_state") \n')
    Lines.append('autosave(m, 50e-12)  \n')
    Lines.append(f'Run({tsim:.5e}) \n')
    Lines.append('\n')
    Lines.append('relax() \n\n')
    Lines.append('saveas(m, "Fianl_state") \n')
    Lines.append('/// ---->  end of the simulation block <------------: \n')
    Lines.append('\n\n\n')
    
    return Lines  


def prog_Skyrmion(SH, fileName='skyrmion.txt', 
               Lx=1000, Ly=1000, Nx=100, Ny=100, cz_nm = 1,
              AexTensorCutoffSi=AexTensorCutoffSi, AniCutoffSi=AniCutoffSi, DMICutoffSi=DMICutoffSi,
              Bext=None, GilbertAlpha=0.05, tsim=0.5e-9):
    r"""
    Generates a whole programm for skyrmion simulation
    """

    L1 = Header()
    L2 = SquareGeometry(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, cz_nm = cz_nm)
    L3 = Material2D(SH, cz_nm = cz_nm, AexTensorCutoffSi=AexTensorCutoffSi, AniCutoffSi=AniCutoffSi, DMICutoffSi=DMICutoffSi)
    L4 = simuSkyrmion_2D(Bext=Bext, GilbertAlpha=GilbertAlpha, tsim=tsim)
    Lines = L1 + L2 + L3 + L4

    with open(fileName, "w") as f:
        for ln in Lines:
            f.write(ln)
