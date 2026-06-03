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
Not perfectly tested - interface with MuMax3
"""



#########################

import numpy as np
from MagnoFallas.Utils import util2 as ut2

####============================

const_Aex_to_Si = ut2.mev_to_J * 1e10
const_DMI_to_Si = ut2.mev_to_J * 1e20
const_Ani_to_Si = ut2.mev_to_J * 1e30

const_muB_Si = 9.2740100657e-24
const_mu0_Si = 1.25663706127e-6

const_hbar_Si = 1.054571817e-34
const_hbar_mev = const_hbar_Si/ut2.mev_to_J

AexTensorCutoff0 = 1e-5   ### in meV/A
AexTensorCutoffSi = AexTensorCutoff0*const_Aex_to_Si

AniCutoff0 = 1e-5
AniCutoffSi = AniCutoff0 * const_Ani_to_Si

DMICutoff0 = 1e-5
DMICutoffSi = AniCutoff0 * const_DMI_to_Si

__all__  = ['const_Aex_to_Si', 'const_DMI_to_Si', 'const_Ani_to_Si', 
           'const_muB_Si', 'const_mu0_Si', 'const_hbar_Si', 'const_hbar_mev', 
           'AexTensorCutoff0', 'AexTensorCutoffSi', 
           'AniCutoff0', 'AniCutoffSi',
           'DMICutoff0','DMICutoffSi']