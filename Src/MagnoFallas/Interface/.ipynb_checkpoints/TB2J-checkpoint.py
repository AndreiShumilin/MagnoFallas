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
import os

import MagnoFallas.OldRadtools as rad



def ReadTB2J(file, spins):
    r"""
    read TB2J results, assigns spins to magnetic atoms (the spins must be provided by user)
    file - name of TB2J  *.out file
    spins - number, array of numbers or array of vectors; the length of the vector must match the number of magnetic atoms

    The spins must be integer or half-integer numbers, or vectors with the corresponding absolute value (directed along the equilibrium magnetization
    of the given atom). Typically the values can be understood from magnetic moments shown in exchange.out file of TB2J, however, exceptions are possible. 
    
    Note: when reading a Hamiltonian of antiferromagnetic or ferrimagnetic material, the array of vectors is the only possible option
    """
    if not os.path.exists(file):
        print('ReadTB2J Error: file does not exist')
        return None
    
    SH = rad.load_tb2j_model(file, quiet=True, standardize=False)
    Nat = len(SH.magnetic_atoms)
    Aspins = np.array(spins)
    shp = Aspins.shape

    if len(shp) == 0:
        sp = Aspins
        for at in SH.magnetic_atoms:
            at.spin = sp
    elif len(shp) == 1:
        for iat,at in enumerate(SH.magnetic_atoms):
            sp = Aspins[iat]
            at.spin_vector = sp*np.array((0,0,1))
    elif len(shp) == 2:
        for iat,at in enumerate(SH.magnetic_atoms):
            at.spin_vector = Aspins[iat]

    return SH