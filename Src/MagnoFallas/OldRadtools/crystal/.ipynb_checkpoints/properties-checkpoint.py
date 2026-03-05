########################################################################################
#  The following part of the code was a version of Rad-Tools project by Andrey Rybakov
# incorporated into MagnoFallas. The license of the original code is provided below
########################################################################################


# RAD-tools - Sandbox (mainly condense matter plotting).
# Copyright (C) 2022-2024  Andrey Rybakov
#
# e-mail: anry@uv.es, web: rad-tools.org
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

from math import pi

import numpy as np
from tqdm import tqdm

__all__ = ["dipole_dipole_energy", "dipole_dipole_interaction"]

CONSTANT = 1.25663706212 * 9.2740100783**2 * 6.241509074 / 1000 / 4 / pi


