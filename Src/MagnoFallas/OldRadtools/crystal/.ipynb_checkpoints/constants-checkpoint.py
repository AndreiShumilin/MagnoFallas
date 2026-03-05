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

__all__ = [
    "ABS_TOL",
    "REL_TOL",
    "MIN_LENGTH",
    "MAX_LENGTH",
    "ABS_TOL_ANGLE",
    "REL_TOL_ANGLE",
    "MIN_ANGLE",
    "ATOM_TYPES",

]

# Length variables
ABS_TOL = 1e-8  # Meant for the linear spatial variables
REL_TOL = 1e-4  # Meant for the linear spatial variables
# MIN_LENGTH is a direct consequence of the REL_TOL and ABS_TOL:
# for l = MIN_LENGTH => ABS_TOL = l * REL_TOL
MIN_LENGTH = ABS_TOL / REL_TOL
# MAX_LENGTH is a direct consequence of the ABS_TOL:
# Inverse of the MAX_LENGTH in the real space has to be meaningful
# in the reciprocal space (< ABS_TOL).
MAX_LENGTH = 1 / ABS_TOL

# TODO Think how to connect angle tolerance with spatial tolerance.

ABS_TOL_ANGLE = 1e-4  # Meant for the angular variables, in degrees.
REL_TOL_ANGLE = 1e-2  # Meant for the angular variables.
# MIN_ANGLE is a direct consequence of the REL_TOL_ANGLE and ABS_TOL_ANGLE:
# for a = MIN_ANGLE => ABS_TOL_ANGLE = a * REL_TOL_ANGLE
MIN_ANGLE = ABS_TOL_ANGLE / REL_TOL_ANGLE  # In degrees

ATOM_TYPES = (
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
)



