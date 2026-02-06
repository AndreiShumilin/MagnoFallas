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
import scipy as sp
import numba as nb

__all__ = ['interpolImatr2D']

ex = np.array((1,0,0), dtype=np.float64)
ey = np.array((0,1,0), dtype=np.float64)
ez = np.array((0,0,1), dtype=np.float64)


@nb.njit
def kz_integrated_M(kz, R0):
    kr = kz*R0
    F1 = 4*kr*np.cos(kr) - 4*np.sin(kr)
    F1 /= (kr)**3
    F2 = -2*kr*np.cos(kr) + 2*np.sin(kr)
    F2 /= (kr)**3
    dia = np.array((F2,F2,F1))
    Mat = np.diag(dia)
    return 2*np.pi*Mat*(1+0j)

@nb.njit
def findEset(ez2):
    if (ez2@ez==1):
        return ex,ey,ez
    elif (ez2@ez==-1):
        return ex,-ey,-ez
    else:
        ex2 = np.cross(ez,ez2)
        ex2 /= np.linalg.norm(ex2)
        ey2  = np.cross(ez2,ex2)
        ey2 /= np.linalg.norm(ey2)
        return ex2, ey2, ez2

@nb.njit
def kv_integrated_M(kv, R0, akmin=1e-5):
    ak = np.linalg.norm(kv)
    if ak<akmin:
        return np.zeros((3,3), dtype=np.complex128)
    else:
        ek = kv/ak

    ex2,ey2,ez2 = findEset(ek)

    M0 = kz_integrated_M(ak, R0)

    T = np.array((
      (ex2@ex, ex2@ey, ex2@ez),  
      (ey2@ex, ey2@ey, ey2@ez),  
      (ez2@ex, ez2@ey, ez2@ez)  
    ), dtype=np.complex128)
    Tt = np.transpose(T)
    M = Tt@M0@T
    return M
