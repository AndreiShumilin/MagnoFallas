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
import numba as nb

from typing import Callable

__all__ = ['FindZeroPoint']


Root_zero = 1.0e-8
False_value = 1.0e9


def FindZeroPoint(fv, vecs, tol=1e-10):
    Nv = len(vecs)
    vals = np.zeros(Nv)
    rv = np.zeros(3)
    for i in range(Nv):
        rv += vecs[i]
        vals[i] = fv(vecs[i])
    rv /= Nv
    Vr = fv(rv)
    
    if np.abs(Vr)<tol:
        return True, rv
    for i in range(Nv):
        if np.abs(vals[i])<tol:
            return True, vecs[i]
            
    iTarg = -1000
    exist = False
    for i in range(Nv):
        if Vr*vals[i] < 0:
            exist = True
            iTarg = i

    if not exist:
        return False, np.zeros(3, dtype=np.float64)
    
    V0 = rv
    V1 = vecs[iTarg]
    eI0 = Vr
    eI1 = vals[iTarg]
    k1 = 0
    k2 = 1
    kT = 0.5
    eT = fv(kT*V1 + (1-kT)*V0)
    
    while np.abs(eT)>tol:
        if (eT*eI0 < 0):
            k2 = kT
            kT = (kT+k1)/2
            eT = fv(kT*V1 + (1-kT)*V0)
        elif (eT*eI1 < 0):
            k1 = kT
            kT = (kT + k2)/2
            eT = fv(kT*V1 + (1-kT)*V0)
        else:
            return False, np.zeros(3, dtype=np.float64) 

    vec_fin = kT*V1 + (1-kT)*V0
    return True, vec_fin


@nb.njit
def FindRoot(foo: Callable, x1: float,x2 :float, N_maxtry=100) -> (bool, float):
    y1 = foo(x1)
    y2 = foo(x2)
    if np.abs(y1)<Root_zero:
        return (True, x1)
    if np.abs(y2)<Root_zero:
        return (True, x2)
    if y1*y2 > 0: 
        return (False, False_value)

    def guess(t1,t2, yt1, yt2):
        ay1 = np.abs(yt1)
        ay2 = np.abs(yt2)
        ay12 = ay1 + ay2
        return t1*(ay2/ay12) + t2*(ay1/ay12)
    
    for itry in range(N_maxtry):
        xt = guess(x1,x2,y1,y2)
        yt = foo(xt)
        if np.abs(yt)<Root_zero:
            return (True,xt)

        if (y1*yt)<0:
            x2 = xt
            y2 = yt
        elif (yt*y2)<0:
            x1 = xt
            y1 = yt
        else:
            return (False, False_value)
    return (False, False_value)
        


