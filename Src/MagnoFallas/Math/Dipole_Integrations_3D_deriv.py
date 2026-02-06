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

from MagnoFallas.Math import Dipole_Integrations_3D as dd3D
from MagnoFallas.Utils import util2 as ut2

### Parameters for the interpolations
### required to make the calculations of D-matrix
### numba-compatible
X0max = 150.0
Ninterpol = 3000



#
#  "IN" for N=0,1,2,3 corresponds to
#   IN = 2pi * int_R0^infty r^2 dr int_{-1}^{1} d cos(\theta)  cos(\theta)^N * exp(i kr cos(\theta) ) * r^{-4}


###----------------------   analytical integrals ----------------------------------------------------

def aI0(k,R0):
    x0 = k*R0
    Six0, Cix0 = sp.special.sici(x0)

    M1 = 4*np.pi/R0
    C1 = 2*x0*np.cos(x0) + 2*np.sin(x0) - (x0*x0)*(np.pi  - 2*Six0)
    D1 = 4*x0
    return M1*C1/D1


def aI1(k,R0):
    M1 = -np.pi*(0+4j)/R0
    x0 = k*R0
    Six0, Cix0 = sp.special.sici(x0)
    
    C1 = x0*np.cos(x0) + (x0**3)*Cix0 - (1+x0*x0)*np.sin(x0)
    D1 = 3*x0*x0
    return M1*C1/D1

def aI2(k,R0):
    x0 = k*R0
    Six0, Cix0 = sp.special.sici(x0)

    M1 = 4*np.pi/R0
    C1 = 2*x0*(2+x0*x0)*np.cos(x0) + 2*(x0*x0 - 2)*np.sin(x0) - (x0**4)*(np.pi - 2*Six0)
    D1 = 4*x0*x0*x0
    return M1*C1/D1

def aI3(k,R0):
    x0 = k*R0
    Six0, Cix0 = sp.special.sici(x0)

    #M1 = np.pi*(0+4j)/R0
    #C1 = 2*x0*(2+x0*x0)*np.cos(x0) + 2*(x0*x0-2)*np.sin(x0) - (x0**4)*(np.pi - 2*Six0)
    #D1 = 4*x0*x0
    M1 = -np.pi*(0+4j)/R0
    C1 = x0*(x0*x0-6)*np.cos(x0) + (x0**5)*Cix0 - (x0**4 + 3*x0*x0 - 6)*np.sin(x0)
    D1 = 5*(x0**4)
    return M1*C1/D1


### 1_2 corresponds tp cos(theta)^1 * sin(\theta)^2 \cos^2(\phi)
def aI1_2(k,R0):
    return 0.5*(aI1(k,R0)  - aI3(k,R0))

###--------------------------------------------------------------------------

## interpolations are made for R0 = 1 and effective k equal to k*R0
AX0int = np.linspace(0, X0max, Ninterpol)
stepX0 = AX0int[1] - AX0int[0]
AX0int[0] = 1e-4
datI0 = np.array([aI0(x, 1) for x in AX0int], dtype=np.complex128)
datI1 = np.array([aI1(x, 1) for x in AX0int], dtype=np.complex128)
datI2 = np.array([aI2(x, 1) for x in AX0int], dtype=np.complex128)
datI3 = np.array([aI3(x, 1) for x in AX0int], dtype=np.complex128)
datI1_2 = np.array([aI1_2(x, 1) for x in AX0int], dtype=np.complex128)
AX0int[0] = 0.0


###----------------------   interpolation integrals to be numba-compatible ----------------------------------------------------

@nb.njit
def intInm(k, R0, n,m):
    x0 = k*R0
    dat1 = np.zeros(Ninterpol, dtype=np.complex128)
    if m==0:
        if n==0:
            dat1 = datI0
        elif n==1:
            dat1 = datI1
        elif n==2:
            dat1 = datI2
        elif n==3:
            dat1 = datI3
    elif m==2:
        if n==1:
            dat1 = datI1_2

    if x0<=0:
        Res1 = dat1[0]
    elif (x0>=X0max):
        Res1 = dat1[-1]
    else:
        ix1 = int(x0 // stepX0)
        ix2 = ix1 + 1
        x1 = AX0int[ix1]
        x2 = AX0int[ix2]
        y1 = dat1[ix1]
        y2 = dat1[ix2]
        Res1 = y1 + (y2-y1)*(x0-x1)/(x2-x1)
    
    return Res1/R0
    


##-------------------------------- construction of the matricies ----------------------


@nb.njit
def kz_integrated_D(kz,R0):
    rI1 = intInm(kz, R0, 1, 0)
    rI3 = intInm(kz, R0, 3, 0)
    rI12 = intInm(kz, R0, 1, 2)

    Dx = np.zeros((3,3), dtype = np.complex128)
    Dx[0,2] = 3*rI1 - 15*rI12
    Dx[2,0] = 3*rI1 - 15*rI12

    Dy = np.zeros((3,3), dtype = np.complex128)
    Dy[1,2] = 3*rI1 - 15*rI12
    Dy[2,1] = 3*rI1 - 15*rI12

    Dz = np.zeros((3,3), dtype = np.complex128)
    Dz[0,0] = -15*rI12 + 3*rI1
    Dz[1,1] = -15*rI12 + 3*rI1
    Dz[2,2] = -15*rI3 + 9*rI1

    return Dx, Dy, Dz





@nb.njit
def kv_integrated_D(kv, R0, akmin=1e-5):
    ak = np.linalg.norm(kv)
    if ak<akmin:
        Dx = np.zeros((3,3), dtype=np.complex128)
        Dy = np.zeros((3,3), dtype=np.complex128)
        Dz = np.zeros((3,3), dtype=np.complex128)
        return Dx,Dy,Dz
    else:
        ek = kv/ak

    ex2,ey2,ez2 = dd3D.findEset(ek)

    D1_0, D2_0, D3_0 = kz_integrated_D(ak, R0)

    T = np.array((
      (ex2@ut2.ex, ex2@ut2.ey, ex2@ut2.ez),  
      (ey2@ut2.ex, ey2@ut2.ey, ey2@ut2.ez),  
      (ez2@ut2.ex, ez2@ut2.ey, ez2@ut2.ez)  
    ), dtype = np.complex128)
    Tt = np.transpose(T)
    
    D1 = Tt@D1_0@T
    D2 = Tt@D2_0@T
    D3 = Tt@D3_0@T

    exR = Tt[0]
    eyR = Tt[1]
    ezR = Tt[2]

    Dx = exR[0]*D1 + exR[1]*D2 + exR[2]*D3
    Dy = eyR[0]*D1 + eyR[1]*D2 + eyR[2]*D3
    Dz = ezR[0]*D1 + ezR[1]*D2 + ezR[2]*D3
    
    return Dx,Dy,Dz






