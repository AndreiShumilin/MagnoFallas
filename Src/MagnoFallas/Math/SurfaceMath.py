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
import scipy as sp
import numba as nb


__all__ = ['bilinInt0','bilinDint','InitDirect','FindZeroPoint','bilinDintV']

@nb.jit(nopython=True)
def bilinInt0(dx,dy,A,B,C,D, betaTol = 1e-6):
    ##it is assumed that A and B have different signes
    ## A=Q11 B=Q21 C=Q12 D=Q22
    if (A*C < 0):
        yM = dy*np.abs(A)/np.abs(C-A)
        #print('x1', yM)
    elif (B*D <0):
        yM = dy*np.abs(B)/np.abs(D-B)
        #print('x2', yM)
    else:
        yM = dy
        #print('x3', yM)
    alp = (B-A)/dx
    bet = (D-C-B+A)/(dx*dy)
    #print(alp,bet)

    if np.abs(bet*yM) < betaTol*np.abs(alp):
        return np.abs(yM/alp)
    if (alp/bet)*(yM + alp/bet)<=0:
        #print('log fail', alp/bet, yM + alp/bet)
        return 0.0
    else:
        c1 = (bet*yM + alp)/(alp)
        res = np.log(c1)/bet
        return np.abs(res)

@nb.jit(nopython=True)
def bilinDint(x1,x2,y1,y2, f11,f12,f21,f22):
    ###
    ###  integrate Dirac delta(f) over some square
    ###  f is represented by bilinear interpolation
    ###  it is assumed that f=0 condition is relized only in one line that crosses two 
    ###  borders of the square
    ###
    if (f11*f21)<0:
        #print('v1')
        return bilinInt0(x2-x1, y2-y1, f11, f21, f12, f22)
    elif (f21*f22)<0:
        #print('v2')
        return bilinInt0(y2-y1, x1-x2, f21, f22, f11, f12)
    elif (f12*f22)<0:
        #print('v3')
        return bilinInt0(x1-x2, y1-y2, f22, f12, f21, f11)
    elif (f12*f11)<0:
        #print('v4')
        return bilinInt0(y1-y2, x2-x1, f12, f11, f22, f21)
    else:
        return 0.0



def bilinDintV(vecs, vals):
    ##### the version where the points are arbitrary
    ##### devined by vecs (4 x 3D vectors) and the corresponding (vals)
    #### expected order:  BL - TL - BR - TR

    f11,f12,f21,f22 = vals[0], vals[1], vals[2], vals[3]
    e1 = vecs[1] - vecs[0]
    e2 = vecs[2] - vecs[0]
    Jac = e1[0]*e2[1] - e1[1]*e2[0]
    Jac = np.abs(Jac)

    Int0 = bilinDint(0.0,1.0,0.0,1.0, f11,f12,f21,f22)  
    return Int0*Jac






@nb.jit(nopython=True)
def InitDirect(x1,x2,y1,y2, f11,f12,f21,f22, Nx=264, Ny=264, de=0.01):
    M = np.array( ((f11,f12),(f21,f22))  )
    dx = x2-x1
    dy = y2-y1
    def fo(x,y):
        u = np.array((x2-x,x-x1))
        v = np.array((y2-y,y-y1))
        return u@(M@v)/(dx*dy)
        
    res = 0.0
    ax = np.linspace(x1,x2,Nx)
    ay = np.linspace(y1,y2,Ny)
    for x in ax:
        for y in ay:
            f = fo(x,y)
            r = np.exp(-f*f/(2*de*de))
            res += r/(de*np.sqrt(2*np.pi))
    return res*(x2-x1)*(y2-y1)/(Nx*Ny)
            


#### now there is a better function generalMath.FindZeroPoint
def FindZeroPoint(fxy, x1,x2,y1,y2, far=None):
    if far != None:
        f11,f12,f21,f22 = far
    else:
        f11,f12,f21,f22 = fxy(x1,y1),fxy(x1,y2),fxy(x2,y1),fxy(x2,y2)
    exist = False
    p1 = np.zeros(2)
    if (f11>0 and f12>0 and f21>0 and f22>0):
        print('Error: surfacemath: all positive')
        return exist, p1
    if (f11<0 and f12<0 and f21<0 and f22<0):
        print('Error: surfacemath: all negative')
        return exist, p1
        
    if (f11*f22 <= 0):
        exist = True
        def ft(t):
            return fxy(x1 + t*(x2-x1), y1 + t*(y2-y1) )
        root = sp.optimize.root_scalar(ft, bracket=(0.0,1.0) )
        t1 = root['root']
        p1 = np.array( (x1 + t1*(x2-x1), y1 + t1*(y2-y1) ) )
        return exist, p1
        
    elif (f12*f21 <= 0):
        exist = True
        def ft(t):
            return fxy(x1 + t*(x2-x1), y2 + t*(y1-y2) )
        root = sp.optimize.root_scalar(ft, bracket=(0.0,1.0) )
        t1 = root['root']
        p1 = np.array( (x1 + t1*(x2-x1), y2 + t1*(y1-y2) ) )
        return exist, p1
        
    elif (f11*f21 <= 0):
        exist = True
        def ft(t):
            return fxy(x1 + t*(x2-x1), y1)
        root = sp.optimize.root_scalar(ft, bracket=(0.0,1.0) )
        t1 = root['root']
        p1 = np.array( (x1 + t1*(x2-x1), y1) )
        return exist, p1

    elif (f11*f12 <= 0):
        exist = True
        def ft(t):
            return fxy(x1, y1 + t*(y2-y1)  )
        root = sp.optimize.root_scalar(ft, bracket=(0.0,1.0) )
        t1 = root['root']
        p1 = np.array( (x1, y1 + t*(y2-y1) ) )
        return exist, p1

    elif (f22*f12 <= 0):
        exist = True
        def ft(t):
            return fxy(x2 - t*(x2-x1) , y2   )
        root = sp.optimize.root_scalar(ft, bracket=(0.0,1.0) )
        t1 = root['root']
        p1 = np.array( (x2 - t*(x2-x1) , y2) )
        return exist, p1

    elif (f22*f21 <= 0):
        exist = True
        def ft(t):
            return fxy(x2, y2 - t*(y2-y1)  )
        root = sp.optimize.root_scalar(ft, bracket=(0.0,1.0) )
        t1 = root['root']
        p1 = np.array( (x2, y2 - t*(y2-y1)) )
        return exist, p1
    
    else:
        print('Error: surfacemath: out of options')
        return exist, p1









