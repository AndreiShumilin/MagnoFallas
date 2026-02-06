import numpy as np
import scipy as sp
import numba as nb


__all__ = ['interpolImatr2D']


ex = np.array((1,0,0), dtype=np.float64)
ey = np.array((0,1,0), dtype=np.float64)
ez = np.array((0,0,1), dtype=np.float64)
ebasis = np.array([ex,ey,ez])


### Parameters for the interpolations
### required to make the calculations of D-matrix
### numba-compatible
X0max = 50.0
Ninterpol = 1000




def analitic2DIx_X0(X0):
    pre = np.pi/(X0*X0)

    T1 = 2*(X0*X0)

    T2 = sp.special.jv(1, X0)
    T2a = -np.pi*(X0**3)*sp.special.struve(0,X0)
    T2 = T2*(2 + 2*X0*X0 + T2a)

    T3 = X0*sp.special.jv(0, X0)
    T3a = -2*(1+X0*X0)
    T3b = np.pi*(X0*X0)*sp.special.struve(1,X0)
    T3 = T3*(T3a + T3b)
    return -pre*(T1 + T2 + T3)*X0


def analitic2DIy_X0(X0):
    pre = 2*np.pi
    T1 = sp.special.jv(1, X0)/X0
    return pre*T1

def analitic2DIz_X0(X0):
    pre = np.pi

    T1 = 2
    T2 = -2*(1+X0*X0)*sp.special.jv(0, X0)/X0
    T3 = sp.special.jv(1, X0)*(2-np.pi*X0*sp.special.struve(0,X0))
    T4 = np.pi*X0*sp.special.jv(0, X0) * sp.special.struve(1,X0)

    return pre*(T1 + T2 + T3 + T4)*X0

def analitic2DIx(k,R0):
    X0 = k*R0
    if X0<1e-3:
        X0 = 1e-3
    return k*analitic2DIx_X0(X0)/X0

def analitic2DIy(k,R0):
    X0 = k*R0
    if X0<1e-3:
        X0 = 1e-3
    return k*analitic2DIy_X0(X0)/X0

def analitic2DIz(k,R0):
    X0 = k*R0
    if X0<1e-3:
        X0 = 1e-3
    return k*analitic2DIz_X0(X0)/X0


AX0int = np.linspace(0, X0max, Ninterpol)
stepX0 = AX0int[1] - AX0int[0]
AX0int[0] = 1e-4
IdataX = np.array([analitic2DIx_X0(x) for x in AX0int])
IdataY = np.array([analitic2DIy_X0(x) for x in AX0int])
IdataZ = np.array([analitic2DIz_X0(x) for x in AX0int])
AX0int[0] = 0.0

@nb.njit
def interpol2DIx(k,R0):
    X0 = k*R0
    if X0<=0:
        Res1 = IdataX[0]
    elif (X0>=X0max):
        Res1 = IdataX[-1]
    else:
        ix1 = int(X0 // stepX0)
        ix2 = ix1 + 1
        x1 = AX0int[ix1]
        x2 = AX0int[ix2]
        y1 = IdataX[ix1]
        y2 = IdataX[ix2]
        Res1 = y1 + (y2-y1)*(X0-x1)/(x2-x1)
    
    return Res1/R0


@nb.njit
def interpol2DIy(k,R0):
    X0 = k*R0
    if X0<=0:
        Res1 = IdataY[0]
    elif (X0>=X0max):
        Res1 = IdataY[-1]
    else:
        ix1 = int(X0 // stepX0)
        ix2 = ix1 + 1
        x1 = AX0int[ix1]
        x2 = AX0int[ix2]
        y1 = IdataY[ix1]
        y2 = IdataY[ix2]
        Res1 = y1 + (y2-y1)*(X0-x1)/(x2-x1)
    
    return Res1/R0


@nb.njit
def interpol2DIz(k,R0):
    X0 = k*R0
    if X0<=0:
        Res1 = IdataZ[0]
    elif (X0>=X0max):
        Res1 = IdataZ[-1]
    else:
        ix1 = int(X0 // stepX0)
        ix2 = ix1 + 1
        x1 = AX0int[ix1]
        x2 = AX0int[ix2]
        y1 = IdataZ[ix1]
        y2 = IdataZ[ix2]
        Res1 = y1 + (y2-y1)*(X0-x1)/(x2-x1)
    
    return Res1/R0

@nb.njit
def interpolImatr2D_x(kx,R0):
    dia = np.array( (interpol2DIx(kx,R0), interpol2DIy(kx,R0), interpol2DIz(kx,R0))   )
    return np.diag(dia)



##### main procedure, to be used outside the module
@nb.njit
def interpolImatr2D(kv,R0):
    kv2 = kv * np.array((1,1,0))
    ak = np.linalg.norm(kv2)
    if ak == 0:
        ex2 = ex
    else:
        ex2 = kv2/ak
    ey2 = np.cross(ez,ex2)
    ey2 = ey2/np.linalg.norm(ey2)

    Mat0 = interpolImatr2D_x(ak,R0)

    T = np.array((
      (ex2@ex, ex2@ey, 0),  
      (ey2@ex, ey2@ey, 0),  
      (0,      0,      1)  
    ))
    Tt = np.transpose(T)
    
    M = Tt@Mat0@T
    return M

