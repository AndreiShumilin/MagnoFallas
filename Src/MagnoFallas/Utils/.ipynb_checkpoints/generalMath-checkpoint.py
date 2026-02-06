##### modified 14.05.2025

import numpy as np
import numba as nb

__all__ = ['FindZeroPoint']

#@nb.njit
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
            