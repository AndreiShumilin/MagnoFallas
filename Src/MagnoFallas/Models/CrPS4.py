import numpy as np

import MagnoFallas.OldRadtools as rad
from MagnoFallas.Utils import util2 as ut2

# import MagnoFallas as mfal
# import MagnoFallas.SHtools.tools as tools
# import MagnoFallas.Utils.util2 as ut2
# import MagnoFallas.OldRadtools as rad

r"""
Model spin Hamiltonian of CrPS4 based on the article:
Nano Lett. 2026, 26, 9, 3018–3025
https://doi.org/10.1021/acs.nanolett.5c05445
(including the corresponsing SI)
"""

# ex = np.array((1.0,0.0,0.0), dtype = np.float64)
# ey = np.array((0.0,1.0,0.0), dtype = np.float64)
# ez = np.array((0.0,0.0,1.0), dtype = np.float64)

ebasis = ut2.ebasis.copy()

a1 = np.array((-5.4983532427, -0.6781217533, 3.5092830665))
a2 = np.array((0.0630374719, 5.5362500618, -3.5092830665))
a3 = np.array((-5.0280860016, -5.6361525335, -9.6530377078))

ex1 = a1/np.linalg.norm(a1)
ez1 = np.cross(a1,a2)
ez1 = ez1 / np.linalg.norm(ez1)
ey1 = np.cross(ez1, ex1)
ebasis1 = np.array((ex1,ey1,ez1))

T1 = np.zeros((3,3))
for i in range(3):
    for j in range(3):
        T1[i,j] = ebasis1[i]@ebasis[j]



aJ1 = 2.85 
aJ2 = 2.59 
aJ3 = 0.05 
aJ4 = 1.11 
aJ5 =-0.83 

aMAEx = 42.3
aMAEy = 42.6


def realpos(at, SH):
    return at.position[0]*SH.a1 + at.position[1]*SH.a2 + at.position[2]*SH.a3

### MODEL
def model(J1=aJ1, J2=aJ2, J3=aJ3, J4=aJ4, J5=aJ5, T=T1, Hnotation = (True, True, -1),
              MAEx=aMAEx, MAEy=aMAEy):
    
    a1 = T@np.array((-5.4983532427, -0.6781217533, 3.5092830665))
    a2 = T@np.array((0.0630374719, 5.5362500618, -3.5092830665))
    a3 = T@np.array((-5.0280860016, -5.6361525335, -9.6530377078))
    
    SH = rad.SpinHamiltonian(cell=(a1,a2,a3), standardize=False)
    SH.notation = Hnotation
    
    S = 3/2
    
    rCr1 = np.array((-0.0001153734, -0.0001153734, 0.2500000000  ))
    rCr2 = np.array((0.5082769486, 0.5082769486, 0.2500000000   ))
    
    Cr1 =  rad.Atom("Cr1", spin=(0,0,S),   position=rCr1)
    Cr2 =  rad.Atom("Cr2", spin=(0,0,S),   position=rCr2)

    matMAE = np.diag((-1e-3*MAEx, -1e-3*MAEy, 0))

    SH.add_atom(Cr1)
    SH.add_atom(Cr2)


    SH.add_bond(Cr1, Cr1, (0,0,0),   matrix=matMAE)  
    SH.add_bond(Cr2, Cr2, (0,0,0),   matrix=matMAE)  
    
    SH.add_bond(Cr2, Cr1, (1,1,0),   iso=J1)  
    
    SH.add_bond(Cr1, Cr2, (0,0,0),   iso=J2)  
    
    SH.add_bond(Cr1, Cr2, (-1,0,0),   iso=J3)  
    SH.add_bond(Cr2, Cr1,  (0,1,0),   iso=J3)  
    
    SH.add_bond(Cr1, Cr1, (1,0,0),   iso=J4)  
    SH.add_bond(Cr1, Cr1, (0,1,0),   iso=J4)  
    SH.add_bond(Cr2, Cr2, (1,0,0),   iso=J4)  
    SH.add_bond(Cr2, Cr2, (0,1,0),   iso=J4)  
     
    #MagM = rad.MagnonDispersion(SH)
    return SH   #, MagM