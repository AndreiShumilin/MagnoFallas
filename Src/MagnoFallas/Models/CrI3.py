import numpy as np

import MagnoFallas.OldRadtools as rad

r"""
Model spin Hamiltonian of CrI3 based on the article:
J.L. Lado and J. Fernández-Rossier 2017 2D Mater. 4 035002
DOI 10.1088/2053-1583/aa75ed
"""


a0 = 6.867
c0 = 20.0

J0 = 2.25
lam0 = 0.09
D0 = 0.0


def model(a=a0, J=J0, lam=lam0, D=D0, c=c0):
    a1 = np.array((a*np.sqrt(3)/2, -a/2, 0))
    a2 = np.array((0, a, 0))
    a3 = np.array((0, 0, c))
    
    SH = rad.SpinHamiltonian(cell=(a1,a2,a3), standardize=False)
    SH.notation = (False, False, -1)
    
    S = 3/2
    
    rCr1 = np.array((0.0, 0.0, 0.0  ))
    #rCr2 = np.array((1/np.sqrt(3), 0.0, 0.0 ))
    rCr2 = np.array((2/3, 1/3, 0.0 ))
    
    Cr1 =  rad.Atom("Cr1", spin=(0,0,S),   position=rCr1)
    Cr2 =  rad.Atom("Cr2", spin=(0,0,S),   position=rCr2)

    matMAE = np.diag((0, 0, D))
    matBND = np.diag((J, J, J+lam))

    SH.add_atom(Cr1)
    SH.add_atom(Cr2)


    SH.add_bond(Cr1, Cr1, (0,0,0),   matrix=matMAE)  
    SH.add_bond(Cr2, Cr2, (0,0,0),   matrix=matMAE)  
    
    SH.add_bond(Cr1, Cr2, (0,0,0),   matrix=matBND)  
    SH.add_bond(Cr1, Cr2, (-1,0,0),   matrix=matBND)  
    SH.add_bond(Cr1, Cr2, (-1,-1,0),   matrix=matBND)  
    return SH     
    

