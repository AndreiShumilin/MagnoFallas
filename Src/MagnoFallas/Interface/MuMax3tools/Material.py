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



r"""
Not perfectly tested - interface with MuMax3
"""


#########################

import numpy as np
import matplotlib.pyplot as plt
import copy


import MagnoFallas as mfal
from MagnoFallas.Utils import util2 as ut2
from MagnoFallas.Utils import MicroMagnetics as micro

from .constants import *

##########################

__all__ = ['Material2D',]

def CheckIsoAex(AexTensor, AexTensorCutoffSi):
    r"""
    Checks if the exchange stiffness tensor
    corresponds to a scalar stiffnes
    """
    iso = True
    if np.abs(AexTensor[0,1]) > AexTensorCutoffSi:
        iso = False
    if np.abs(AexTensor[1,1] - AexTensor[0,0]) > AexTensorCutoffSi:
        iso = False
    return iso


def CheckSurfaceDMITensor(Dt, precision):
    r"""
    Checks if the provided DMI tensor corresponds to the conventional
    "surface DMI"
    """
    D0 = Dt[0,2,0]
    good = True
    if np.abs(Dt[0,0,2] + D0) > precision:
        good = False
    if np.abs(Dt[1,2,1] - D0) > precision:
        good = False
    if np.abs(Dt[1,1,2] + D0) > precision:
        good = False
    badElements = np.array([Dt[0,0,1], Dt[1,0,1], Dt[0,1,0], Dt[1,1,0],
                  Dt[0,1,2], Dt[0,2,1], Dt[1,0,2],Dt[1,2,0],Dt[2,0,1], Dt[2,1,0]])
    if np.max(np.abs(badElements)) > precision:
        good = False
    return good

def DerString(im, ix, C1=1):
    r"""
    Denerates string for the m-derivative
    """
    cell = ['cx', 'cy', 'cz']
    ee = ['eex','eey','eez']
    i1,j1,k1 = int(ix==0), int(ix==1), int(ix==2)
    i2,j2,k2 = -i1, -j1, -k1
    con1 = f'Const({C1*0.5}/' + cell[ix] +')'
    con2 = f'Const({-C1*0.5}/' + cell[ix] +')'
    str1 = f'Add( Mul({con1}, Shifted(m, {i1}, {j1}, {k1})), Mul({con2}, Shifted(m, {i2}, {j2}, {k2})))'
    str2 = f'Dot({str1}, {ee[im]})'
    return str2

def DMILines(m, D, alp, bet, gam):
    r"""Generates lines for a single component of the DMI tensor
    """
    Lines = []
    evecs = ['eex','eey','eez']
    nameFi = 'BDMIx' + str(m)
    nameFi1 = 'BDMIx' + str(m) +'x1'
    nameFi2 = 'BDMIx' + str(m) +'x2'
    nameFi3 = 'BDMIx' + str(m) + 'cmb'
    nameC1 = 'conD' + str(m) 
    Dl1 = f'{nameC1} :=  {D} / (Msat.Average()) \n'     #### Patch!!!!!!!!!!    (initially was -1*D)
    Lines.append(Dl1)
    derL1 =  DerString(gam, alp, C1=1) 
    l1 = f'{nameFi1} := Mul({evecs[bet]}, Mul(Const({nameC1}), {derL1} )) \n'
    Lines.append(l1)
    derL2 =  DerString(bet, alp, C1=-1) 
    l2 = f'{nameFi2} := Mul({evecs[gam]}, Mul(Const({nameC1}), {derL2} ))  \n'
    Lines.append(l2)
    l3 = f'{nameFi3} := Add({nameFi1}, {nameFi2}) \n'
    Lines.append(l3)
    if m==0:
        l4 = f'{nameFi} := {nameFi3} \n\n'
        Lines.append(l4)
    else:
        oldFi = 'BDMIx' + str(m-1)
        l4 = f'{nameFi} := Add({nameFi3}, {oldFi}) \n\n'
        Lines.append(l4)
    return Lines



    
def Material2D(SH, cz_nm = 1, AexTensorCutoffSi=AexTensorCutoffSi, AniCutoffSi=AniCutoffSi, DMICutoffSi=DMICutoffSi):
    r"""
    Prepears the MuMax3 script defining the materials properties (Aex, Anisotropi and DMI)
    """

    Scell = ut2.cellVolume(SH.cell, regime2D=True)
    M2Dx10nm = micro.Msat(SH, dim=2, zef=10*cz_nm)
    Msat = M2Dx10nm * const_muB_Si *1e30

    
    Lines = []
    Lines.append('/// ---->  Material Properties Block <------------: \n')
    Lines.append(f'msat = {Msat:.2e}  \n\n')

    ##------------------------- exchange stiffness tensor----------
    customEx = False
    customAniS = False
    customDMI = False
    customAni = False
    
    ATensor0 = micro.AExTensor(SH, zef=10*cz_nm) 
    ATensorSi = ATensor0*const_Aex_to_Si
    
    if CheckIsoAex(ATensorSi, AexTensorCutoffSi):
        Lines.append('///Exchange stiffness was found to be isotropic (in plane): \n')
        Aex = ATensorSi[0,0] 
        Lines.append(f'Aex = {Aex:.5e}')
    else:
        nonDiagAex = np.abs(ATensorSi[0,1]) > AexTensorCutoffSi
        
        Lines.append('///Exchange stiffness was found to be anisotropic and has to be introduced as a custom filed: \n')
        Aex = 0.75*np.min(np.diag(ATensorSi[:2]))
        Lines.append(f'Aex0 := {Aex:.5e}  \n /// sanity check: no results should depend on this parameter \n')
        Lines.append(f'Aex = Aex0   \n\n')

        Lines.append(f'Ae1xx := {ATensorSi[0,0] :.5e} - Aex0  \n')
        Lines.append(f'Ae1yy := {ATensorSi[1,1] :.5e} - Aex0   \n')
        if nonDiagAex:
            Lines.append(f'Ae1xy := {ATensorSi[0,1] :.5e}  \n')
        Lines.append('\n\n')
        Lines.append('ConstAe1xx := Const( (2 * Ae1xx) / (cx*cx*Msat.Average()) ) \n')
        Lines.append('ConstAe1yy := Const( (2 * Ae1yy) / (cy*cy*Msat.Average()) ) \n')

        Lines.append('Term1xx := Add(Shifted(m, 1,0,0), Shifted(m, -1,0,0)) \n')
        Lines.append('Term2xx := Mul(Const(-2.0), m) \n')
        Lines.append('exBxx := Mul( ConstAe1xx, Add(Term1xx, Term2xx) )     \n')
        Lines.append('Term1yy := Add(Shifted(m, 0,1,0), Shifted(m, 0,-1,0)) \n')
        Lines.append('Term2yy := Mul(Const(-2.0), m) \n')
        Lines.append('exByy := Mul( ConstAe1yy, Add(Term1yy, Term2yy) )   \n')
        if nonDiagAex:
            Lines.append('\n')

            Lines.append('ConstAe1xy := Const( (Ae1xy) / (cx*cy*Msat.Average()) ) \n')
            Lines.append('Term1xy := Add(Shifted(m, 1,1,0), Shifted(m, -1,-1,0) ) \n')
            Lines.append('Term2xy := Mul(Const(-1.0),  Add(Shifted(m, 1,-1,0), Shifted(m, -1,1,0) ) )\n')
            Lines.append('exBxy := Mul( ConstAe1xy, Add(Term1xy,Term2xy)   )  \n')
            
            Lines.append('Bex0 := Add(exByy, exBxy)   \n' )
            Lines.append('Bex := Add(exBxx, Bex0)   \n' )
            Lines.append('Wex := Mul(Const(-0.5), Dot(Bex,M_full))   \n' )
            customEx = True
        else:
            Lines.append('Bex := Add(exBxx, exByy)   \n' )
            Lines.append('Wex := Mul(Const(-0.5), Dot(Bex,M_full))   \n' )
            customEx = True
            
    ##-----------------------------------
    Lines.append('\n\n\n' )

    ##-----------    Anisotropy  ---------
    K1, K2, aK1, aK2 = micro.AniParameters(SH, zef=10*cz_nm)  
    K1Si = K1*const_Ani_to_Si
    K2Si = K2*const_Ani_to_Si
    if (np.abs(K1Si)<AniCutoffSi) and (np.abs(K2Si)<AniCutoffSi):
        Lines.append('///No significant anisotropy detected  \n\n\n')
    elif (np.abs(K2Si)<AniCutoffSi):
        Lines.append('///Hard Axis anisotropy detected  \n')
        Lines.append(f'Ku1 = {-K1Si:.5e} \n')
        Lines.append(f'anisU   = vector({aK1[0]}, {aK1[1]}, {aK1[2]}) \n')
    elif (np.abs(K2Si - K1Si)<AniCutoffSi):
        Lines.append('///Easy Axis anisotropy detected  \n')
        axK0 = np.cross(aK1,aK2)
        axK0 /= np.linalg.norm(axK0)
        Lines.append(f'Ku1 = {K1Si:.5e}\n')
        Lines.append(f'anisU   = vector({axK0[0]}, {axK0[1]}, {axK0[2]})\n')
    else:
        Lines.append('///3-Axis "even" anisotropy detected  \n')
        Lines.append('///has to be included as a custom field  \n\n')
        Lines.append(f'K1 := {K1Si:.5e}\n')
        Lines.append(f'K2 := {K2Si:.5e}\n')
        Lines.append(f'K1factor := Const( (-2 * K1) / (Msat.Average())) \n')
        Lines.append(f'K2factor := Const( (-2 * K2) / (Msat.Average())) \n')
        Lines.append(f'K1u :=  Normalized(ConstVector({aK1[0]}, {aK1[1]}, {aK1[2]})) \n')
        Lines.append(f'K2u :=  Normalized(ConstVector({aK2[0]}, {aK2[1]}, {aK2[2]})) \n')
        Lines.append('\n')
        Lines.append('AnisField1 := Mul(K1factor, Mul( Dot(K1u, m), K1u)) \n')
        Lines.append('AnisField2 := Mul(K2factor, Mul( Dot(K2u, m), K2u)) \n')
        Lines.append('BaniS := Add(AnisField1, AnisField2)   \n' )
        Lines.append('WaniS := Mul(Const(-0.5),Dot(BaniS, M_full))   \n' )
        customAniS = True
    
    ##-------------------   DMI  -------------------------------------
    Lines.append('\n\n')
    DMI0 = micro.DMItensor(SH, dim=2, zef=10*cz_nm)
    DMI = DMI0*const_DMI_to_Si
    #print(np.max(np.abs(DMI)))
    if np.max(np.abs(DMI)) < DMICutoffSi:
        customDMI = False
        Lines.append('/// No significant DMI detected \n')
    elif CheckSurfaceDMITensor(DMI, DMICutoffSi):
        customDMI = False
        Lines.append('/// Surface DMI detected \n')
        Lines.append(f'Dind = {0.5*DMI[0,2,0]:.5e} \n')    #### Patch!!!!!!!!!!
    else:
        customDMI = True
        Lines.append('/// anisotropic DMI detected: has to be described by a custom field \n\n')
        ##['eex','eey','eez']
        Lines.append('eex :=Normalized(ConstVector(1,0,0)) \n' )
        Lines.append('eey :=Normalized(ConstVector(0,1,0)) \n' )
        Lines.append('eez :=Normalized(ConstVector(0,0,1)) \n\n' )
        
        mDMI = 0
        for alp in [0,1]:
            for bet in [0,1,2]:
                for gam in range(bet):
                    D1 = DMI[alp,bet,gam]
                    if np.abs(D1) > DMICutoffSi:
                        Lns1 = DMILines(mDMI, D1, alp, bet, gam)
                        Lines = Lines + Lns1
                        mDMI += 1
        lastField = 'BDMIx' + str(mDMI-1)
        Lines.append(f'BDMI := {lastField}\n')
        Lines.append(f'WDMI := Mul(Const(-0.5), Dot(BDMI,M_full))  \n\n')
        
    ###----------------Combining custom fields--------------------------
    Lines.append('\n\n\n')
    if customEx or customAniS or customDMI:
        Lines.append('///combining custom fields: \n')   

    if not(customAniS or customDMI):
        customAni = False
    else:
        customAni = True
        if customAniS and not(customDMI):
            Lines.append('Bani := BaniS \n')
            Lines.append('Wani := WaniS \n')
        if not(customAniS) and customDMI:
            Lines.append('Bani := BDMI \n')
            Lines.append('Wani := WDMI \n')
        else:
            Lines.append('Bani := Add(BaniS, BDMI) \n')
            Lines.append('Wani := Add(WaniS, WDMI) \n')
        Lines.append('\n')
        
    if customEx and (not customAni):
        Lines.append('cusFie := Bex \n')
        Lines.append('cusEne := Wex \n')
    elif not(customEx) and customAni:
        Lines.append('cusFie := Bani \n')
        Lines.append('cusEne := Wani \n')
    elif customEx and customAni:
        Lines.append('cusFie := Add(Bani, Bex) \n')
        Lines.append('cusEne := Add(Wani, Wex) \n')

    if customEx or customAni:
        Lines.append('addFieldTerm(cusFie) \n')   
        Lines.append('addEdensTerm(cusEne) \n') 
    Lines.append('/// ---->  end of materials properties <------------: \n')
    Lines.append('\n\n\n')
    return Lines    

