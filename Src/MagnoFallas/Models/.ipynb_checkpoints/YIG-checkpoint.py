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
The module used to Model yitrium iron garner (YIG) based on the published exchange interaction
(although, the modification of values to consider a usre-provided interactions is also available).\

Most of the functionality required for modeling is included into YIG class.

"""


import numpy as np
import numpy.linalg
import numba as nb
import scipy as sp
import copy

import MagnoFallas.OldRadtools as rad
from MagnoFallas.Interface import PseudoRad as prad
from MagnoFallas.Utils import util2 as ut2
from MagnoFallas.Utils import DipoleDipole as didi
from MagnoFallas.Interface import PseudoRad as prad

mev_to_THz = 0.2417990504024
cub_side0 = 12.376

###  The following values for exchange interaction in YIG are
### based on [npj Quantum Materials (2017) 2:63 ; doi:10.1038/s41535-017-0067-y]
JexDi0 = { 'J1': -6.8,       
         'J2': -0.52,
          'J3a': 0.0,
          'J3b': -1.1,
          'J4': 0.07,
          'J5': -0.47,
          'J6O': -0.09,
          'J6T': 0.0,
          'J7': 0.0,
          'J8': 0.0
}


class YIGbond:
    def __init__(self, name, dis, neiT, neiD, typs=('X','Y'), exc = 0.0):
        self.name=name
        self.d = dis
        self.nT = neiT
        self.nd = neiD
        self.typs = typs
        self.exc = exc

def comparebonds(bo1,bo2, prec=1e-5):
    b1 = np.abs(bo1.d - bo2.d) < prec
    b2 = (bo1.nT == bo2.nT)
    b3 = np.abs(bo1.nd- bo2.nd) < prec
    return (b1 and b2 and b3)

def checkBond(bnd, Known):
    res = -10
    for ib,bnd2 in enumerate(Known):
        if comparebonds(bnd,bnd2):
            res = ib
    return res

dvecs0 = []
for ix in (-1,0,1):
    for iy in (-1,0,1):
        for iz in (-1,0,1):
            dvecs0.append( (ix,iy,iz) )
dvecs0 = np.array(dvecs0)

def CercaAtom(r, SH, dvecs=dvecs0):
    drcur = 1e5
    for dv in dvecs:
        for at in SH.atoms:
            pos = at.position + dv
            rp = pos[0]*SH.a1 + pos[1]*SH.a2 + pos[2]*SH.a3
            dr = np.linalg.norm(r-rp)
            if dr < drcur:
                Ttyp = at.name[:3]
                Trp = rp
                drcur = dr
    return Trp, Ttyp

def bondInfo(at1,at2, dv, SH):
    pos1 = at1.position
    pos2 = at2.position + dv
    r1 = pos1[0]*SH.a1 + pos1[1]*SH.a2 + pos1[2]*SH.a3
    r2 = pos2[0]*SH.a1 + pos2[1]*SH.a2 + pos2[2]*SH.a3

    dist = np.linalg.norm(r1-r2)
    rc = (r1 + r2)/2

    rnei, neiTyp = CercaAtom(rc, SH)

    disnei = np.linalg.norm(rc-rnei)

    return dist, disnei, neiTyp




class YIG:
    def __init__(self, JexDi=JexDi0, cub_side=cub_side0, sdir=ut2.ez, Ndip = 2, R0= 0, B=1e-4, LR=False):
        r"""
        Initialization of the model of YIG
        JexDi - dictionary of the exchange interactions (the default value corresponds to the values extracted from [doi:10.1038/s41535-017-0067-y])
        cub_side - size of the cubic unit cell (in A)
        sdir - direction of spin polarization
        Rmax - maximum distance of Short-range dipole-dipole interaction (SRDD) included in the Hamiltonian (Rmax=0 means no SRDD is included)
        Ndip - superlatice size used to calculate Short-range dipole-dipole interaction
        B - effective magnetic field. Might be nessecery to stabilize magnons (especially when dipole-dipole interaction is included)
        LR - shows if Long-range dipole-dipole interaction should be included into the model
        """
        
        self.JexDi = JexDi
        self.cub_side = cub_side
        self.sdir = sdir
        self.B = B
        self.LR = LR
        self.R0 = R0

        self.prim_side = cub_side * np.sqrt(3)/2

        a1 = np.array((1,1,-1))
        a2 = np.array((1,-1,1))
        a3 = np.array((-1,1,1))
        side0 = np.linalg.norm(a1)
        
        self.a1 = a1*self.prim_side/side0
        self.a2 = a2*self.prim_side/side0
        self.a3 = a3*self.prim_side/side0

        self.SH = rad.SpinHamiltonian(cell=(self.a1,self.a2,self.a3), standardize=False)
        self.SH.notation = (False, False, -1)

        FeOpos = [(0.000000000,         0.000000000  ,       0.000000000),
                  (0.000000000,         0.500000000  ,       0.000000000),
                  (0.500000000,         0.500000000  ,       0.000000000),
                  (0.500000000,         0.000000000 ,        0.000000000),
                  (0.000000000,         0.000000000  ,       0.500000000),
                  (0.500000000,         0.500000000  ,       0.500000000),
                  (0.500000000,         0.000000000  ,       0.500000000),
                  (0.000000000,         0.500000000  ,       0.500000000)]
        
        FeTpos = [  (0.875000000,         0.250000000,         0.125000000),
                    (0.750000000,         0.875000000,         0.125000000),
                    (0.125000000,         0.875000000,         0.250000000),
                    (0.625000000,         0.375000000,         0.250000000),
                    (0.625000000,         0.750000000,         0.375000000),
                    (0.250000000,         0.625000000,         0.375000000),
                    (0.375000000,         0.250000000,         0.625000000),
                    (0.750000000,         0.375000000,         0.625000000),
                    (0.875000000,         0.125000000,         0.750000000),
                    (0.375000000,         0.625000000,         0.750000000),
                    (0.250000000,         0.125000000,         0.875000000),
                    (0.125000000,         0.750000000,         0.875000000)]
        
        li1 = []
        for i, pos in enumerate(FeOpos):
            nam = 'FeO' + str(i)
            at = rad.Atom(nam, spin=-5/2*self.sdir,   position=pos)
            li1.append(at)
            self.SH.add_atom(at)
        
        li2 = []
        for i, pos in enumerate(FeTpos):
            nam = 'FeT' + str(i)
            at = rad.Atom(nam, spin=5/2*self.sdir,   position=pos)
            self.SH.add_atom(at)
            li2.append(at)
        
        self.Nat = len(self.SH.atoms)
        self.realpos = np.zeros((self.Nat, 3))
        for i in range(self.Nat):
            pos = self.SH.atoms[i].position
            self.realpos[i] = pos[0]*self.a1 + pos[1]*self.a2 + pos[2]*self.a3

        x = cub_side/cub_side0
        self.BTypes = [YIGbond('J1v1', 3.4591970401084553*x, 'FeO', 1.7295985200542274*x, typs=('FeO','FeT'), exc = self.JexDi['J1']),
             YIGbond('J1v2', 3.4591970401084553*x, 'FeT', 1.7295985200542274*x, typs=('FeO','FeT'), exc = self.JexDi['J1']), 
             YIGbond('J2', 3.7893606622317186*x, 'FeT', 1.894680331115859*x, typs=('FeT','FeT'), exc = self.JexDi['J2']), 
             YIGbond('J3a', 5.35896519861*x, 'FeT', 2.1877882939566*x, typs=('FeO','FeO'), exc = self.JexDi['J3a']),
             YIGbond('J3b', 5.35896519861*x, 'FeO', 2.679482428767*x, typs=('FeO','FeO'), exc = self.JexDi['J3b']),
             YIGbond('J4',  5.577787611516*x, 'FeT', 2.32049996307063*x, typs=('FeO','FeT'), exc = self.JexDi['J4']),
             YIGbond('J5',  5.7883439838983*x, 'FeO', 1.8946802105283*x, typs=('FeT','FeT'), exc = self.JexDi['J5']),
             YIGbond('J6O',  6.18799990150747*x, 'FeT', 1.5469999753859*x, typs=('FeO','FeO'), exc = self.JexDi['J6O']),
             YIGbond('J6T',  6.18799990150747*x, 'FeT', 3.093999950775098*x, typs=('FeT','FeT'), exc = self.JexDi['J6T'])
        ]

        self.Exdmax = 0.0
        for b in self.BTypes:
            if (b.d > self.Exdmax) and (np.abs(b.exc)>1e-5):
                self.Exdmax = b.d
        self.Exdmax += 1e-3

        for iat1,at1 in enumerate(self.SH.atoms):
            for iat2,at2 in enumerate(self.SH.atoms):
                for dv in dvecs0:
                    dr = self.realpos[iat2] - self.realpos[iat1]
                    dr += dv[0]*self.a1 + dv[1]*self.a2 + dv[2]*self.a3
                    dis = np.linalg.norm(dr)
                    if (dis<self.Exdmax) and (dis>0.0001):
                        disX,disnei,neiTyp = bondInfo(at1,at2, dv, self.SH)
                        if not ( abs(dis-disX)<1e-6):
                            print('Ahtung!!! No sabes distanca!!!', dis, disX)
                        tmpB = YIGbond('tmp', dis, neiTyp, disnei)
                        chk = checkBond(tmpB, self.BTypes)
                        if chk>=0:
                            Jex = self.BTypes[chk].exc
                            if np.abs(Jex) > 1e-6:
                                if not ( (at1,at2, tuple(dv)) in self.SH   ):
                                    self.SH.add_bond(at1, at2, tuple(dv),   iso=Jex)  

        
        self.SH0 = copy.deepcopy(self.SH)
        if self.R0>0:
            self.SH = didi.AddDD(self.SH, NxMax=Ndip, NyMax=Ndip, NzMax=Ndip, exs=ut2.ex, eys=ut2.ey,ezs=ut2.ez, Rmax=self.R0)

        self.Magn = rad.MagnonDispersion(self.SH)

        self.spinproj = np.array([at.spin_vector@sdir for at in self.SH.magnetic_atoms])
        #self.pSH = prad.make_pSH2(self.SH, B=self.B)
        self.pSH = prad.make_pSH2(self.SH, B=self.B, LR=self.LR, dim=3, R0=self.R0)
        #self.pSH = prad.make_pSH(self.Magn, B=self.B, spinproj=self.spinproj)

    def Kpath(self, Nst=33):
        r"""
        Generates a k-path common for a representation of magnons in YIG
        Nst is the number of points per sector of k-path
        """
        pG = np.zeros(3)
        pN = np.array([np.pi/self.cub_side, np.pi/self.cub_side, 0])
        pH = np.array([2*np.pi/self.cub_side, 0, 0])

        points0 = [pN, pG, pH]
        labels = [r'$N$', r'$\Gamma$', r'$H$']
        Npoi0 = len(points0)
        kpoi = [pN,]
        Xmarks = [0,]
        xx = [0,]
        cnt=0
        xFull = 0
        for i in range(Npoi0-1):
            len1 = np.sqrt(np.sum( (points0[i+1] - points0[i])**2   ))
            dx = len1/Nst
            for i1 in range(Nst):
                cnt += 1
                xFull +=  dx
                k1 = ((Nst-i1-1)/Nst) * points0[i] + ((i1+1)/Nst)* points0[i+1]
                kpoi.append(k1)
                xx.append(xFull)
            Xmarks.append(xFull)
        return xx, kpoi, Xmarks, labels

        
                    






