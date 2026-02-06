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
import numba as nb
import scipy as sp
import matplotlib.pyplot as plt
import copy

import MagnoFallas.OldRadtools as rad

from MagnoFallas.Interface import PseudoRad as prad
from MagnoFallas.Utils import util2 as ut2
from MagnoFallas.Utils import DipoleDipole as dd
from MagnoFallas.Utils import Logs
from MagnoFallas.Utils import Grids

from MagnoFallas.SpinPhonon import SPhUtil as sphut
from MagnoFallas.SpinPhonon import StrategyDipDip as sphDD

from MagnoFallas.Quantum.process2M1Ph import Quantum2M1Ph as quant
from MagnoFallas.Quantum.process2M1Ph import Quantum2M1Ph_LRDD as quantLR
from MagnoFallas.Boltzmann.process2M1Ph import ScatteringList2M1Ph as slist
from MagnoFallas.Boltzmann.process2M1Ph import kinetics2M1Ph as kin


class Boltzman_alpha2M1Ph:
        
    def __init__(self, SH, Ephon, Ngx, Ngy, Ngz, rKxM = 1.0, rKyM = 1.0, rKzM = 1.0, dim=None, lamAcu=None, Ecut=None, roles = slist.rolesMC,
                B = 0.0, MaxBranch = None, Name = 'bolt_2M_1Ph', atom_map = None,
                SpinPhon = None,
                LRdd=False, includeSRDD=False, R0=12):
        r"""
        SH - spin Hamiltonian (note: should not include dipole-dipole interactions if includeSRDD is true)
        Ephon - information on phonons according to Interface.UtilPhonopy 
        Ngx, Ngy, Ngz - size of the k-grid
        rKxM, rKyM, rKzm - relaive cutoffs for the grid along b1, b2, b3 vectors of SH
        dim - dimensionality (None leads to automatic selectrion based on Ngz)
        lamAcu - number of acoustic magnon branch (None leads to automatic selection)
        Ecut - energy cutoff (not used)
        roles - for magnon-conserving/magnon-non-conserving processes
        B - external magnetic field [T] (sometimes important to stabilize magnons with respect to dipole-dipole interaction)

        MaxBranch - maximum phonon branch
        Name - name of the process
        atom_map - map of atom names, required to relate spin hamiltonian with phonons (none if names are atomic symbols)

        SpinPhon - External information of the spin-phonon interaction. "None" still allows the calculation with interaction from the 
                    dipole-dipole interaciton
                  
        
        ### long range dipole-dipole interaction, to be implemented
        LRdd - wether to include long-rage DD
        includeSRDD - True leads to automatic addition of short-range DD with cutoff R0
        R0 - cutoff for long/short range dipole-dipole interaction
        """
        self.Ngx = Ngx
        self.Ngy = Ngy
        self.Ngz = Ngz
        self.B = B
        self.roles = roles
        self.rKxM = rKxM   #### relative maximum momentum along b1 (1 corresponds to whole range [-b1/2,b1/2])
        self.rKyM = rKyM
        self.rKzM = rKzM
        if dim is None:
            if Ngz <= 1:
                self.dim=2
            else:
                self.dim=3
        else:
            self.dim = dim


        self.Name = Name
        self.atom_map = atom_map
        self.logfile = self.Name + '.log'
        self.Log = Logs.Log(self.logfile)

        self.Log.Twrite2('2Magnon 1 Phonon calcualtions')
        self.Log.write('process name = ' + self.Name)
        self.Log.write('roles = ' + str(self.roles))   
        self.Log.cut()
        
        SHzero = SH
        self.LR = LRdd
        self.SR = includeSRDD
        if (self.LR or self.SR):
            self.R0 = R0
        if self.SR:
            self.NddX, self.NddY, self.NddZ = dd.calculateNdd(SHzero, self.R0, dim=self.dim)
            self.SH = dd.AddDD(SHzero, NxMax=self.NddX, NyMax=self.NddY, NzMax=self.NddZ, Rmax = self.R0)
        else:
            self.SH = copy.deepcopy(SHzero)

        self.realpositions = np.array([ut2.realposition(self.SH, at) for at in self.SH.magnetic_atoms ])
        self.Magn = rad.MagnonDispersion(self.SH)

        ##-------------
        if self.LR:
            self.pSH = prad.make_pSH2(self.SH, self.B, LR=self.LR, dim=self.dim, R0=self.R0)  
                            ###Note prad pSH should be made from initial (non-ferromagnetized) spin Hamiltonian
                            ### to correctly track the effect of magnetic field on sublattices
        else:
            self.pSH = prad.make_pSH2(self.SH, self.B, LR=False, dim=2, R0=0.0)  
        ##----------------
        
        self.Ephon = Ephon
        self.liMag, self.liAt = self.Ephon.relate(self.SH, dim=self.dim, Nmap = self.atom_map)
        
        if self.SR:
            SpinPhon2 = sphDD.Estimate_auto_dipole(self.SH, self.realpositions, self.liMag, disMax = self.R0, lst=SpinPhon)
        else:
            SpinPhon2 = SpinPhon
        self.SpinPhon = SpinPhon2

        self.Nb =  len(self.SH.magnetic_atoms)
        
        if MaxBranch is None:
            self.Nbranch = self.Ephon.Nband
        else:
            self.Nbranch = MaxBranch
        
        if lamAcu is None:
            self.lamAcu = self.Nb-1
        else:
            self.lamAcu = lamAcu

        self.KGr, self.gGr = Grids.KGrid(self.Ngx, self.Ngy, self.Ngz,  self.SH.cell,  rKXmax=self.rKxM, rKYmax=self.rKyM,
                                       rKZmax=self.rKzM, regime2D=(self.dim == 2))
        self.Log.Twrite('K-grid created')
        self.EGrS = Grids.Egrid_prad(self.pSH, self.KGr)
        self.Log.Twrite('EM-grid created')
        self.EGrPh = Grids.Egrid_Ephonon(self.Ephon, self.KGr,  MaxBranch = self.Nbranch)
        self.Log.Twrite('EPh-grid created')
        
        kG = np.zeros(3)
        self.E0 = prad.omega(self.pSH, kG)[0][self.Nb - 1]

        self.cellV = ut2.cellVolume(self.SH.cell, regime2D=(self.dim==2))
        

    def initialize(self):
        r"""
        Finds scattering events; 
        calculates the real k-vectors
        calculates delta function integrals
        find matrix elements

        most slow part ofthe calculations
        """
        self.Log.Twrite2('Initialization step')
        self.Log.cut()
        
        self.Hlines = quant.Full_dHS_toMagn(self.SpinPhon, self.SH) 
        if self.LR:
            self.HlinesLR = quantLR.LR_to_Lines_2M1Ph(self.SH, self.R0, self.liMag, self.dim) 
        self.Log.Twrite('quantum Hamiltonian calculated')
        
        list0 = slist.EmptyScatList2M1Ph()
        self.events = slist.alpha_Klist_2M1Ph(self.KGr, self.EGrS, self.EGrPh, list0, self.dim, ek0=self.E0, lam0=self.lamAcu, roles=self.roles)
        self.Nevents = len(self.events)
        self.Log.Twrite(str(self.Nevents) + ' scattering events found')
        
        for sc in self.events:
            slist.findCenter2M1Ph(sc, self.pSH, self.Ephon)
            slist.findDeltaInt2M1Ph(sc, self.pSH, self.Ephon)
        self.Log.Twrite('real momenta + delta-function integrals')
        
        if self.LR:
            kin.initiateMel(self.events, self.Hlines, self.SH, self.pSH, self.Ephon, self.LR, self.HlinesLR, Vuc =self.cellV)
        else:    
            kin.initiateMel(self.events, self.Hlines, self.SH, self.pSH, self.Ephon)
        self.Log.Twrite('Matrix elements calculated')

    def ShowScatterings_2D(self, figX=5, figY=5, Ms=1):
        Lsym = ['o','s','D','H','>']
        Lcol = ['mediumblue','red','olive','darkorange','darkorchid','teal','seagreen']

        def getL(L,i):
            N = len(L)
            i1 = i%N
            return L[i1]

        plt.figure(figsize=(figX,figY))
        for sc in self.events:
            k = sc.k1real
            br = sc.qBra
            plt.scatter([k[0],],[k[1],], c=getL(Lcol, br), marker=getL(Lsym,br), s=Ms )

    def alpha(self, TK):
        r"""
        calculates Gilbert \alpha for the temperature TK [K]
        """
        return kin.alpha2M1Ph(self.events, TK,  self.SH.cell)

    def alphaTerm_Lst(self, TK):
        r"""
        calculates the list of contributions to  
        Gilbert \alpha for the temperature TK [K]
        the order of contributions correspond to self.events
        """
        Tmev = TK*ut2.K_to_mev
        cellV = ut2.cellVolume(self.SH.cell, regime2D=(self.dim == 2))
        TermList = []
        for sc in self.events:
            term = kin.alphaTerm2M1Ph(sc, Tmev, cellV)
            TermList.append(term)
        return TermList

    def phononContrib_2D(self, TK, Nx = None, Ny = None):
        if Nx is None:
            Nx = self.Ngx
        if Ny is None:
            Ny = self.Ngy
        Arr = np.zeros((self.Nbranch,Nx,Ny))
        Terms = self.alphaTerm_Lst(TK)

        kXmin = np.min(self.KGr[...,0])
        kXmax = np.max(self.KGr[...,0])
        kYmin = np.min(self.KGr[...,1])
        kYmax = np.max(self.KGr[...,1])
        dkX = (kXmax - kXmin)/Nx
        dkY = (kYmax - kYmin)/Ny

        axisX = np.linspace(kXmin, kXmax, Nx)
        axisY = np.linspace(kYmin, kYmax, Ny)

        def findij(k):
            i = int((k[0] - kXmin)//dkX)
            if i >=Nx:
                i = Nx-1
            if i<0:
                i=0
            j = int((k[1] - kYmin)//dkY)
            if j >=Ny:
                j = Ny-1
            if j<0:
                j=0
            return i,j
        
        Nscat = len(Terms)
        for isc in range(Nscat):
            val = Terms[isc]
            sc = self.events[isc]
            qBra = sc.qBra
            q = sc.qreal
            i,j = findij(q)
            Arr[qBra,i,j] += val
        return axisX, axisY, Arr


    def contrMapCut_3D(self, T, kzMin=None, kzMax=None):
        contrLST = self.alphaTerm_Lst(T)
        res = np.zeros((self.Ngx,self.Ngy))
        kxm, kxM = np.min(self.KGr[...,0]),  np.max(self.KGr[...,0])
        kym, kyM = np.min(self.KGr[...,1]),  np.max(self.KGr[...,1])
    
        if kzMin is None:
            dkz = (np.max(self.KGr[...,2]) - np.min(self.KGr[...,1]))/self.Ngz
            kzMin = -dkz
            kzMax = dkz
        
        for iev, ev in enumerate(self.events):
            C1 = contrLST[iev]
            kz = ev.k1real[2]
            if (kz>kzMin) and (kz<kzMax):
                kx = ev.k1real[0]
                ky = ev.k1real[1]
                ikx = int( self.Ngx*(kx - kxm)/(kxM-kxm)   )
                iky = int( self.Ngy*(ky - kym)/(kyM-kym)   )
                if (ikx>=0) and (ikx<self.Ngx):
                    if (iky>=0) and (iky<self.Ngy):
                        res[ikx,iky] += C1
        akx = np.linspace(kxm,kxM,self.Ngx)
        aky = np.linspace(kym,kyM,self.Ngy)
        return akx,aky, res
    

    def saveScatterings(self, file1 = None):
        r"""
        saves scattering events to file1 (can be selected automatically)
        """
        if file1 is None:
            file1 = self.Name + '_events.pkl'            
        slist.saveSlist(self.events, file1) 
        self.Log.Twrite2('Scattering events saved:')
        self.Log.write(file1)

    def loadScatterings(self, file1 = None):
        r"""
        loads scattering events from file1 (can be selected automatically)
        """
        if file1 is None:
            file1 = self.Name + '_events.pkl'            
        self.events = slist.loadSlist(file1) 
        self.Log.Twrite2('Scattering events loaded:')
        self.Log.write(file1)
