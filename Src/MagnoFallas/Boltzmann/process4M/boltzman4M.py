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
Main module for 4-magnon process simulations
"""



import numpy as np
import numpy.linalg
import scipy as sp
import matplotlib.pyplot as plt
import copy

import numba as nb
from numba.experimental import jitclass

import inspect 
import pickle


import MagnoFallas.OldRadtools as rad

from MagnoFallas.Interface import PseudoRad as prad

from MagnoFallas.Utils import util2 as ut2
from MagnoFallas.Utils import DipoleDipole as dd
from MagnoFallas.Utils import Logs
from MagnoFallas.Utils import Grids

from MagnoFallas.Boltzmann.process4M import ScatteringList4M as slist

from MagnoFallas.Quantum import quantum_util as qut
from MagnoFallas.Quantum.process4M import quantum4M as quant
from MagnoFallas.Quantum.process4M import quantum4M_IDD as quant4dd
from MagnoFallas.Boltzmann.process4M import kinetics4M as kin


from MagnoFallas.Boltzmann.process4M.ScatteringList4M import rolesMC
from MagnoFallas.Boltzmann.process4M.ScatteringList4M import rolesMNC

class Boltzman_alpha:
    r"""
    Main class responsible for simulation 4-magnon scattering
    """
    def __init__(self, SH, Ngx, Ngy, Ngz=1, rKxM = 1.0, rKyM = 1.0, rKzM = 1.0, lamAcu=None, dim=None,
                 acuRegime=True, acuDeltaCor = True, B=0.0, roles=rolesMC, Ecut=None, NKXac=None, NKYac=None, NKZac=None, Nbmax=None,
                 Name = 'bolt_4M',
                 LRdd=False, includeSRDD=False, R0=12):
        r"""
        SH - Spin Hamiltonian
        Ngx, Ngy, Ngz - size of the K-grid
        rKxM, rKyM, rKzM - from 0 to 1: parameters (for x,y,z directions respectively) responsible for focusing on the central part of k-space. 
                           The grid along b1 direction would be between vectors [-0.5*rKxM*b1, 0.5*rKxM*b1]
        lamAcu - number of acoustic magnon band (can be identified automatically)                  
        dim - system dymension
        acuRegime - shows that "acoustic-type" K-grid should be constructed for k-vectors k2, k3
        acuDeltaCor - corrects the "acoustic grid" with respect to k=0 magnon energy (should always be True)
        B - external magnetic field [in T]
        roles - rolesMC/rolesMNC: controls wether we consider magnon-conserving or magnon-non-conserving processes
        dim - system dymension (None leads to identification based on Ngz)
        Ecut - energy cutoff
        NKXac, NKYae, NKZac - sizes of acoustic grids (relevant for acuRegime=True)
        Nbmax - number of magnon bands to consider
        Name - name of the process
        LRdd - wether to include long-range dipole-dipole interaction
        includeSRDD - wether to automatically add short-range dipole-dipole interaction to spin Hamiltonian
        R0 - cutoff radius for long/short-range dipole-dipole interaction
        """
        
        if dim is None:
            if Ngz<=1:
                self.dim = 2
            else:
                self.dim = 3
        else:
            self.dim = dim 
        
        self.Ngx = Ngx
        self.Ngy = Ngy
        self.Ngz = Ngz
        self.B = B
        self.rKxM = rKxM   #### relative maximum momentum along b1 (1 corresponds to whole range [-b1/2,b1/2])
        self.rKyM = rKyM
        self.rKzM = rKzM
        self.roles = roles
        
        self.Name = Name
        self.logfile = Name + '.log'
        self.Log = Logs.Log(self.logfile)

        self.Log.Twrite2('4-Magnon calcualtions')
        self.Log.write('process name = ' + self.Name)
        self.Log.write('roles = ' + str(self.roles))  
        self.Log.cut()


        SHzero = SH

        self.LR = LRdd
        self.SR = includeSRDD
        if (self.LR or self.SR):
            self.R0 = R0
            self.NddX, self.NddY, self.NddZ = dd.calculateNdd(SHzero, self.R0, dim=self.dim)
        if self.SR:
            self.SH = dd.AddDD(SHzero, NxMax=self.NddX, NyMax=self.NddY, NzMax=self.NddZ, Rmax = self.R0)
        else:
            self.SH = copy.deepcopy(SHzero)
      
        if self.LR:
            self.pSH = prad.make_pSH2(self.SH, self.B, LR=self.LR, dim=self.dim, R0=self.R0)  
                            ###Note prad pSH should be made from initial (non-ferromagnetized) spin Hamiltonian
                            ### to correctly track the effect of magnetic field on sublattices
        else:
            self.pSH = prad.make_pSH2(self.SH, self.B, LR=False, dim=self.dim, R0=0.0)  
        
        self.Nb =  len(self.SH.magnetic_atoms)
        if Nbmax is None:
            self.Nbmax = self.Nb
        else:
            self.Nbmax = Nbmax

        self.Scell = np.linalg.norm( np.cross(SH.a1,SH.a2)  )
        self.Vcell3D = np.abs( np.dot(SH.a3, np.cross(SH.a1,SH.a2)) )
        if self.dim == 3:
            self.Vcell = self.Vcell3D
        else:
            self.Vcell = self.Scell


        if lamAcu is None:
            self.lamAcu = self.Nb-1
        else:
            self.lamAcu = lamAcu

        if self.dim==2:
            self.KGr, self.gGr = Grids.KGrid(Ngx, Ngy, 1,  self.SH.cell,  rKXmax=self.rKxM, rKYmax=self.rKyM, regime2D=True)
        else:
            self.KGr, self.gGr = Grids.KGrid(Ngx, Ngy, Ngz,  self.SH.cell,  rKXmax=self.rKxM, rKYmax=self.rKyM, rKZmax=self.rKzM, regime2D=False)
            

        gridStr = str(self.Ngx) + 'x' + str(self.Ngy) + 'x'  + str(self.Ngz) + '  K-grid created'
        self.Log.Twrite(gridStr)

        #------------ acoustic grid parameters --------------
        self.acuReg = acuRegime
        self.acuDeltaCor = acuDeltaCor
        
        NKac0 = (0,32,8)
        if NKXac is None:
            self.NKXac = NKac0[self.dim-1]
        else:
            self.NKXac = NKXac

        if NKYac is None:
            self.NKYac = NKac0[self.dim-1]
        else:
            self.NKYac = NKYac

        if NKZac is None:
            self.NKZac = NKac0[self.dim-1]
        else:
            self.NKZac = NKZac
        #---------------------------------------
 
        self.EGr = Grids.Egrid_prad(self.pSH, self.KGr)
        kG = np.zeros(3)
        self.E0 = prad.omega(self.pSH, kG)[0][self.lamAcu]    #####self.Magn.omega(kG)[self.lamAcu]   
        self.Ecut = Ecut
        self.Log.Twrite('EM-grid created')

        self.pos =  np.array([ut2.realposition(self.SH, at) for at in self.SH.magnetic_atoms])   
                                                            #####  We assume that initial positions are always in relative coordinates
        self.SHferro = qut.make_FerroHam(self.SH)
        self.SHlines = quant.PermutedMagnon4_Ham(self.SH)
        
        if self.LR:
            self.LRlines = quant4dd.Create_IDD_lines(self.SH, R0, dim=3)
        self.Log.Twrite('Quantum Hamiltonian created')

        self.LogCountMax = 15

    
    def GenScatSet(self):
        r"""
        generates the scattering events based on the energy-conservation law
        (with the precission of grid cell cize)
        """
        scD = {}
        scDk = {}
        for ib_0 in range(self.Nbmax):
            ib = self.Nb - 1 - ib_0
            if self.dim==2:
                D1, Dk =  slist.Sets4MLists2D(self.KGr, self.SH, self.pSH, self.lamAcu,
                                              ib, self.E0, acuRegime=self.acuReg, acuDelCor = self.acuDeltaCor, NKXac=self.NKXac, 
                                              NKYac=self.NKYac, Ecut=self.Ecut,  roles = self.roles, Nbmax=self.Nbmax)
            else:
                D1, Dk =  slist.Sets4MLists3D(self.KGr, self.SH, self.pSH, self.lamAcu,
                                              ib, self.E0, acuRegime=self.acuReg, acuDelCor = self.acuDeltaCor, NKXac=self.NKXac, 
                                              NKYac=self.NKYac, NKZac=self.NKZac, Ecut=self.Ecut,  roles = self.roles, Nbmax=self.Nbmax,
                                              Log = self.Log)
            scD.update(D1)
            scDk.update(Dk)
        self.Sc_listSets = scD
        self.Sc_k_dict = scDk

        Nlist = len(self.Sc_listSets)
        self.Nevents = 0
        for k,v in self.Sc_listSets.items():
            self.Nevents += len(v)

        self.Log.Twrite2('Scattering events found')
        stri1 = str(Nlist) + ' lists;  ' + str(self.Nevents) + ' events'
        self.Log.write(stri1)
        self.Log.cut()
        
    def InitScater(self):
        r"""
        For each scattering event identifies the (single) exact value of momentum fulfilling the energy conservation
        + calculates the matrix element and k-integral of the energy delta-function
        """
        readyScat = 0
        LogCount = 0
        #LogCountMax = 10
        for k,lst in self.Sc_listSets.items():
            Nlst1 = len(lst)
            for sc in lst:
                if self.dim==2:
                    slist.findCenter2D(sc, self.pSH)
                else:
                    slist.findCenter3D(sc, self.pSH)
                slist.findDeltaInt(sc, self.pSH)
            if self.LR:
                kin.InitiateMel_List(lst, self.pSH, self.SHlines, self.pos, LR=self.LR, LRlines=self.LRlines)
            else:
                kin.InitiateMel_List(lst, self.pSH, self.SHlines, self.pos, LR=False)
            readyScat += Nlst1
            LogCount += 1
            if LogCount >= self.LogCountMax:
                stri1 = str(readyScat) + ' events intialized'
                self.Log.Twrite(stri1) 
                LogCount = 0
        self.Log.Twrite2('Scattering events initialized')

    def alpha(self,TK, reg=1):
        r"""
        Calculates the Gilbert damping at temperature TK [in K]
        reg controls the analitical formula for Gilbert damping
        """
        alp = 0.0
        for k,lst in self.Sc_listSets.items():
            alp += kin.alphaList(lst, TK, self.gGr,  regime=reg)
        if self.dim==2:
            alp *= self.Scell/(4*np.pi*np.pi)     ##### related to the integration over k2
        else:
            alp *= self.Vcell/(8*np.pi*np.pi*np.pi) 
        return alp


    def alphaContrib(self, TK, lam=None, reg=1):
        r"""
        Calculates the contribution to the Gilbert damping from different points of the main K-grid
        TK - temperature [in K]
        reg controls the analitical formula for Gilbert damping
        """
        if self.dim==2:
            return self.alphaContrib2D(TK, lam=lam, reg=reg)
        else:
            return self.alphaContrib3D(TK, lam=lam, reg=reg)

    def alphaContrib2D(self, TK, lam=None, reg=1):
        if lam is None:
            lam = self.lamAcu
        
        Nx = self.Ngx
        Ny = self.Ngy
        Arr = np.zeros((Nx,Ny))

        axis=(np.linspace(-self.rKxM*0.5, self.rKxM*0.5, self.Ngx ),
              np.linspace(-self.rKyM*0.5, self.rKyM*0.5, self.Ngy )
             )

        #alpa_full = self.alpha(TK, reg=reg)
        gK1 = self.gGr
        
        for key,lst in self.Sc_listSets.items():
            ix,iy,l1 = key
            if l1 == lam:
                alp1 = kin.alphaList(lst, TK, self.gGr,  regime=1)
                alp1 *= self.Scell/(4*np.pi*np.pi) 
                Arr[ix,iy] += alp1*gK1
        Arr /= np.sum(Arr)
        return axis, Arr

    def alphaContrib3D(self, TK, lam=None, reg=1):
        if lam is None:
            lam = self.lamAcu
        
        Nx = self.Ngx
        Ny = self.Ngy
        Nz = self.Ngz

        axis=(np.linspace(-self.rKxM*0.5, self.rKxM*0.5, self.Ngx ),
              np.linspace(-self.rKyM*0.5, self.rKyM*0.5, self.Ngy ),
              np.linspace(-self.rKzM*0.5, self.rKzM*0.5, self.Ngz )
             )
        
        Arr = np.zeros((Nx,Ny, Nz))
        

        #alpa_full = self.alpha(TK, reg=reg)
        gK1 = self.gGr

        alpa_full = 0.0
        for key,lst in self.Sc_listSets.items():
            ix,iy,iz, il = key
            if il == lam:
                alp1 = kin.alphaList(lst, TK, self.gGr,  regime=1)
                #alp1 *= self.Vcell/(8*np.pi*np.pi*np.pi) 
                Arr[ix,iy,iz] += alp1  #*gK1
                alpa_full += alp1
        return axis, Arr/alpa_full

    def saveScatterings(self, file1 = None, filek = None):
        r"""
        Saves the scattering events to files
        file1, filek - names of the files
        """
        if file1 is None:
            file1 = self.Name + '_scatterings.pkl'   
        if filek is None:
            filek = self.Name + '_scatter_K.pkl'   
        slist.saveSlist(self.Sc_listSets, file1) 
        with open(filek, 'wb') as fp:
            pickle.dump(self.Sc_k_dict, fp)
        self.Log.Twrite2('Scattering events saved')
        self.Log.write(file1)
        self.Log.write(filek)

    def loadScatterings(self, file1 = None, filek = None):
        r"""
        Loads the scattering events from files
        file1, filek - names of the files
        """
        if file1 is None:
            file1 = self.Name + '_scatterings.pkl'   
        if filek is None:
            filek = self.Name + '_scatter_K.pkl'   
        self.Sc_listSets = slist.loadSlist(file1) 
        with (open(filek, "rb")) as openfile:
            self.Sc_k_dict = pickle.load(openfile)  
        self.Log.Twrite2('Scattering events loaded')
        self.Log.write(file1)
        self.Log.write(filek)
