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
Module for statistical analysis of magnon-phonon scattering
"""


import h5py
import numpy as np
import numpy.linalg
import scipy as sp
import matplotlib.pyplot as plt

import numba as nb
from numba.experimental import jitclass

import phonopy
import inspect 


import MagnoFallas.OldRadtools as rad

from MagnoFallas.Interface import PseudoRad as prad
from MagnoFallas.Interface import UtilPhonopy as utph
from MagnoFallas.SpinPhonon import SPhUtil as sphut
from MagnoFallas.SpinPhonon import StrategyDipDip as strdi
from MagnoFallas.SHtools import tools as tools
from MagnoFallas.Utils import util2 as ut2
from MagnoFallas.Utils import DipoleDipole as dd
from MagnoFallas.Models import YIG as yig
from MagnoFallas.Quantum import quantum_util as qut

from MagnoFallas.Boltzmann.process2M1Ph import ScatteringList2M1Ph as slist
from MagnoFallas.Boltzmann.process2M1Ph import kinetics2M1Ph as kin



class EventAnalizer:
    r"""
    Calss for analysis of the contribution of different scattering events to
    magnon-phonon Gilbert damping
    """
    def __init__(self, events, qBraMax, T=300, cellV = 1.0, alp=None, finished=True):
        r"""
        events - list of scattering events
        qBraMax - maximum phonon branch to consider
        T - temperature [in K]
        cellV - unit cell volume
        alp - value of Gilbert dampoing 
        finished - wether the main calculation is finished (i.e., events are properly initialized)
        """
        self.events = events
        self.qBraMax = qBraMax
        self.T = T
        self.Tmev = self.T * ut2.K_to_mev
        self.cellV = cellV
        self.Dat = []
        self.alpDat = []
        self.finished = finished
        for i in range(qBraMax):
            self.Dat.append([])
            self.alpDat.append([])
        for ev in self.events:
            br1 = ev.qBra
            if br1<self.qBraMax:
                self.Dat[br1].append(ev)
                if self.finished:
                    contr = kin.alphaTerm2M1Ph(ev, self.Tmev, self.cellV)
                else:
                    contr = 1.0
                self.alpDat[br1].append(contr)
        for i in range(self.qBraMax):
            self.alpDat[i] = np.array(self.alpDat[i])
        if alp is None:
            alp1 = 0
            for i in range(self.qBraMax):
                alp1 += np.sum(self.alpDat[i])
            self.alp = alp1
        else:
            self.alp = alp

    def BranchCont(self):
        r"""
        contributions of the different phonon branches
        """
        res = np.zeros(self.qBraMax)
        for i in range(self.qBraMax):
            res[i] = np.sum(self.alpDat[i])/self.alp
        return(res)

    def printBranchK(self):
        r"""
        print information on k-vectors important for different branches
        """
        for i in range(self.qBraMax):
            kk = np.array([np.linalg.norm(ev.k1real) for ev in self.Dat[i]])
            kmin = np.min(kk)
            kmax = np.max(kk)
            averk = np.sum(kk*self.alpDat[i])/np.sum(self.alpDat[i])
            print(i, ' : ', kmin, averk, kmax)

    def printKcorners(self):
        r"""
        print information on k-vectors important for different branches
        can be used before event initialization
        """
        for i in range(self.qBraMax):
            kk = np.array([ev.k1corners for ev in self.Dat[i]])
            kminP = np.min(kk)
            kmaxP = np.max(kk)

            lkAmax = np.zeros(len(self.Dat[i]))
            lkAmin = np.zeros(len(self.Dat[i]))
            for i2 in range( len(self.Dat[i]) ):
                kco = kk[i2]
                kkk1 = np.array([np.linalg.norm(k) for k in kco])
                lkAmax[i2]  = np.max(kkk1)
                lkAmin[i2]  = np.min(kkk1)

            kminA = np.min(lkAmin)
            kmaxA = np.max(lkAmax)
            
            print(i, len(self.Dat[i]),  ' : ', kminP, kmaxP, ' : ', kminA, kmaxA)


