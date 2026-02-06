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

import phonopy


__all__ = ['conv_phonopy_meV','conv_meV_THz','EXTphonopy']


from MagnoFallas.SpinPhonon import SPhUtil as sphut

conv_phonopy_meV = 64.65415129579071 ### from internal phonopy units, different from THz
conv_meV_THz = 0.2417990504024


class EXTphonopy:
    r"""
    class that extends phonopy by addind useful features like andusting negative frequencies of acoustic phonons near Gamma-point
    
    It is initialized from a phonopy object phonon, which is available as .phon0

    It is important for the calculations that no negative phonon are present even near Gamma-point. 
    """
    def __init__(self, phonon, Adj_ecut=0.5, name_map=None):
        self.phon0 = phonon
        self.Nat = len(phonon.primitive.symbols)
        self.Nband = 3*len(phonon.primitive.symbols)
        self.c = np.zeros((self.Nband, 3))
        self.Adj_ecut = Adj_ecut      #### maximum energy for the adjustments of acoustic phonons
        self.NAcc = 3       ## number of acoustic bands
        self.cell = self.phon0.primitive.cell
        
        self.a1 = self.cell[0,...]
        self.a2 = self.cell[1,...]
        self.a3 = self.cell[2,...]
        
        self.b1 = np.cross(self.a2,self.a3)/ np.dot(self.a1, np.cross(self.a2,self.a3))
        self.b2 = np.cross(self.a3,self.a1)/ np.dot(self.a2, np.cross(self.a3,self.a1))
        self.b3 = np.cross(self.a1,self.a2)/ np.dot(self.a3, np.cross(self.a1,self.a2))

        self.bmatr = np.array([self.b1,self.b2,self.b3])

        self.masses = self.phon0.primitive.masses
        self.positions = self.phon0.primitive.positions

        self.name_map = name_map
    
    

    def InternalQ(self, qreal):
        q1 = qreal / (2*np.pi)
        qInt = np.linalg.solve(self.bmatr, q1)
        return qInt
    

    def getC(self, c3, q, aqmin=1e-6):
        r""" 
        estimate sound velocity taking into account the values over 3 reciprocal axes (array c3) and the direction of q
        """
        aq = np.sqrt(np.abs(np.sum( q*q )))
        if aq<aqmin:
            eq = np.array((1,0,0))
        else:
            eq = q/aq
        cvec = eq*c3
        ca = np.sqrt(np.sum(cvec*cvec))
        return ca

    def estimateC(self, aQref, isreal=False):
        r"""
        estimates the sound velocities. Requires reference absolute value of Q, corresponding to good linear dispersion in all directions
        is real shows that aQref0 is in 1/A instead of phonopy notations
        """

        ee = np.eye(3)
        for ib in range(self.NAcc):
            for ie in range(3):
                q0 = aQref*ee[ie]
                if isreal:
                    q = self.InternalQ(q0)
                    aQref2 = np.linalg.norm(q)
                else:
                    q = q0
                    aQref2 = aQref
                en = self.energy(q, ib, cmin=0.0, isreal=False )
                self.c[ib,ie] = en/aQref2
        
    def energy(self, qinp, qBra, cmin=None, isreal=True ):
        r"""
        calculate phonon energy with wavevector qinp and branch qBra
        cmin allows to introduce minimal possible sound velocity to avoid negative phonons
        isreal controls the representation of wavevector
            True: real coordinates, qinp in A^{-1}
            False: releative coordinates, qinp in reciprocal vecotrs
        """
        if isreal:
            q = self.InternalQ(qinp)
        else:
            q = qinp
            
        ### Note: q should be in phonopy notations
        ### i.e., defined with respect to lattice vectors realQ = q[0]*b1 + q[1]*b2 + q[2]*b3
        q1 = (q +0.5) % 1 - 0.5
        if cmin is None:
            cmin = self.getC(self.c[qBra], q1)

        #print('ququq : ', q)
        M = self.phon0.get_dynamical_matrix_at_q(q1)
        eig0, Wf = np.linalg.eigh(M)
        Aoms0 = np.sqrt(np.abs(eig0))
        oms = Aoms0 * np.sign(eig0)
        emev = oms*conv_phonopy_meV
        e1 = emev[qBra]
    
        aq = np.sqrt(np.abs(np.sum( q1*q1 )))
    
        if ((e1<aq*cmin) and (e1<self.Adj_ecut)):
            e1 = cmin*aq
            
        return e1  

    def energies_all(self, qinp, cmins=None, isreal=True ):
        r"""
        calculate all phonon energies with wavevector qinp
        cmins allows to introduce minimal possible sound velocity to avoid negative phonons
        isreal controls the representation of wavevector
            True: real coordinates, qinp in A^{-1}
            False: releative coordinates, qinp in reciprocal vecotrs
        """
        if isreal:
            q = self.InternalQ(qinp)
        else:
            q = qinp
            
        ### Note: q should be in phonopy notations
        ### i.e., defined with respect to lattice vectors realQ = q[0]*b1 + q[1]*b2 + q[2]*b3
        q1 = (q +0.5) % 1 - 0.5
        if cmins is None:
            cmins = np.zeros(3)
            for qBra in range(3):
                cmins[qBra] = self.getC(self.c[qBra], q1)

        M = self.phon0.get_dynamical_matrix_at_q(q1)
        eig0, Wf = np.linalg.eigh(M)
        Aoms0 = np.sqrt(np.abs(eig0))
        oms = Aoms0 * np.sign(eig0)
        emev = oms*conv_phonopy_meV

        aq = np.sqrt(np.abs(np.sum( q1*q1 )))
        for qBra in range(3):
            e1 = emev[qBra]
            cmin = cmins[qBra]
            if ((e1<aq*cmin) and (e1<self.Adj_ecut)):
                e1 = cmin*aq
            emev[qBra] = e1
        return emev

    
    def eVector(self, qinp, qBra, isreal=True):
        r"""
        calculate phonon self-vecotr
        qinp:  wavevector 
        qBra:  branch
        cmin allows to introduce minimal possible sound velocity to avoid negative phonons
        isreal controls the representation of wavevector
            True: real coordinates, qinp in A^{-1}
            False: releative coordinates, qinp in reciprocal vecotrs
        """
        if isreal:     ### Note: q should be in phonopy notations
            q = self.InternalQ(qinp)
        else:
            q = qinp
        q1 = (q +0.5) % 1 - 0.5
        
        M = self.phon0.get_dynamical_matrix_at_q(q1)
        eig0, Wf = np.linalg.eigh(M)
        Psi = np.transpose(Wf).reshape((self.Nband,self.Nat,3))
        return Psi[qBra]


    def fullSolve(self, qinp, isreal=True):
        if isreal:     ### Note: q should be in phonopy notations
            q = self.InternalQ(qinp)
        else:
            q = qinp
        q1 = (q +0.5) % 1 - 0.5

        M = self.phon0.get_dynamical_matrix_at_q(q1)
        eig0, Wf = np.linalg.eigh(M)
        Aoms0 = np.sqrt(np.abs(eig0))
        oms = Aoms0 * np.sign(eig0)
        emev = oms*conv_phonopy_meV
        
        Psi = np.transpose(Wf).reshape((self.Nband,self.Nat,3))
        return emev, Psi

    def relate(self, SH, dim=3, Nmap=None):
        r"""
        relates the atom enumeration in phonopy with the enumeration in Spin Hamiltonian
        should be performed before any calculations of spin-phonon interaciton
        dim - Hamiltonian dimension
        Nmap - map of the element names
        """
        if Nmap is None:
            Nmap = self.name_map
        return sphut.relate_TB2J_Phonopy(SH, self.phon0, dim, name_map=Nmap)
        
       
