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
import matplotlib.pyplot as plt
import matplotlib as mpl


from MagnoFallas.Interface import PseudoRad as prad
import MagnoFallas.OldRadtools as rad

def Chiralities(SH, Magn, kv, NB=False):
    r"""
    Calculates the chiralities of all the magnon modes at wavevector kv
    the direction of 1st magnetic sublatice is taken as a reference (z-axis)
    SH - spin Hamiltonian
    Magn - Magon dispersion (if NB=False) or "Numba-compatible Hamiltonian" (is NB=True)
    kv - k-vector
    """
    Nat = len(SH.magnetic_atoms)
    z0 = SH.magnetic_atoms[0].spin_vector
    z0 = z0/np.linalg.norm(z0)
    if NB:
        oms, G,Gi = prad.omega(Magn, kv)
    else:
        oms, G = Magn.omega(kv, return_G=True)
    Nom = 2*Nat
    chiralities = np.zeros(Nat)
    for i in range(Nat):
        chi1 = 0
        norm1 = 0
        for j in range(Nom):
            iat2 = j%Nat
            dirz = np.sign(SH.magnetic_atoms[iat2].spin_vector@z0)
            if j<Nat:
                mul = 1
            else:
                mul = -1
            Psi2 = np.abs(G[i,j]*G[i,j])
            norm1 += Psi2
            chi1 += Psi2 * dirz*mul
        chi1 /= norm1
        chiralities[i] = chi1
    return chiralities





def ChiralBandPlot(SH, Magn, kpoi, xx=None, Xmarks=None, labels=None, cmap=None, NB=False, toPlot=False, size=0.25, saveFile=None, colorbar=False):
    r"""
    Automatic tool to make chiral-resolved plots of the bands
    returns the information required for plot and can make the plot itself with toPlot=True
    
    SH - spin Hamiltonian
    Magn - Magnon-dispesion (if NB=True) of "numba-compatible spin Hamiltonian" (with NB=False)
    kpoi - set of k-points
    xx, Xmarks, labels - standard information for band-plots (output of util2.Kpath)
    cmap - colormap (for plot)
    NB - if numba-compatible methods should be used
    toPlot - if the plot should be made
    size - size of a point for the plot
    saveFile - can save the plot to a file
    colorbar - True will add colorbar to the plot
    """
    
    Nat = len(SH.magnetic_atoms)
    res = []
    
    Nx = len(kpoi)
    oms = []
    vals = []
    cols = []

    if cmap is None:
        cmap = plt.get_cmap('bwr')
    
    for i in range(Nat):
        oms.append([])
        vals.append([])
        cols.append([])
        
    for i in range(Nx):
        kv = kpoi[i]
        if NB:
            oms1 = prad.omega0(Magn, kv) 
        else:
            oms1 = Magn.omega(kv)
        chirs = Chiralities(SH, Magn, kv, NB=NB)
        for ib in range(Nat):
            oms[ib].append(oms1[ib])
            vals[ib].append(chirs[ib])
            cols[ib].append(cmap( (chirs[ib]+1)/2 ))
    oms = np.array(oms)
    vals = np.array(vals)
    cols = np.array(cols)
    resD = {}
    
    if not xx is None:
        resD['x'] = xx
    resD['oms'] = oms
    resD['chi'] = vals
    resD['cols'] = cols
    resD['Nb'] = Nat

    if toPlot:
        fig, ax = plt.subplots(figsize=(4,3))
        for ib in range(Nat):
            ax.scatter(xx, oms[ib], c=cols[ib], s=size)
        ax.set_xlim(0, np.max(xx))
        ax.set_ylim(0,1.05*np.max(oms))
        ax.set_ylabel(r'$\varepsilon$, meV')
        ax.set_xticks(Xmarks, labels)

        if colorbar:
            #fig.colorbar(sm, ax=ax, boundaries=bounds)

            norm = mpl.colors.Normalize(-1,1)
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            fig.colorbar(sm, ax=ax) #, boundaries=bounds)

        if not saveFile is None:
            plt.savefig(saveFile, bbox_inches='tight')
    
    return resD