r"""
This tutorial is focused on the magnon-phonon magnon-damping mechanism
We calculate the damping in a CrSBr monolayer basing on dipole-dipole interaction
between magnons and phonons
"""

import numpy as np
import matplotlib.pyplot as plt

# we import phonopy API directly to control the 
# importing the phonopy results
import phonopy

import MagnoFallas as mfal


## the following line shoud contain a valid path to the tutorial data
TutorialData = '..//TutorialData'

file1 = TutorialData + '//CrSBr//exchange.out'
SH0 = mfal.ReadTB2J(file1, 1.5)
# 1.5 = 3/2 - Cr spin in CrSBr



# The available data for CrSBr includes one inconsistancy in the notation: the y-axis for coordinates
# is considered as z-axis for spin
# the following procedure resolve such an inconsistency
def correctCrSBt(SH0):
    sv0 = np.array((0,1.5,0))
    for at in SH0.magnetic_atoms:
        at.spin_vector = sv0
    
    SHi = mfal.qut.make_FerroHam(SH0)
    for at in SHi.magnetic_atoms:
        at.spin_vector = sv0
    
    return SHi

SH = correctCrSBt(SH0)


# Here we use phonopy API to import phonon data
PHdir = TutorialData + '//CrSBr//'
phonon = phonopy.load(PHdir+"phonopy_disp.yaml", force_sets_filename=PHdir+"FORCE_SETS")


## we create special object to inclode phonons in our code
Ephon = mfal.EXTphonopy(phonon, Adj_ecut=0.1)
Ephon.estimateC(0.0001)
## Adj_ecut and estimateC are tailored to estimate sound velocity and use it to avoid negative phonon frequencies at very small 
## wavevectors


## parameters of the calculation

B0 = 0.0  #external magnetic field

Ng = 64   # K-grid size in one dimension
rk = 0.45  # grid cutoff - allows to focus only on the centrall part of the grid

R0 = 12  #cutoff distance for SRDD/LRDD

MaxBranch = 5   ## maximal number of phonon branch to be included
                ## high optical branches usually have too low occupation numbers to 
                ## play a significant role



NameMC = 'Boltzman-2M1Ph-MC'
dampingMC = mfal.damping2M1Ph(SH, Ephon, Ng, Ng, 1, dim=2, rKxM = rk, rKyM = rk, LRdd=True, includeSRDD=True, R0=R0,B=B0,
                                   MaxBranch = MaxBranch, Name=NameMC, roles = mfal.roles['PhononMC'])

# to calculate the magnon damping we create a special object mfal.damping2M1Ph()
# it requires at least the spin Hamiltonian (SH) and information on the phonon (Ephon)
# and information on the K-grid (Ng, rk and dim - grid dimension)
#
# Here we calculate magnon-phonon coupling only from the dipole-dipole interaction. It can be done inside the damping2M1Ph object
# Otherwise, external information on the spin-phonon coupling should be provided
#
# The parameters required to correctly treat the dipole-dipole interaction:
# includeSRDD - shows that short-ranged dipole-dipole interaction shoul be added both to spin Hamiltonian and magnon-phonon interaction
# LRdd - shows that long-range part of dipole-dipole interaction should be taken into account
# R0 - the cutoff distance for SRDD/LRDD
#
# Other parameters:
# roles = mfal.roles['PhononMC'] --- shows that we study the magnon-conserving processes
# B - external magnetic field
# Name - the name of the process (used in Logs and save/load system for scattering events)



# most important (and sometimes most time-consuming) part of the calculation: initialization of the scattering events
dampingMC.initialize()

dampingMC.saveScatterings()
# the scattering events can be saved for a future use
# dampingMC.loadScatterings()  - the procedure to load them and avoid "initialize()"


## here we select temperatures for the calculation of damping and do the calculation itself
## with the scattering events initialized, this part is usually fast
tt = np.linspace(0.1, 50, 100)
alps = np.array([dampingMC.alpha(t) for t in tt])


## to study the contribution of Non-Magnon-Conserving processes
## we create another instance of damping2M1Ph object
NameNMC = 'Boltzman-2M1Ph-NMC'
dampingNMC = mfal.damping2M1Ph(SH, Ephon, Ng, Ng, 1, dim=2, rKxM = rk, rKyM = rk, LRdd=True, includeSRDD=True, R0=R0,B=B0,
                                   MaxBranch = MaxBranch, Name=NameNMC, roles = mfal.roles['PhononNMC'])

dampingNMC.initialize()


alpsNMC = np.array([dampingNMC.alpha(t) for t in tt])


plt.figure(figsize=(4.6,3.7))
plt.plot(tt,alps,'b-', label='MC')
plt.plot(tt,alpsNMC,'m-', label='NMC')

plt.legend(frameon=False)

plt.xlabel('T, K')
plt.ylabel(r'Gilbert $\alpha$')
plt.savefig('CrSBr-phonon-damping.png', bbox_inches='tight')

plt.show()


axisX, axisY, contrMC = dampingMC.phononContrib_2D(50)
print(contrMC.shape)

### to control the calculations (and, in particular, K-grid) it is useful to check the contribution of the different
## k-vecotrs. In 2D (when it is natural to plot the contributions as maps) it can be done with the method
##  .phononContrib_2D(T)
## where T is temperature in K
## the result is axises (useful for plotting) and contrMC array [MaxBranch x Ng x Ng]
## it contains a separate information for each phonon branch


### Here we calculate the total contributions of all the branches and show the result
ScontMC = np.sum(contrMC, axis=0)

plt.figure(figsize=(4.0,3.5))
plt.pcolormesh(axisX, axisY, np.transpose(np.log10(ScontMC/np.max(ScontMC))), vmin=-5, cmap='Blues' )
plt.xlabel(r'$q_x$, $\AA^{-1}$')
plt.ylabel(r'$q_y$, $\AA^{-1}$')
plt.colorbar()

plt.savefig('phonon-contribution-MC.png', bbox_inches='tight')

plt.show()


##Below we do a similar calculations for NMC calculations

axisX2, axisY2, contrNMC = dampingNMC.phononContrib_2D(50)


ScontNMC = np.sum(contrNMC, axis=0)

plt.figure(figsize=(4.0,3.5))
plt.pcolormesh(axisX2, axisY2, np.transpose(np.log10(ScontNMC/np.max(ScontNMC))), vmin=-5, cmap='Reds' )
plt.xlabel(r'$q_x$, $\AA^{-1}$')
plt.ylabel(r'$q_y$, $\AA^{-1}$')
plt.colorbar()

plt.savefig('phonon-contribution-NMC.png', bbox_inches='tight')

plt.show()
