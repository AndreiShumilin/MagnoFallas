*********
Magno-Fallas
*********

The code is designed to calculate Gilbert and non-Gilbert damping in magnetic insulators from ab initio–derived spin Hamiltonians. The calculations are made within Boltzmann approximation. Some parts of the calculations use JIT compilation provided my numba library. 

It contains interfaces with TB2J and phonopy to import information on spin Hamiltonian and phonons, respectively. The magnon-phonon interaction can be calculated in two major ways: from the set of TB2J results with displaced atoms and based on the dipole-dipole interaction which is calculated automatically based on the atom coordinates. 

The code contains three main functionalities:
**(I)** calculation of the damping based on **4-magnon** (both magnon-conserving (MC) and magnon-non-conserving (MNC) ) processes from user-provided spin Hamiltonian. No information on the phonons or spin-phonon interaction is requires.
**(II)** calculation of the damping based on **2 magnon- 1 phonon** interaction (with both MC and MNC contributions) from the provided information on magnons, phonons and the interaction.
The physical backgound for the calculations can be found in: *Archive*
**(III)** set of procedures for the manipulation with **spin Hamiltonians** and visualization of magnon information. This includes:
- Transformation of the spin Hamiltonian to another unit cell (e.g. supercell) and band unfolding
- Automatic addition of dipole-dipole interaction including analytically calculated long-range dipole-dipole interaction
- Tools for the visualization of spin Hamiltonian and magnon energies/group velocities (both for selected k-path and as a maps in the whole 2D Brillouin zone)

Finally the code contains utility functions supporting the main functionalities, including:
- interfaces with TB2J and Phonopy codes
- Several models for automatic construction of spin Hamiltonian in representative cases including the miniman model of 2D and 3D ferromagnetic and antiferromagnetic materials and yitrium iron garnet with exchange interactions either provided by user or derived from literature. 

Some parts of the code (mostly the interface with TB2J and non-numba version of spin Hamiltonian diagonalization) are based on an old version of Rad-Tools code by Andrei Rybakov. 

Currently the code should be used as a Python library. The examples section provide explained examples of the representative calculations made with the code. 
