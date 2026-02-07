*********
![Valencia MagnoFallas]([images/Fallas2.png](https://github.com/AndreiShumilin/MagnoFallas/blob/main/Fallas-3.png))
*********

The code is designed to calculate Gilbert and non-Gilbert damping in magnetic insulators from ab initio–derived spin Hamiltonians. The calculations are made within Boltzmann approximation. Some parts of the calculations use JIT compilation provided my numba library. 

It contains interfaces with TB2J and phonopy to import information on spin Hamiltonian and phonons, respectively. The magnon-phonon interaction can be calculated in two major ways: from the set of TB2J results with displaced atoms and based on the dipole-dipole interaction which is calculated automatically based on the atom coordinates. 

The code contains three main functionalities:

**(I)** calculation of the damping based on **4-magnon** (both magnon-conserving (MC) and magnon-non-conserving (MNC) ) processes from user-provided spin Hamiltonian. No information on the phonons or spin-phonon interaction is requires.

**(II)** calculation of the damping based on **2 magnon- 1 phonon** interaction (with both MC and MNC contributions) from the provided information on magnons, phonons and the interaction.

**(III)** set of procedures for the manipulation with **spin Hamiltonians** and visualization of magnon information. This includes:

- Transformation of the spin Hamiltonian to another unit cell (e.g. supercell) and band unfolding
- Automatic addition of dipole-dipole interaction including analytically calculated long-range dipole-dipole interaction
- Tools for the visualization of spin Hamiltonian and magnon energies/group velocities (both for selected k-path and as a maps in the whole 2D Brillouin zone)

Finally the code contains utility functions supporting the main functionalities, including:
- interfaces with TB2J and Phonopy codes
- Several models for automatic construction of spin Hamiltonian in representative cases including the miniman model of 2D and 3D ferromagnetic and antiferromagnetic materials and yitrium iron garnet with exchange interactions either provided by user or derived from literature. 

Some parts of the code (mostly the interface with TB2J and non-numba version of spin Hamiltonian diagonalization) are based on an old version of Rad-Tools code by Andrei Rybakov. 

Currently the code should be used as a Python library. The examples section provide explained examples of the representative calculations made with the code. 



### Physical background


The pysical background can be found in *Archive*


### Dependencies

The following python libraries are required:

-numpy <br>
-scipy <br>
-numba <br>
-matplotlib <br>
-termcolor <br>
-tqdm

The phonopy API is required for the interaction with phonons. It can be installed with

pip install phonopy

Please visit: https://phonopy.github.io/phonopy/install.html  for further information. 


### Installation


First, download the code from github. 

We presume that the following steps are made from the project main folder (where the pyproject.toml is located)

*We recomend starting from a clean conda environment generated with the provided .yml file* <br>
conda env create -f environment.yml <br>
conda activate MagnoFallas

*While using some features without phonopy API is possible, we recomend installing it* <br>
pip install phonopy

*MagnoFallas can be installed from the project folder with pip* <br>
pip install .

*We recomend using jupyter notebook* <br>
pip install jupyter <br>
*To use jupyter with the created environment, the kernel should be added* <br>
python -m ipykernel install --user --name MagnoFallas --display-name "Python (MagnoFallas)" <br>
*To run jupyter notebook from the environment* <br>
jupyter notebook<br>
*The Python (MagnoFallas) kernel can be than selected within jupyter notebook* <br>



### Getting started


We recomend to start by running the provided tutorials and reading comments.

At the current stage the project does not contain a detailed manual, so, please contact the author directly for the guidance. 


*********
## Contacts

Andrei Shumilin  <br>
email: andrei.shumilin@uv.es

We also recomend visiting Valencia in March for further inspiration




