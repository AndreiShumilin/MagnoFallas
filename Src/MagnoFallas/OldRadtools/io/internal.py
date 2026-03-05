########################################################################################
#  The following part of the code was a version of Rad-Tools project by Andrey Rybakov
# incorporated into MagnoFallas. The license of the original code is provided below
########################################################################################


# RAD-tools - Sandbox (mainly condense matter plotting).
# Copyright (C) 2022-2024  Andrey Rybakov
#
# e-mail: anry@uv.es, web: rad-tools.org
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
Input-output for the files related with this package.
"""

__all__ = [
    "dump_spinham_txt",
    "dump_pickle",
    "load_pickle",
]

import numpy as np

from MagnoFallas.OldRadtools.decorate.array import print_2d_array
from MagnoFallas.OldRadtools.spinham.constants import TXT_FLAGS
from MagnoFallas.OldRadtools.spinham.hamiltonian import SpinHamiltonian

meV_TO_J = 1.602176634e-22


def dump_pickle(object, filename):
    """
    Save any python Object in a binary format.

    Parameters
    ----------
    filename : str
        Name of the file for the Object to be saved in.
        ".pickle" is added automatically.
    """
    import pickle

    with open(f"{filename}.pickle", "wb") as file:
        pickle.dump(object, file)


def load_pickle(filename):
    r"""
    Load any python Object from a binary format.

    Parameters
    ----------
    filename : str
        Name of the .pickle file for the Object to be loaded from.
    """

    import pickle

    with open(filename, "rb") as file:
        object = pickle.load(file)
    return object



def dump_spinham_txt(
    spinham: SpinHamiltonian,
    filename=None,
    anisotropic=True,
    matrix=True,
    dmi=True,
    decimals=4
):
    """
    Save the :py:class:`.SpinHamiltonian` in a human-readable format.

    Parameters
    ----------
    spinham : :py:class:`.SpinHamiltonian`
        Spin Hamiltonian to be saved.
    filename : str, optional
        Name of the file for the Hamiltonian to be saved in.
        If not given, the Hamiltonian will be printed in the console.
    anisotropic : bool, default True
        Whether to output anisotropic exchange.
    matrix : bool, default True
        Whether to output whole matrix exchange.
    dmi : bool, default True
        Whether to output DMI exchange.
    decimals : int, default 4
        Number of decimals to be printed (only for the exchange values).
    """

    main_separator = "=" * 80 + "\n"
    separator = "-" * 80 + "\n"
    spinham_txt = []
    fmt = f"{decimals+4}.{decimals}f"

    spinham_txt.append(main_separator)
    spinham_txt.append(TXT_FLAGS["cell"] + "\n")
    spinham_txt.append(
        print_2d_array(
            spinham.cell,
            borders=False,
            fmt="^.8f",
            print_result=False,
            header_row=["x", "y", "z"],
        )
        + "\n"
    )
    spinham_txt.append(main_separator)
    spinham_txt.append(TXT_FLAGS["atoms"] + "\n")
    header_column = []
    header_row = [
        f"Index Name",
        "a1 (rel)",
        "a2 (rel)",
        "a3 (rel)",
        "x",
        "y",
        "z",
    ]
    atom_data = np.zeros((len(spinham.atoms), 6))
    for a_i, atom in enumerate(spinham.atoms):
        header_column.append(f"{atom.index:<5} {atom.name:<4}")
        atom_data[a_i, :3] = spinham.get_atom_coordinates(atom, relative=True)
        atom_data[a_i, 3:] = spinham.get_atom_coordinates(atom, relative=False)
    spinham_txt.append(
        print_2d_array(
            atom_data,
            borders=False,
            fmt="^.8f",
            print_result=False,
            header_row=header_row,
            header_column=header_column,
        )
        + "\n"
    )
    spinham_txt.append(main_separator)

    write_spins = True
    write_magmoms = True
    for atom in spinham.magnetic_atoms:
        try:
            atom.magmom
        except ValueError:
            write_magmoms = False
        try:
            atom.spin
            atom.spin_vector
        except ValueError:
            write_spins = False
    if write_magmoms:
        spinham_txt.append(TXT_FLAGS["magmoms"] + "\n")
        header_column = []
        header_row = [f"Index Name", "m_x", "m_y", "m_z"]
        magmom_data = np.zeros((len(spinham.magnetic_atoms), 3))
        for a_i, atom in enumerate(spinham.magnetic_atoms):
            header_column.append(f"{atom.index:<5} {atom.name:<4}")
            magmom_data[a_i] = atom.magmom
        spinham_txt.append(
            print_2d_array(
                magmom_data,
                borders=False,
                fmt="^.8f",
                print_result=False,
                header_row=header_row,
                header_column=header_column,
            )
            + "\n"
        )
        spinham_txt.append(main_separator)
    if write_spins:
        spinham_txt.append(TXT_FLAGS["spins"] + "\n")
        header_column = []
        header_row = [f"Index Name", "S", "S_x", "S_y", "S_z"]
        spin_data = np.zeros((len(spinham.magnetic_atoms), 4))
        for a_i, atom in enumerate(spinham.magnetic_atoms):
            header_column.append(f"{atom.index:<5} {atom.name:<4}")
            spin_data[a_i, 0] = atom.spin
            spin_data[a_i, 1:] = atom.spin_vector
        spinham_txt.append(
            print_2d_array(
                spin_data,
                borders=False,
                fmt="^.8f",
                print_result=False,
                header_row=header_row,
                header_column=header_column,
            )
            + "\n"
        )
        spinham_txt.append(main_separator)

    spinham_txt.append(TXT_FLAGS["notation"] + "\n")
    spinham_txt.append(f"{spinham.notation}\n")
    spinham_txt.append(spinham.notation_string + "\n")
    spinham_txt.append(main_separator)


    spinham_txt.append(TXT_FLAGS["spinham"] + "\n")
    spinham_txt.append(
        f"{'Atom1':6} {'Atom2':6} (  i,   j,   k) {'J_iso':^{decimals+4}} {'Distance':^8}\n"
    )
    bonds_data = []
    for atom1, atom2, (i, j, k), J in spinham:
        data_entry = []
        distance = spinham.get_distance(atom1, atom2, (i, j, k))
        atom1 = f"{atom1.name}({atom1.index})"
        atom2 = f"{atom2.name}({atom2.index})"
        data_entry.append(separator)
        data_entry.append(
            f"{atom1:6} {atom2:6} "
            + f"({i:>3}, {j:>3}, {k:>3}) "
            + f"{J.iso:^{decimals+4}.{decimals}f} "
            + f"{distance:^8.4f}\n"
        )
        if matrix:
            data_entry.append(TXT_FLAGS["matrix"] + "\n")
            data_entry.append(
                print_2d_array(
                    J.matrix, fmt=fmt, borders=False, print_result=False, shift=2
                )
                + "\n"
            )
        if anisotropic:
            data_entry.append(TXT_FLAGS["aniso"] + "\n")
            data_entry.append(
                print_2d_array(
                    J.aniso, fmt=fmt, borders=False, print_result=False, shift=2
                )
                + "\n"
            )
        if dmi:
            data_entry.append(TXT_FLAGS["dmi_module"] + "\n")
            data_entry.append(f"  {J.dmi_module:{decimals+4}.{decimals}f}\n")
            data_entry.append(TXT_FLAGS["dmi_relative"] + "\n")
            data_entry.append(f"  {J.rel_dmi:{decimals+4}.{decimals}f}\n")
            data_entry.append(TXT_FLAGS["dmi"] + "\n")
            data_entry.append(
                print_2d_array(
                    J.dmi, fmt=fmt, borders=False, print_result=False, shift=2
                )
                + "\n"
            )
        bonds_data.append(["".join(data_entry), distance])
    bonds_data = sorted(bonds_data, key=lambda x: x[1])
    bonds_data = [x[0] for x in bonds_data]
    spinham_txt.append("".join(bonds_data))
    
    spinham_txt.append(main_separator)
    spinham_txt = "".join(spinham_txt)
    if filename is not None:
        with open(filename, "w", encoding="utf-8") as file:
            file.write(spinham_txt)
    else:
        print(spinham_txt)
