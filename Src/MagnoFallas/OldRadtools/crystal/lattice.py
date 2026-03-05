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
General 3D lattice.
"""

import numpy as np
from scipy.spatial import Voronoi

import MagnoFallas.OldRadtools.crystal.cell as Cell


from MagnoFallas.OldRadtools.crystal.constants import (
    REL_TOL,
)

from MagnoFallas.OldRadtools.geometry import angle, volume

__all__ = ["Lattice"]


class Lattice:
    r"""
    General 3D lattice.

    When created from the cell orientation of the cell is respected,
    however the lattice vectors may be renamed with respect to [1]_.

    Creation may change the angles and the lengths of the cell vectors.
    It preserve the volume, right- or left- handedness, lattice type and variation
    of the cell.

    The lattice vector`s lengths are preserved as a set.

    The angles between the lattice vectors are preserved as a set with possible
    changes of the form: :math:`angle \rightarrow 180 - angle`.

    The returned cell may not be the same as the input one, but it is translationally
    equivalent.

    Lattice can be created in a three alternative ways:



    Parameters
    ----------
    cell : (3, 3) |array_like|_
        Unit cell, rows are vectors, columns are coordinates.
    a1 : (3,) |array_like|_
        First vector of unit cell (cell[0]).
    a2 : (3,) |array_like|_
        SEcond vector of unit cell (cell[1]).
    a3 : (3,) |array_like|_
        Third vector of unit cell (cell[2]).
    a : float, default=1
        Length of the :math:`a_1` vector.
    b : float, default=1
        Length of the :math:`a_2` vector.
    c : float, default=1
        Length of the :math:`a_3` vector.
    alpha : float, default=90
        Angle between vectors :math:`a_2` and :math:`a_3`. In degrees.
    beta : float, default=90
        Angle between vectors :math:`a_1` and :math:`a_3`. In degrees.
    gamma : float, default=90
        Angle between vectors :math:`a_1` and :math:`a_2`. In degrees.
    standardize : bool, default True
        Whether to standardize the cell.
        The consistence of the predefined k paths is not guaranteed in the cell is not unified.

    Attributes
    ----------
    eps_rel : float, default 1e-4
        Relative error for the :ref:`library_lepage` algorithm.

    References
    ----------
    .. [1] Setyawan, W. and Curtarolo, S., 2010.
        High-throughput electronic band structure calculations: Challenges and tools.
        Computational materials science, 49(2), pp.299-312.
    """

    def __init__(self, *args, standardize=True, **kwargs) -> None:
        self._eps_rel = REL_TOL
        self._cell = None
        self._type = None
        if "cell" in kwargs:
            cell = kwargs["cell"]
        elif "a1" in kwargs and "a2" in kwargs and "a3" in kwargs:
            cell = np.array([kwargs["a1"], kwargs["a2"], kwargs["a3"]])
        elif (
            "a" in kwargs
            and "b" in kwargs
            and "c" in kwargs
            and "alpha" in kwargs
            and "beta" in kwargs
            and "gamma" in kwargs
        ):
            cell = Cell.from_params(
                kwargs["a"],
                kwargs["b"],
                kwargs["c"],
                kwargs["alpha"],
                kwargs["beta"],
                kwargs["gamma"],
            )
        elif len(args) == 1:
            cell = np.array(args[0])
        elif len(args) == 3:
            cell = np.array(args)
        elif len(args) == 6:
            a, b, c, alpha, beta, gamma = args
            cell = Cell.from_params(a, b, c, alpha, beta, gamma)
        elif len(args) == 0 and len(kwargs) == 0:
            cell = np.eye(3)
        else:
            raise ValueError(
                "Unable to identify input parameters. "
                + "Supported: cell ((3,3) array_like), "
                + "or a1, a2, a3 (each is an (3,) array_like), "
                + "or a, b, c, alpha, beta, gamma (floats)."
            )

        self.fig = None
        self.ax = None

        self._set_cell(cell, standardize=standardize)

    # Primitive cell parameters
    @property
    def cell(self):
        r"""
        Unit cell of the lattice.

        Notes
        -----
        In order to rotate the cell with an arbitrary rotation matrix :math:`R` use the syntax:

        .. code-block:: python

            rotated_cell = cell @ R.T

        Transpose is required, since the vectors are stored as rows.

        Returns
        -------
        cell : (3, 3) :numpy:`ndarray`
            Unit cell, rows are vectors, columns are coordinates.
        """
        if self._cell is None:
            raise AttributeError(f"Cell is not defined for lattice {self}")
        return np.array(self._cell)

    # For the child`s overriding
    def _set_cell(self, new_cell, standardize=True):
        try:
            new_cell = np.array(new_cell)
        except:
            raise ValueError(f"New cell is not array_like: {new_cell}")
        if new_cell.shape != (3, 3):
            raise ValueError(f"New cell is not 3 x 3 matrix.")
        self._cell = new_cell
        # Reset type
        self._type = None


    @cell.setter
    def cell(self, new_cell):
        self._set_cell(new_cell)

    @property
    def a1(self):
        r"""
        First lattice vector :math:`\vec{a}_1`.

        Returns
        -------
        a1 : (3,) :numpy:`ndarray`
            First lattice vector :math:`\vec{a}_1`.
        """
        return self.cell[0]

    @property
    def a2(self):
        r"""
        Second lattice vector :math:`\vec{a}_2`.

        Returns
        -------
        a2 : (3,) :numpy:`ndarray`
            Second lattice vector :math:`\vec{a}_2`.
        """
        return self.cell[1]

    @property
    def a3(self):
        r"""
        Third lattice vector :math:`\vec{a}_3`.

        Returns
        -------
        a3 : (3,) :numpy:`ndarray`
            Third lattice vector :math:`\vec{a}_3`.
        """
        return self.cell[2]

    @property
    def a(self):
        r"""
        Length of the first lattice vector :math:`\vert\vec{a}_1\vert`.

        Returns
        -------
        a : float
        """

        return np.linalg.norm(self.cell[0])

    @property
    def b(self):
        r"""
        Length of the second lattice vector :math:`\vert\vec{a}_2\vert`.

        Returns
        -------
        b : float
        """

        return np.linalg.norm(self.cell[1])

    @property
    def c(self):
        r"""
        Length of the third lattice vector :math:`\vert\vec{a}_3\vert`.

        Returns
        -------
        c : float
        """

        return np.linalg.norm(self.cell[2])

    @property
    def alpha(self):
        r"""
        Angle between second and third lattice vector.

        Returns
        -------
        angle : float
            In degrees
        """

        return angle(self.a2, self.a3)

    @property
    def beta(self):
        r"""
        Angle between first and third lattice vector.

        Returns
        -------
        angle : float
            In degrees
        """

        return angle(self.a1, self.a3)

    @property
    def gamma(self):
        r"""
        Angle between first and second lattice vector.

        Returns
        -------
        angle : float
            In degrees
        """

        return angle(self.a1, self.a2)

    @property
    def unit_cell_volume(self):
        r"""
        Volume of the unit cell.

        Returns
        -------
        volume : float
            Unit cell volume.
        """

        return volume(self.a1, self.a2, self.a3)

    @property
    def parameters(self):
        r"""
        Return cell parameters.

        :math:`(a, b, c, \alpha, \beta, \gamma)`

        Returns
        -------
        a : float
        b : float
        c : float
        alpha : float
        beta : float
        gamma : float
        """
        return self.a, self.b, self.c, self.alpha, self.beta, self.gamma


    # Reciprocal parameters
    @property
    def reciprocal_cell(self):
        r"""
        Reciprocal cell. Always primitive.

        Returns
        -------
        reciprocal_cell : (3, 3) :numpy:`ndarray`
            Reciprocal cell, rows are vectors, columns are coordinates.
        """

        return Cell.reciprocal(self.cell)

    @property
    def b1(self):
        r"""
        First reciprocal lattice vector.

        .. math::

            \vec{b}_1 = \frac{2\pi}{V}\vec{a}_2\times\vec{a}_3

        where :math:`V = \vec{a}_1\cdot(\vec{a}_2\times\vec{a}_3)`

        Returns
        -------
        b1 : (3,) :numpy:`ndarray`
            First reciprocal lattice vector :math:`\vec{b}_1`.
        """

        return self.reciprocal_cell[0]

    @property
    def b2(self):
        r"""
        Second reciprocal lattice vector.

        .. math::

            \vec{b}_2 = \frac{2\pi}{V}\vec{a}_3\times\vec{a}_1

        where :math:`V = \vec{a}_1\cdot(\vec{a}_2\times\vec{a}_3)`

        Returns
        -------
        b2 : (3,) :numpy:`ndarray`
            Second reciprocal lattice vector :math:`\vec{b}_2`.
        """

        return self.reciprocal_cell[1]

    @property
    def b3(self):
        r"""
        Third reciprocal lattice vector.

        .. math::

            \vec{b}_3 = \frac{2\pi}{V}\vec{a}_1\times\vec{a}_2

        where :math:`V = \vec{a}_1\cdot(\vec{a}_2\times\vec{a}_3)`

        Returns
        -------
        b3 : (3,) :numpy:`ndarray`
            Third reciprocal lattice vector :math:`\vec{b}_3`.
        """

        return self.reciprocal_cell[2]

    @property
    def k_a(self):
        r"""
        Length of the first reciprocal lattice vector :math:`\vert\vec{b}_1\vert`.

        Returns
        -------
        k_a : float
        """

        return np.linalg.norm(self.b1)

    @property
    def k_b(self):
        r"""
        Length of the second reciprocal lattice vector :math:`\vert\vec{b}_2\vert`.

        Returns
        -------
        k_b : float
        """

        return np.linalg.norm(self.b2)

    @property
    def k_c(self):
        r"""
        Length of the third reciprocal lattice vector :math:`\vert\vec{b}_3\vert`.

        Returns
        -------
        k_c : float
        """

        return np.linalg.norm(self.b3)

    @property
    def k_alpha(self):
        r"""
        Angle between second and third reciprocal lattice vector.

        Returns
        -------
        angle : float
            In degrees.
        """

        return angle(self.b2, self.b3)

    @property
    def k_beta(self):
        r"""
        Angle between first and third reciprocal lattice vector.

        Returns
        -------
        angle : float
            In degrees.
        """

        return angle(self.b1, self.b3)

    @property
    def k_gamma(self):
        r"""
        Angle between first and second reciprocal lattice vector.

        Returns
        -------
        angle : float
            In degrees.
        """

        return angle(self.b1, self.b2)

    @property
    def reciprocal_cell_volume(self):
        r"""
        Volume of the reciprocal cell.

        .. math::

            V = \vec{b}_1\cdot(\vec{b}_2\times\vec{b}_3)

        Returns
        -------
        volume : float
            Volume of the reciprocal cell.
        """

        return volume(self.b1, self.b2, self.b3)

    @property
    def reciprocal_parameters(self):
        r"""
        Return reciprocal cell parameters.

        :math:`(a, b, c, \alpha, \beta, \gamma)`

        Returns
        -------
        a : float
        b : float
        c : float
        alpha : float
        beta : float
        gamma : float
        """
        return self.k_a, self.k_b, self.k_c, self.k_alpha, self.k_beta, self.k_gamma

    # Lattice type routines and properties
    @property
    def eps(self):
        r"""
        Epsilon parameter.

        Derived from :py:attr:`.eps_rel` as
        .. math::

            \epsilon = \epsilon_{rel}\cdot V^{\frac{1}{3}}
        """

        return self.eps_rel * abs(self.unit_cell_volume) ** (1 / 3.0)

    @property
    def eps_rel(self):
        r"""
        Relative epsilon

        Returns
        -------
        eps_rel : float
        """

        return self._eps_rel

    @eps_rel.setter
    def eps_rel(self, new_value):
        try:
            self._eps_rel = float(new_value)
        except ValueError:
            raise ValueError("Not a float")
        self._type = None


