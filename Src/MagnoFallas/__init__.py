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



from . import Boltzmann
from . import Interface
from . import Math
from . import Models
from . import Quantum
from . import SHtools
from . import SpinPhonon
from . import Utils
from . import OldRadtools as rad


from . import SHtools as tools
from . import SpinPhonon
from .Quantum import quantum_util as qut

#-------------------------------------------------------
from .Utils.util2 import Kpath
from .Utils.util2 import groupV
from .Utils.util2 import groupV_P as groupV_NB
from .Utils.util2 import rolesDict as roles
from .Utils.util2 import GetRealPositions
from .Utils.DipoleDipole import AddSRDD
#--------------------------------------------------------

#-------------------------------------------------------
from .Models.ToyFM import ToyModel as ToyModelFM
from .Models.ToyAFM import ToyModelAFM as ToyModelAFM
#-------------------------------------------------------

#-------------------------------------------------------
from .Interface.TB2J import ReadTB2J
#-------------------------------------------------------

#-------------------------------------------------------
from .OldRadtools import SpinHamiltonian
from .OldRadtools import MagnonDispersion
#-------------------------------------------------------

#-------------------------------------------------------
from .Interface.PseudoRad  import make_pSH2 as NBhamiltonian
from .Interface.PseudoRad  import omega0 as NBomega
from .Interface.PseudoRad  import omega as NBomegaFull
#-------------------------------------------------------

from .Boltzmann.process4M.boltzman4M import Boltzman_alpha as damping4M



#------------------------ part dependent on phonopy
try:
    from .Interface.UtilPhonopy import EXTphonopy
    from .Boltzmann.process2M1Ph.boltzmann2M1Ph import Boltzman_alpha2M1Ph as damping2M1Ph
except ImportError:
    raise ImportError('Warning: phonopy import failed, no phonons are possible')


