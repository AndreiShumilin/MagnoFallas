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


from datetime import datetime


__all__ = ['Log']


class Log:
    def __init__(self, fname=None, erase=True):
        self.fname = fname
        if not fname is None:
            if erase:
                open(fname, 'w').close()

    def write(self, stri):
        if self.fname is None:
            print(stri)
        else:
            with open(self.fname, 'a') as f:
                f.write(stri + '\n')

    def Twrite(self, stri):
        dt = datetime.now()
        dtstr = dt.strftime("%m/%d/%Y, %H:%M:%S.%f")
        striAll = dtstr + ' : ' + stri
        if self.fname is None:
            print(striAll)
        else:
            with open(self.fname, 'a') as f:
                f.write(striAll + '\n') 
            
    def cut(self):
        if self.fname is None:
            print('----------')
        else:
            with open(self.fname, 'a') as f:
                f.write('---------- \n') 

    def Twrite2(self, stri):
        dt = datetime.now()
        if self.fname is None:
            print(dt)
            print(stri)
        else:
            with open(self.fname, 'a') as f:
                f.write('\n') 
                f.write(dt.strftime("%m/%d/%Y, %H:%M:%S.%f")+'\n')
                f.write(stri + '\n') 