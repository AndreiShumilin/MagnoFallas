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
import scipy as sp
import numba as nb


AreaMin=1e-15

@nb.njit
def OtherEl(pair,el):
    ### find the secon element in a pair
    ### inegers assumed
    if pair[0]==el:
        return pair[1]
    elif pair[1]==el:
        return pair[0]
    else:
        return -1000


########################################################################################################
############ Block Related to the estimation od Dirac-Delta integral with trilinear interpolation  #####
########################################################################################################

###------------  Trilinear interpolation

MtriD = np.array(
    [[1,0,0,0, 0,0,0,0],
     [1,0,0,1, 0,0,0,0],
     [1,0,1,0, 0,0,0,0],
     [1,0,1,1, 0,0,1,0],
     [1,1,0,0, 0,0,0,0],
     [1,1,0,1, 0,1,0,0],
     [1,1,1,0, 1,0,0,0],
     [1,1,1,1, 1,1,1,1]]
)
MtriI = np.linalg.inv(MtriD)


@nb.njit
def nfTonxyz(nf):
    ## relation between a single vertice index nf
    ## and the coordinate indicies ix,iy,iz
    iz = nf%2
    iy = (nf//2)%2
    ix = nf//4
    return ix,iy,iz

@nb.njit
def vecTonf(v):
    ix = int(np.trunc(v[0]))
    iy = int(np.trunc(v[1]))
    iz = int(np.trunc(v[2]))
    return iz + 2*iy + 4*ix



def FooToFval(fun, r0, dr):
    ### calculates the verticies values for an arbitrary function fun
    #### Order: 1) convert i into ix,iy,iz (binary writting)
    ## 2) x-> 1,0,0; y->0,1,0; z->0,0,1
    dx,dy,dz = dr 
    fVal = np.zeros(8)
    for i in range(8):
        ix,iy,iz = nfTonxyz(i)
        # iz = i%2
        # iy = (i//2)%2
        # ix = i//4
        dr1 = np.array((ix*dx, iy*dy, iz*dz ))
        fVal[i] = fun(r0 + dr1)
    return fVal


@nb.njit
def FvalToAval(fval, dr):
    ### calculates the coefficients of trilinear approximation from the corner values
    ## Order: a0, ax, ay, az, axy, axz, ayz, axyz
    dx,dy,dz = dr
    b = MtriI@fval
    aval = np.array((b[0], b[1]/dx, b[2]/dy, b[3]/dz,
                    b[4]/(dx*dy), b[5]/(dx*dz), b[6]/(dy*dz), b[7]/(dx*dy*dz)))
    return aval

@nb.njit
def TriLinF(r, aval, r0):
    ### trilinear approximation, function value
    dx, dy, dz = r-r0
    Res = aval[0] + aval[1]*dx + aval[2]*dy + aval[3]*dz
    Res += aval[4]*dx*dy + aval[5]*dx*dz + aval[6]*dy*dz
    Res += aval[7]*dx*dy*dz
    return Res

@nb.njit
def TriLinDx(r, aval, r0):
    ### trilinear approximation, Derivative over x
    dx, dy, dz = r-r0
    Res = aval[1] 
    Res += aval[4]*dy + aval[5]*dz 
    Res += aval[7]*dy*dz
    return Res

@nb.njit
def TriLinDy(r, aval, r0):
    ### trilinear approximation, Derivative over y
    dx, dy, dz = r-r0
    Res = aval[2] 
    Res += aval[4]*dx + aval[6]*dz
    Res += aval[7]*dx*dz
    return Res

@nb.njit
def TriLinDz(r, aval, r0):
    ### trilinear approximation, Derivative over z
    dx, dy, dz = r-r0
    Res = aval[3]
    Res += aval[5]*dx + aval[6]*dy
    Res += aval[7]*dx*dy
    return Res


###----------------- enumeration of the cube verticies/edges/faces
### contains the information about the connections 
### between verticies, edges and faces
### in the enumeration used in the module

CuEdges = np.array([(0,4),(4,6),(2,6),(0,2),
          (0,1),(4,5),(6,7),(2,3),
          (1,5),(5,7),(3,7),(1,3)], dtype = np.intc)

## connection of verticies to edges
VertConnect = np.array([
    (0,3,4),
    (4,8,11),
    (2,3,7),
    (7,10,11),
    (0,1,5),
    (5,8,9),
    (1,2,6),
    (6,9,10),    
])

## connection of verticies to other verticies
VertConnectV = np.array([
    (1,2,4),
    (0,3,5),
    (0,3,6),
    (1,2,7),
    (0,5,6),
    (1,4,7),
    (2,4,7),
    (3,5,6),
])

FaceConnectEd = np.array([
    (0,1,2,3),
    (0,4,5,8),
    (1,5,6,9),
    (2,6,7,10),
    (3,4,7,11),
    (8,9,10,11),
])


EdgeConnectFaces = np.array([(0,1),(0,2),(0,3),(0,4),
          (1,4),(1,2),(2,3),(3,4),
          (1,5),(2,5),(3,5),(4,5)], dtype = np.intc)


VertToEdges = np.zeros((8,8), dtype = np.intc) - 1000
for i in range(8):
    for j in range(8):
        for ie, edg in enumerate(CuEdges):
            if (i==edg[0]) and (j==edg[1]):
                VertToEdges[i,j] = ie        
                VertToEdges[j,i] = ie        



##------------------- construction of the cube cross-section and its division over triangles
@nb.njit
def BrokenEdges(fval, dr):
    ### finds all the edges where the verticies correspond to different sign of fval
    ### and the corresponding zero-points based on trilinear approxiamtion
    eBrok = np.zeros(12, dtype=np.intc) - 1000
    drBrok = np.zeros((12,3), dtype=np.float64) - 1000.0
    icou = 0
    for ie in range(12):
        i1, i2 = CuEdges[ie]
        f1, f2 = fval[i1], fval[i2]
        if f1*f2 <=0:
            eBrok[icou] = ie
            ix1,iy1,iz1 = nfTonxyz(i1)
            ix2,iy2,iz2 = nfTonxyz(i2)
            dr1 = np.array((dr[0]*ix1, dr[1]*iy1, dr[2]*iz1))
            dr2 = np.array((dr[0]*ix2, dr[1]*iy2, dr[2]*iz2))
            if f1 == 0:
                drEdge = dr1
            elif f2 == 0:
                drEdge = dr2
            else:
                drEdge = (np.abs(f2)*dr1 + np.abs(f1)*dr2)/(np.abs(f1) + np.abs(f2))
            drBrok[icou] = drEdge
            icou += 1
    Nbroken = icou
    return Nbroken, eBrok, drBrok

@nb.njit
def MakeTriangles(fval, dr):
    ### function that calculate the set of triangles
    ### describing the cube cross-section
    
    Nbro, ebro, drbro = BrokenEdges(fval, dr)
    if Nbro<3:
        Ntri = 0
        Atri = np.zeros( (1,1,1), dtype=np.float64 )
        return Ntri, Atri
        

    aPoi = np.zeros((12,3), dtype=np.float64) - 1000.0
    aEdges = np.zeros(12, dtype=np.intc) - 1000

    icou = 0
    ie0 = ebro[0]
    aPoi[icou] = drbro[0]
    aEdges[icou] = ebro[0]
    old_f = EdgeConnectFaces[ebro[0]][0]
    old_e = ebro[0]
    icou += 1
    
    connect = False
    error = False
    
    while not connect:
        ie = -1000

        
        face = OtherEl(EdgeConnectFaces[old_e], old_f)
        if face>=0:
            for iet in FaceConnectEd[face]:
                if (iet != old_e) and (iet in ebro):
                    ie = iet
                    brokeArg = np.argwhere(ebro == ie)
                    brokeArg = brokeArg.flatten()[0]
        if ie>=0:
            aEdges[icou] = ie
            aPoi[icou] = drbro[brokeArg]
            icou += 1
            if ie == ie0:
                connect = True
            else:
                old_f = face
                old_e = ie
            if icou>=12:
                error = True
                connect = True
        else:
            error = True
            connect = True

    Npoi = icou - 1
    if error:
        Npoi = 0

    if Npoi >=3:
        Ntri = Npoi - 2
        Atri = np.zeros( (Ntri,3,3), dtype=np.float64 )
        for itri in range(Ntri):
            Atri[itri,0] = aPoi[0]
            Atri[itri,1] = aPoi[itri+1]
            Atri[itri,2] = aPoi[itri+2]
    else:
        Ntri = 0
        Atri = np.zeros( (1,1,1), dtype=np.float64 )
        
    return Ntri, Atri

####-------------- estimates integral of delta-function over a single triangle ---------------
@nb.njit
def DeltaIntTriangle(tri, aval, r0, Smin=AreaMin):
    dr1, dr2, dr3 = tri[0], tri[1], tri[2]
    vec1 = dr2 - dr1
    vec2 = dr3 - dr1
    Svec = np.cross(vec1,vec2)/2
    S = np.linalg.norm(Svec)
    if S<Smin:
        return 0.0
    Sevec = Svec/S

    drc = (dr1+dr2+dr3)/3.0

    gradV = np.array((TriLinDx(r0+drc, aval, r0),  TriLinDy(r0+drc, aval, r0), TriLinDz(r0+drc, aval, r0)  ))
    gradA = np.abs(np.dot(gradV, Sevec))
    return S/gradA

###!!!!!!!!!!!!!!!
####-------------- Main functions that combines everything ---------------
@nb.njit
def DeltaIntCube(fval, r0, dr):
    Ntri, Atri = MakeTriangles(fval, dr)
    if Ntri <1:
        return 0.0

    aval = FvalToAval(fval, dr)
    Res = 0.0
    for itri in range(Ntri):
        Res += DeltaIntTriangle(Atri[itri], aval, r0)
    return Res


@nb.njit
def DeltaIntCell(fval, dr1,dr2,dr3):
    ### estimates Delta-function integral for an arbitrary orthorombic cell with side vecors dr1, dr2, dr3
    Jac = np.abs(np.dot(dr1, np.cross(dr2,dr3)))
    if Jac == 0:
        return 0.0
    r0 = np.zeros(3, dtype=np.float64)
    r1 = np.zeros(3, dtype=np.float64) + 1.0
    res0 = DeltaIntCube(fval, r0, r1)
    return res0*Jac
    
########################################################################################################







### A function that select a best pair of points to search for the "Real zero"
@nb.njit
def BestLine(fval, r0,dr):
    Rdr1 = np.zeros(3, dtype=np.float64 )
    Rdr2 = np.zeros(3, dtype=np.float64 )
    dr_crit = 1e20
    for i1 in range(8):
        for i2 in range(8):
            if fval[i1]*fval[i2] <=0:
                ix1,iy1,iz1 = nfTonxyz(i1)
                ix2,iy2,iz2 = nfTonxyz(i2)
                dr1 = np.array((dr[0]*ix1, dr[1]*iy1, dr[2]*iz1))
                dr2 = np.array((dr[0]*ix2, dr[1]*iy2, dr[2]*iz2))
                if fval[i1]==0:
                    drE = dr1
                elif fval[i2]==0:
                    drE = dr2
                else:
                    f1 = fval[i1]
                    f2 = fval[i2]
                    drE = (np.abs(f2)*dr1 + np.abs(f1)*dr2)/(np.abs(f1) + np.abs(f2))
                cri = np.linalg.norm( drE - (dr/2)  )
                if cri < dr_crit:
                    Rdr1 = dr1
                    Rdr2 = dr2
                    dr_crit = cri
    return r0+Rdr1, r0+Rdr2
                    
                
                              





