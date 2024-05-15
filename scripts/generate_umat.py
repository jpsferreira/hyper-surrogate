import sympy as sym

from hyper_surrogate.materials import NeoHooke

material = NeoHooke()
pk2 = material.pk2_symb
cmat = material.cmat_symb

# pk2 and cmat are sympy expressions written in terms of the material parameters and right Cauchy-Green tensor components.
# create symbolic deformation gradient tensor
f = sym.Matrix(3, 3, lambda i, j: sym.Symbol(f"DFGRD1({i+1},{j+1})"))
# calculate the right Cauchy-Green tensor from the deformation gradient tensor
c = f.T * f
# C_11 is c[0,0], C_22 is c[1,1], C_33 is c[2,2], C_12 is c[0,1], C_13 is c[0,2], C_23 is c[1,2]
# We can substitute these values into the pk2 and cmat expressions to get the expressions in terms of the deformation gradient tensor components.
sub_exp = {material.c_tensor[i, j]: c[i, j] for i in range(3) for j in range(3)}
pk2 = pk2.subs(sub_exp)
cmat = cmat.subs(sub_exp)
# We can convert these expressions to Fortran code using sympy's fcode function.
pk2_code = sym.fcode(pk2, standard=90, source_format="free")
cmat_code = sym.fcode(cmat, standard=90, source_format="free")


umat_code = f"""
      SUBROUTINE UMAT(STRESS, STATEV, DDSDDE, SSE, SPD, SCD,
     & RPL, DDSDDT, DRPLDE, DRPLDT, STRAN, DSTRAN, TIME, DTIME,
     & TEMP, DTEMP, PREDEF, DPRED, CMNAME, NDI, NSHR, NTENS,
     & NSTATV, PROPS, NPROPS, COORDS, DROT, PNEWDT, CELENT,
     & DFGRD0, DFGRD1, NOEL, NPT, LAYER, KSPT, KSTEP, KINC)
      IMPLICIT NONE
      INCLUDE 'ABA_PARAM.INC'
      DOUBLE PRECISION STRESS(NTENS), STATEV(NSTATV),
     & DDSDDE(NTENS,NTENS), SSE, SPD, SCD, RPL, DDSDDT, DRPLDE,
     & DRPLDT, STRAN(NTENS), DSTRAN(NTENS), TIME(2), DTIME,
     & TEMP, DTEMP, PREDEF(*), DPRED(*), PROPS(*), COORDS(3),
     & DROT(3,3), PNEWDT, CELENT, DFGRD0(3,3), DFGRD1(3,3)
      INTEGER NDI, NSHR, NTENS, NSTATV, NPROPS, NOEL, NPT,
     & LAYER, KSPT, KSTEP, KINC
      CHARACTER*80 CMNAME
      DOUBLE PRECISION C10  ! Material property, example

      ! Initialize material properties
      C10 = PROPS(1)

      ! Define the stress calculation from the pk2 symbolic expression
STRESS={pk2_code}

      ! Define the tangent matrix calculation from the cmat symbolic expression
DDSDDE={cmat_code}

      RETURN
      END
"""


with open("UMAT_NeoHooke.f", "w") as file:
    file.write(umat_code)
