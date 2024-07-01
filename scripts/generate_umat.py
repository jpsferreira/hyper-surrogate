import logging

import sympy as sym

from hyper_surrogate.materials import NeoHooke

# set loglevel to INFO
logging.basicConfig(level=logging.INFO)

material = NeoHooke()
pk2 = material.pk2_symb
cmat = material.cmat_symb

# pk2 and cmat are sympy expressions written in terms of the material parameters and right Cauchy-Green tensor components.
# create symbolic deformation gradient tensor
f = sym.Matrix(3, 3, lambda i, j: sym.Symbol(f"DFGRD1({i+1},{j+1})"))
# calculate the right Cauchy-Green tensor from the deformation gradient tensor
c = f.T * f
# We can substitute these values into the tensor expressions to get the expressions in terms of the deformation gradient tensor components.
# C_11 is c[0,0], C_22 is c[1,1], C_33 is c[2,2], C_12 is c[0,1], C_13 is c[0,2], C_23 is c[1,2]
sub_exp = {material.c_tensor[i, j]: c[i, j] for i in range(3) for j in range(3)}
logging.info("Pushing forward pk2...")
sigma = NeoHooke.pushforward_2nd_order(pk2, f)
logging.info("Reducing pk2...")
sigma = NeoHooke.reduce_2nd_order(sigma)
logging.info("Pushing forward cmat...")
smat = NeoHooke.pushforward_4th_order(cmat, f)
logging.info("Reducing cmat...")
smat = NeoHooke.reduce_4th_order(smat)
logging.info("Substituting expressions...")
sigma = sigma.subs(sub_exp)
smat = smat.subs(sub_exp)


# Extracting individual components to avoid using unsupported structures
sigma_components = [sigma[i] for i in range(6)]
smat_components = [smat[i, j] for i in range(6) for j in range(6)]
logging.info("Gathering components...")
# Generate Fortran code for each component
sigma_code = [
    sym.fcode(comp, standard=90, source_format="free", assign_to=f"STRESS({i+1})")
    for i, comp in enumerate(sigma_components)
]
smat_code = [
    sym.fcode(comp, standard=90, source_format="free", assign_to=f"DDSDDE({i//6+1},{i%6+1})")
    for i, comp in enumerate(smat_components)
]

pk2_code_str = "\n      ".join(sigma_code)
cmat_code_str = "\n      ".join(smat_code)

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

      ! Define the stress calculation from the pk2 symbolic expression.
      ! iterate over all components of the stress pk2_code
      {pk2_code_str}

      ! Define the tangent matrix calculation from the cmat symbolic expression
      {cmat_code_str}

      RETURN
      END
"""


with open("UMAT_NeoHooke.f", "w") as file:
    file.write(umat_code)
