import logging
from typing import Any

import sympy as sym

from hyper_surrogate.materials import NeoHooke
from hyper_surrogate.umat_handler import UMATHandler

# set loglevel to INFO
logging.basicConfig(level=logging.INFO)


def common_subexpressions(tensor: sym.Matrix, var_name: str) -> Any:
    """
    Perform common subexpression elimination on a vector or matrix and generate Fortran code.

    Args:
        vector (list): The symbolic vector or matrix to process.
        var_name (str): The base name for the variables in the Fortran code.

    Returns:
        tuple: A tuple containing Fortran code for auxiliary variables and reduced expressions.
    """
    # Extract individual components
    # tensor_components = [tensor[i] for i in range(tensor.shape[0])]
    tensor_components = [tensor[i, j] for i in range(tensor.shape[0]) for j in range(tensor.shape[1])]
    # Convert to a matrix to check shape
    tensor_matrix = sym.Matrix(tensor)
    # Perform common subexpression elimination
    replacements, reduced_exprs = sym.cse(tensor_components)

    # Generate Fortran code for auxiliary variables (replacements)
    aux_code = [
        sym.fcode(expr, standard=90, source_format="free", assign_to=sym.fcode(var, standard=90, source_format="free"))
        for var, expr in replacements
    ]

    # Generate Fortran code for reduced expressions
    if tensor_matrix.shape[1] == 1:  # If vector
        reduced_code = [
            sym.fcode(expr, standard=90, source_format="free", assign_to=f"{var_name}({i + 1})")
            for i, expr in enumerate(reduced_exprs)
        ]
    else:  # If matrix
        _, cols = tensor.shape
        reduced_code = [
            sym.fcode(expr, standard=90, source_format="free", assign_to=f"{var_name}({i // cols + 1},{i % cols + 1})")
            for i, expr in enumerate(reduced_exprs)
        ]

    return aux_code + reduced_code


material = NeoHooke()

umat = UMATHandler(material)
umat.generate("UMAT_NeoHooke.f90")
# cauchy = umat.cauchy_stress()
# logging.info(cauchy.shape)
# tangent = umat.tangent_matrix()
# logging.info(tangent.shape)
# # umat.generate("UMAT_NeoHooke.f90")
# umat.write_umat_code(cauchy, tangent, "UMAT_NeoHooke.f90")

# # pk2 and cmat are sympy expressions written in terms of the material parameters and right Cauchy-Green tensor components.
# # create symbolic deformation gradient tensor
# f = sym.Matrix(3, 3, lambda i, j: sym.Symbol(f"DFGRD1({i + 1},{j + 1})"))
# # calculate the right Cauchy-Green tensor from the deformation gradient tensor
# c = f.T * f
# # We can substitute these values into the tensor expressions to get the expressions in terms of the deformation gradient tensor components.
# # C_11 is c[0,0], C_22 is c[1,1], C_33 is c[2,2], C_12 is c[0,1], C_13 is c[0,2], C_23 is c[1,2]
# sub_exp = {material.c_tensor[i, j]: c[i, j] for i in range(3) for j in range(3)}
# # logging.info("Pushing forward pk2...")
# sub_exp = umat.sub_exp

# sigma = material.cauchy(f).subs(sub_exp)
# smat = material.tangent(f).subs(sub_exp)

# logging.info("Gathering components...")
# # Generate Fortran code for each component

# sigma_code = common_subexpressions(sigma, "stress")
# smat_code = common_subexpressions(smat, "ddsdde")

# sigma_code_str = "\n".join(sigma_code)
# smat_code_str = "\n".join(smat_code)

# today = datetime.datetime.now()
# description = "Automatic generated code"
# umat_code = f"""
# !>********************************************************************
# !> Record of revisions:                                              |
# !>        Date        Programmer        Description of change        |
# !>        ====        ==========        =====================        |
# !>     {today}    Joao Ferreira      {description}           |
# !>--------------------------------------------------------------------
# !>     Description:
# !C>
# !C>
# !C>
# !>--------------------------------------------------------------------
# !>---------------------------------------------------------------------

# SUBROUTINE umat(stress,statev,ddsdde,sse,spd,scd, rpl,ddsddt,drplde,drpldt,  &
#     stran,dstran,time,dtime,temp,dtemp,predef,dpred,cmname,  &
#     ndi,nshr,ntens,nstatev,props,nprops,coords,drot,pnewdt,  &
#     celent,dfgrd0,dfgrd1,noel,npt,layer,kspt,kstep,kinc)
# !
# !use global
# !----------------------------------------------------------------------
# !--------------------------- DECLARATIONS -----------------------------
# !----------------------------------------------------------------------
# INTEGER, INTENT(IN OUT)                  :: noel
# INTEGER, INTENT(IN OUT)                  :: npt
# INTEGER, INTENT(IN OUT)                  :: layer
# INTEGER, INTENT(IN OUT)                  :: kspt
# INTEGER, INTENT(IN OUT)                  :: kstep
# INTEGER, INTENT(IN OUT)                  :: kinc
# INTEGER, INTENT(IN OUT)                  :: ndi
# INTEGER, INTENT(IN OUT)                  :: nshr
# INTEGER, INTENT(IN OUT)                  :: ntens
# INTEGER, INTENT(IN OUT)                  :: nstatev
# INTEGER, INTENT(IN OUT)                  :: nprops
# DOUBLE PRECISION, INTENT(IN OUT)         :: sse
# DOUBLE PRECISION, INTENT(IN OUT)         :: spd
# DOUBLE PRECISION, INTENT(IN OUT)         :: scd
# DOUBLE PRECISION, INTENT(IN OUT)         :: rpl
# DOUBLE PRECISION, INTENT(IN OUT)         :: dtime
# DOUBLE PRECISION, INTENT(IN OUT)         :: drpldt
# DOUBLE PRECISION, INTENT(IN OUT)         :: temp
# DOUBLE PRECISION, INTENT(IN OUT)         :: dtemp
# CHARACTER (LEN=8), INTENT(IN OUT)        :: cmname
# DOUBLE PRECISION, INTENT(IN OUT)         :: pnewdt
# DOUBLE PRECISION, INTENT(IN OUT)         :: celent

# DOUBLE PRECISION, INTENT(IN OUT)         :: stress(ntens)
# DOUBLE PRECISION, INTENT(IN OUT)         :: statev(nstatev)
# DOUBLE PRECISION, INTENT(IN OUT)         :: ddsdde(ntens,ntens)
# DOUBLE PRECISION, INTENT(IN OUT)         :: ddsddt(ntens)
# DOUBLE PRECISION, INTENT(IN OUT)         :: drplde(ntens)
# DOUBLE PRECISION, INTENT(IN OUT)         :: stran(ntens)
# DOUBLE PRECISION, INTENT(IN OUT)         :: dstran(ntens)
# DOUBLE PRECISION, INTENT(IN OUT)         :: time(2)
# DOUBLE PRECISION, INTENT(IN OUT)         :: predef(1)
# DOUBLE PRECISION, INTENT(IN OUT)         :: dpred(1)
# DOUBLE PRECISION, INTENT(IN)             :: props(nprops)
# DOUBLE PRECISION, INTENT(IN OUT)         :: coords(3)
# DOUBLE PRECISION, INTENT(IN OUT)         :: drot(3,3)
# DOUBLE PRECISION, INTENT(IN OUT)         :: dfgrd0(3,3)
# DOUBLE PRECISION, INTENT(IN OUT)         :: dfgrd1(3,3)

# DOUBLE PRECISION :: C10  ! Material property, example

# ! Initialize material properties
# C10 = PROPS(1)

# ! Define the stress calculation from the pk2 symbolic expression.
# {sigma_code_str}

# ! Define the tangent matrix calculation from the smat symbolic expression
# {smat_code_str}

# RETURN
# END SUBROUTINE umat
# """


# with open("UMAT_NeoHooke.f90", "w") as file:
#     file.write(umat_code)
