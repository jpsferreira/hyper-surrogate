import datetime
import logging
from pathlib import Path
from typing import Any

import sympy as sym


class UMATHandler:
    def __init__(self, material_model: Any) -> None:
        """
        Initialize the UMAT handler with a specific material model.

        Args:
            material_model: The material model (e.g., NeoHooke) to generate UMAT code for.
        """
        self.material = material_model
        self.sigma_code = None
        self.smat_code = None

    @staticmethod
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
            sym.fcode(
                expr,
                standard=90,
                source_format="free",
                assign_to=sym.fcode(var, standard=90, source_format="free"),
            )
            for var, expr in replacements
        ]

        # Generate Fortran code for reduced expressions
        if tensor_matrix.shape[1] == 1:  # If vector
            reduced_code = [
                sym.fcode(
                    expr,
                    standard=90,
                    source_format="free",
                    assign_to=f"{var_name}({i + 1})",
                )
                for i, expr in enumerate(reduced_exprs)
            ]
        else:  # If matrix
            _, cols = tensor.shape
            reduced_code = [
                sym.fcode(
                    expr,
                    standard=90,
                    source_format="free",
                    assign_to=f"{var_name}({i // cols + 1},{i % cols + 1})",
                )
                for i, expr in enumerate(reduced_exprs)
            ]

        return aux_code + reduced_code

    @property
    def f(self) -> sym.Matrix:
        """The deformation gradient tensor."""
        return sym.Matrix(3, 3, lambda i, j: sym.Symbol(f"DFGRD1({i + 1},{j + 1})"))

    @property
    def sub_exp(self) -> dict:
        """Substitution expressions for the right Cauchy-Green tensor."""
        c = self.f.T * self.f
        return {self.material.c_tensor[i, j]: c[i, j] for i in range(3) for j in range(3)}

    def generate(self, filename: Path) -> None:
        """
        Generate the UMAT code for the material model and write it to a file.

        Args:
            filename (str): The file path where the UMAT code will be written.
        """
        sigma_code = self.generate_expression(self.cauchy, "stress")
        smat_code = self.generate_expression(self.tangent, "ddsdde")
        sigma_code_str = self.code_as_string(sigma_code)
        smat_code_str = self.code_as_string(smat_code)
        self.write_umat_code(sigma_code_str, smat_code_str, filename)

    @property
    def cauchy(self) -> Any:
        """
        Generate the symbolic expression for the Cauchy stress tensor.
        """
        logging.info("Generating Cauchy stress tensor...")
        return self.material.cauchy(self.f).subs(self.sub_exp)

    @property
    def tangent(self) -> Any:
        """
        Generate the symbolic expression for the tangent matrix.
        """
        logging.info("Generating tangent matrix...")
        return self.material.tangent(self.f).subs(self.sub_exp)

    def generate_expression(self, tensor: Any, var_name: str) -> Any:
        logging.info(f"Generating CSE for {var_name}...")
        return self.common_subexpressions(tensor, var_name)

    @staticmethod
    def code_as_string(code: list) -> str:
        """
        Convert a list of code lines into a single string.

        Args:
            code (list): The list of code lines.

        Returns:
            str: The code as a single string.
        """
        return "\n".join(code)

    def write_umat_code(self, sigma_code_str: str, smat_code_str: str, filename: Path) -> None:
        """
        Write the generated Fortran code into a UMAT subroutine file.

        Args:
            filename (Path): The file path where the UMAT code will be written.
        """
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        description = f"Automatically generated code for the UMAT subroutine using {self.material.__class__.__name__}."
        umat_code = f"""
!>********************************************************************
!> Record of revisions:                                              |
!>        Date        Programmer        Description of change        |
!>        ====        ==========        =====================        |
!>     {today}    Automatic Code      {description}                  |
!>--------------------------------------------------------------------
!C>
!C>   Material model: {self.material.__class__.__name__}
!C>
!---------------------------------------------------------------------

SUBROUTINE umat(stress, statev, ddsdde, sse, spd, scd, rpl, ddsddt, drplde, drpldt,  &
    stran, dstran, time, dtime, temp, dtemp, predef, dpred, cmname,  &
    ndi, nshr, ntens, nstatev, props, nprops, coords, drot, pnewdt,  &
    celent, dfgrd0, dfgrd1, noel, npt, layer, kspt, kstep, kinc)
!
!use global
!----------------------------------------------------------------------
!--------------------------- DECLARATIONS -----------------------------
!----------------------------------------------------------------------
INTEGER, INTENT(IN OUT)                  :: noel
INTEGER, INTENT(IN OUT)                  :: npt
INTEGER, INTENT(IN OUT)                  :: layer
INTEGER, INTENT(IN OUT)                  :: kspt
INTEGER, INTENT(IN OUT)                  :: kstep
INTEGER, INTENT(IN OUT)                  :: kinc
INTEGER, INTENT(IN OUT)                  :: ndi
INTEGER, INTENT(IN OUT)                  :: nshr
INTEGER, INTENT(IN OUT)                  :: ntens
INTEGER, INTENT(IN OUT)                  :: nstatev
INTEGER, INTENT(IN OUT)                  :: nprops
DOUBLE PRECISION, INTENT(IN OUT)         :: sse
DOUBLE PRECISION, INTENT(IN OUT)         :: spd
DOUBLE PRECISION, INTENT(IN OUT)         :: scd
DOUBLE PRECISION, INTENT(IN OUT)         :: rpl
DOUBLE PRECISION, INTENT(IN OUT)         :: dtime
DOUBLE PRECISION, INTENT(IN OUT)         :: drpldt
DOUBLE PRECISION, INTENT(IN OUT)         :: temp
DOUBLE PRECISION, INTENT(IN OUT)         :: dtemp
CHARACTER (LEN=8), INTENT(IN OUT)        :: cmname
DOUBLE PRECISION, INTENT(IN OUT)         :: pnewdt
DOUBLE PRECISION, INTENT(IN OUT)         :: celent

DOUBLE PRECISION, INTENT(IN OUT)         :: stress(ntens)
DOUBLE PRECISION, INTENT(IN OUT)         :: statev(nstatev)
DOUBLE PRECISION, INTENT(IN OUT)         :: ddsdde(ntens, ntens)
DOUBLE PRECISION, INTENT(IN OUT)         :: ddsddt(ntens)
DOUBLE PRECISION, INTENT(IN OUT)         :: drplde(ntens)
DOUBLE PRECISION, INTENT(IN OUT)         :: stran(ntens)
DOUBLE PRECISION, INTENT(IN OUT)         :: dstran(ntens)
DOUBLE PRECISION, INTENT(IN OUT)         :: time(2)
DOUBLE PRECISION, INTENT(IN OUT)         :: predef(1)
DOUBLE PRECISION, INTENT(IN OUT)         :: dpred(1)
DOUBLE PRECISION, INTENT(IN)             :: props(nprops)
DOUBLE PRECISION, INTENT(IN OUT)         :: coords(3)
DOUBLE PRECISION, INTENT(IN OUT)         :: drot(3, 3)
DOUBLE PRECISION, INTENT(IN OUT)         :: dfgrd0(3, 3)
DOUBLE PRECISION, INTENT(IN OUT)         :: dfgrd1(3, 3)

DOUBLE PRECISION :: C10  ! Example material property

! Initialize material properties
C10 = PROPS(1)

! Define the stress calculation from the pk2 symbolic expression
{sigma_code_str}

! Define the tangent matrix calculation from the smat symbolic expression
{smat_code_str}

RETURN
END SUBROUTINE umat
"""

        with open(filename, "w") as file:
            file.write(umat_code)
        logging.info(f"UMAT subroutine written to {filename}")

    # def generate(self, filename: str):
    #     """
    #     Generate the UMAT code for the material model and write it to a file.

    #     Args:
    #         filename (str): The file path where the UMAT code will be written.
    #     """
    #     self.sigma_code = self.generate_expression(self.cauchy_stress, "STRESS")
    #     self.smat_code = self.generate_expression(self.tangent_matrix, "DDSDDE")
    #     self.write_umat_code(filename)
