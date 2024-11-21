"""
Generates a batch of random deformation gradients.
"""

import argparse
import logging

# time
from pathlib import Path

import numpy as np

from hyper_surrogate.deformation_gradient import DeformationGradientGenerator
from hyper_surrogate.kinematics import Kinematics as K
from hyper_surrogate.materials import MooneyRivlin, NeoHooke
from hyper_surrogate.reporter import Reporter

# set log level
logging.basicConfig(level=logging.INFO)
# set numpy print options
np.set_printoptions(precision=4, suppress=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_path", "-o", type=Path, help="Output file", required=True)
    parser.add_argument(
        "--batch_size",
        "-size",
        type=int,
        required=True,
        help="Number of deformation gradients to generate",
    )
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument(
        "--tensors",
        nargs="+",
        help="Tensors to generate",
        choices=["f", "pk2", "cmat"],
        default=["f", "pk2", "cmat"],
    )
    parser.add_argument(
        "--material",
        type=str,
        help="Material model to use",
        default="nh",
        choices=["nh", "mr"],
    )
    parser.add_argument("--model_params", type=float, nargs="+", help="Material model parameters")
    args = parser.parse_args()

    seed = args.seed if args.seed else None

    # if "pk2" in args.tensors or "cmat" in args.tensors:
    #     assert args.material, "Material model must be specified to generate pk2 or cmat tensors."
    #     assert len(args.model_params) > 0, "Material model parameters must be specified."

    # set material model
    if args.material == "nh":
        material = NeoHooke()
    elif args.material == "mr":
        material = MooneyRivlin()

    results = {"f": None, "pk2": None, "cmat": None}
    # numeric generator
    f = DeformationGradientGenerator(seed=seed, size=args.batch_size).generate()
    results["f"] = f
    logging.info(f"Generated {args.batch_size} deformation gradients.")
    c = K.right_cauchy_green(f)
    # first invariant
    I1 = K.invariant1(c)

    # report pdf to output_path name
    reporter = Reporter(c, args.output_path.parent)
    reporter.create_report()

    if "pk2" in args.tensors:
        pk2_func_iterator = material.evaluate_iterator(material.pk2(), c, 1)
        results["pk2"] = np.array(list(pk2_func_iterator))
        logging.info(f"Generated {args.batch_size} pk2 tensors.")
    if "cmat" in args.tensors:
        cmat_func_iterator = material.evaluate_iterator(material.cmat(), c, 1)
        results["cmat"] = np.array(list(cmat_func_iterator))
        logging.info(f"Generated {args.batch_size} cmat tensors.")

    np.savez(args.output_path, **results)
    logging.info(f"Saved results to {args.output_path}.npz")
