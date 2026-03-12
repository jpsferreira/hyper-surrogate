"""
Generates a batch of random deformation gradients.
"""

import argparse
import logging

# time
from pathlib import Path

import numpy as np

from hyper_surrogate.data.deformation import DeformationGenerator
from hyper_surrogate.mechanics.kinematics import Kinematics as K
from hyper_surrogate.mechanics.materials import MooneyRivlin, NeoHooke
from hyper_surrogate.reporting.reporter import Reporter

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

    seed = args.seed if args.seed else 42

    # set material model
    if args.material == "nh":
        material = NeoHooke()
    elif args.material == "mr":
        material = MooneyRivlin()

    results = {"f": None, "pk2": None, "cmat": None}
    # numeric generator
    f = DeformationGenerator(seed=seed).combined(args.batch_size)
    results["f"] = f
    logging.info(f"Generated {args.batch_size} deformation gradients.")
    c = K.right_cauchy_green(f)

    # report pdf to output_path name
    reporter = Reporter(c)
    reporter.create_report(args.output_path.parent)

    if "pk2" in args.tensors:
        results["pk2"] = material.evaluate_pk2(c)
        logging.info(f"Generated {args.batch_size} pk2 tensors.")
    if "cmat" in args.tensors:
        results["cmat"] = material.evaluate_cmat(c)
        logging.info(f"Generated {args.batch_size} cmat tensors.")

    np.savez(args.output_path, **results)
    logging.info(f"Saved results to {args.output_path}.npz")
