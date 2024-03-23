"""
Generates a batch of random deformation gradients.
"""

import argparse

# time
import time
from pathlib import Path

import numpy as np
import sympy as sym

from hyper_surrogate.deformation_gradient import DeformationGradientGenerator
from hyper_surrogate.kinematics import Kinematics as K
from hyper_surrogate.symbolic import SymbolicHandler as handler

# if main ==
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_path", "-o", type=Path, help="Output file")
    parser.add_argument(
        "--batch_size",
        "-size",
        type=int,
        required=True,
        help="Number of deformation gradients to generate",
    )
    parser.add_argument("--seed", type=int, help="Random seed")
    args = parser.parse_args()

    seed = args.seed if args.seed else None

    h = handler()
    # symbolic starter
    sef = (h.invariant1 - 3) * sym.Symbol("C10") + (h.invariant2 - 3) * sym.Symbol("C01")
    sef_params = {"C10": 1, "C01": 1}
    pk2 = h.pk2_tensor(sef)
    # numeric generator
    start_time = time.time()
    f = DeformationGradientGenerator(seed=seed, size=args.batch_size).generate()
    c = K.right_cauchy_green(f)
    end_time = time.time()
    print(f"Generated {args.batch_size} deformation gradients in {end_time - start_time:.5f} seconds.")
    #
    start_time = time.time()
    pk2_iterator = h.substitute_iterator(pk2, c, sef_params)
    # add each entrue of pk2_iterator to np array
    a = np.array(list(pk2_iterator))
    end_time = time.time()
    print(f"Generated {args.batch_size} pk2 tensors in {end_time - start_time:.5f} seconds.")
    # average per entry
    print(f"Average time per entry: {(end_time - start_time) / args.batch_size:.5f} seconds.")

    # evaluate lambdify
    pk2_func = h.lambdify(pk2, *sef_params.keys())
    start_time = time.time()
    pk2_func_iterator = h.evaluate_iterator(pk2_func, c, 1, 1)
    b = np.array(list(pk2_func_iterator))
    end_time = time.time()
    print(f"Generated {args.batch_size} pk2 lambdified tensors in {end_time - start_time:.5f} seconds.")
    print(f"Average time per entry (ms): {(end_time - start_time)*1000 / args.batch_size:.5f} ms.")
