"""
Generates a batch of random deformation gradients.
"""

import argparse

# time
import time
from pathlib import Path

import numpy as np

from hyper_surrogate.deformation_gradient import DeformationGradientGenerator
from hyper_surrogate.kinematics import Kinematics as K
from hyper_surrogate.materials import NeoHooke

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

    # numeric generator
    start_time = time.time()
    f = DeformationGradientGenerator(seed=seed, size=args.batch_size).generate()
    end_time = time.time()
    print(f"Generated {args.batch_size} deformation gradients in {end_time - start_time:.5f} seconds.")
    c = K.right_cauchy_green(f)

    nh = NeoHooke()
    # evaluate lambdify
    start_time = time.time()
    pk2_generator = nh.pk2()
    pk2_func_iterator = nh.evaluate_iterator(pk2_generator, c, 1)
    b = np.array(list(pk2_func_iterator))
    end_time = time.time()
    print(b)
    print(f"Generated {args.batch_size} pk2 lambdified tensors in {end_time - start_time:.5f} seconds.")
    print(f"Average time per entry (ms): {(end_time - start_time)*1000 / args.batch_size:.5f} ms.")
