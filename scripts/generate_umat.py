import argparse
import logging
from pathlib import Path

from hyper_surrogate.materials import MooneyRivlin, NeoHooke
from hyper_surrogate.umat_handler import UMATHandler

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate UMAT code for a specific material model.")
    parser.add_argument(
        "--material",
        type=str,
        help="The material model to generate UMAT code for.",
        choices=("NeoHooke", "MooneyRivlin"),
        default="NeoHooke",
        required=True,
    )
    parser.add_argument("--output", type=Path, help="The output file name for the generated UMAT code.", required=True)
    args = parser.parse_args()

    logging.info(f"Generating UMAT code for {args.material}...")
    material = NeoHooke() if args.material == "NeoHooke" else MooneyRivlin()
    umat = UMATHandler(material)
    logging.info(f"Writing UMAT code to {args.output}...")
    umat.generate(args.output)
