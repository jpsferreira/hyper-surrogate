import argparse
import logging
from pathlib import Path

from hyper_surrogate.umat_handler import UMATHandler

# set loglevel to INFO
logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate UMAT code for a specific material model.")
    parser.add_argument(
        "material_model",
        type=str,
        help="The material model to generate UMAT code for.",
        choices=["NeoHooke", "MooneyRivlin"],
        default="NeoHooke",
    )
    parser.add_argument("--output", type=Path, help="The output file name for the generated UMAT code.", required=True)
    args = parser.parse_args()

    logging.info(f"Generating UMAT code for {args.material_model}...")
    material = globals()[args.material_model]()
    umat = UMATHandler(material)
    logging.info(f"Writing UMAT code to {args.output}...")
    umat.generate(args.output)
