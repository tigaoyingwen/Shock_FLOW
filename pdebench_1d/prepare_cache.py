"""CLI for building the reduced-resolution PDEBench cache."""

import argparse

from .data import prepare_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--spatial-size", type=int, default=256)
    parser.add_argument("--time-stride", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--spatial-reduction", choices=("conservative", "linear"), default="conservative"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    path = prepare_cache(
        args.source,
        args.output_dir,
        spatial_size=args.spatial_size,
        time_stride=args.time_stride,
        batch_size=args.batch_size,
        spatial_reduction=args.spatial_reduction,
        overwrite=args.overwrite,
    )
    print(path, flush=True)


if __name__ == "__main__":
    main()
