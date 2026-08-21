"""Build audited paper figures from explicitly selected run artifacts."""

import argparse
import json

from paper_figure_registry import FIGURE_REGISTRY
from paper_figures import build_paper_figures


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument(
        "--figure", default="all", choices=("all", *FIGURE_REGISTRY.keys())
    )
    parser.add_argument("--output-root", default="results/paper_figures")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = build_paper_figures(
        args.spec, figure=args.figure, output_root=args.output_root
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
