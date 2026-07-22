"""
This script creates a jsonl file containing the first (by default) 50 errors (answer_status == "incorrect" or "invalid")
from a specified results file that is already in the output directory.

Usage example: py find_errors.py 2026-04-07_14-52-31_qc1319_one-shot_results.jsonl --error-count 10
Resulting in: 2026-04-07_14-52-31_qc1319_one-shot_first_10_errors.jsonl
"""

import argparse
import json
from pathlib import Path
import sys


def main():
    args = get_command_line_args()

    if "results.jsonl" not in args.filename:
        sys.exit("Filename must include \"results.jsonl\"")

    base_path = Path(__file__).parent.parent / "output"
    results_path = base_path / args.filename
    first_x_errors = base_path / args.filename.replace("results.jsonl",
                                                       f"first_{args.error_count}_errors.jsonl")

    with results_path.open('r') as res, first_x_errors.open('w') as errors:
        i, count = 0, 0
        lines = res.readlines()
        while count < args.error_count and i < len(lines):
            curr = json.loads(lines[i])
            if curr["answer_status"] != "correct":
                errors.write(lines[i])
                count += 1
            i += 1
        print(f"{count} failures found by position {i - 1}.")


def get_command_line_args():
    parser = argparse.ArgumentParser(
        description="Get the first X errors from the specified results file.")

    parser.add_argument(
        "filename",
        type=str,
        help="Name of results file in the output directory (must already exist)",
    )
    parser.add_argument(
        "-ec",
        "--error-count",
        type=int,
        default=50,
        help="Number of errors to find",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
