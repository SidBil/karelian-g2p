#!/usr/bin/env python3
"""
Process Karelian terms dataset by removing:
1. Entries where the term starts with a hyphen ('-')
2. Single-character entries
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def process_karelian_data(input_path: Path, output_path: Path, dry_run: bool = False) -> dict:
    """
    Reads Karelian terms from input_path, filters out entries starting with '-'
    or having single-character terms, and writes the clean dataset to output_path.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    rows = []
    with open(input_path, mode="r", encoding="utf-8-sig", newline="") as infile:
        reader = csv.reader(infile)
        rows = list(reader)

    total_count = len(rows)
    deleted_hyphen = []
    deleted_single_char = []
    retained_rows = []

    for line_num, row in enumerate(rows, start=1):
        if not row:
            continue

        term = row[0].strip()

        if term.startswith("-"):
            deleted_hyphen.append((line_num, row))
        elif len(term) <= 1:
            deleted_single_char.append((line_num, row))
        else:
            retained_rows.append(row)

    stats = {
        "total_read": total_count,
        "deleted_hyphen_count": len(deleted_hyphen),
        "deleted_single_char_count": len(deleted_single_char),
        "total_deleted": len(deleted_hyphen) + len(deleted_single_char),
        "total_retained": len(retained_rows),
        "deleted_hyphen_items": deleted_hyphen,
        "deleted_single_char_items": deleted_single_char,
    }

    print(f"=== Karelian Data Processing Summary ===")
    print(f"Input file:             {input_path}")
    print(f"Total entries read:     {stats['total_read']}")
    print(f"Deleted (hyphen prefix):{stats['deleted_hyphen_count']}")
    for line_num, r in deleted_hyphen:
        print(f"  Line {line_num:4d}: {r}")

    print(f"Deleted (single char):  {stats['deleted_single_char_count']}")
    for line_num, r in deleted_single_char:
        print(f"  Line {line_num:4d}: {r}")

    print(f"Total deleted:          {stats['total_deleted']}")
    print(f"Total retained:         {stats['total_retained']}")

    if not dry_run:
        with open(output_path, mode="w", encoding="utf-8", newline="") as outfile:
            writer = csv.writer(outfile)
            writer.writerows(retained_rows)
        print(f"Saved processed dataset to: {output_path}")
    else:
        print("[Dry Run] Output file was not modified.")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Process Karelian dataset: delete entries starting with hyphen and single-character entries."
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=ROOT / "data" / "karelian_terms.csv",
        help="Path to input CSV file (default: data/karelian_terms.csv)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Path to output CSV file (default: overwrites input file)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform filtering without modifying files"
    )

    args = parser.parse_args()

    output_file = args.output if args.output is not None else args.input

    try:
        process_karelian_data(args.input, output_file, dry_run=args.dry_run)
    except Exception as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
