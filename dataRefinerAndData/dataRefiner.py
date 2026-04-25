import argparse
#!/usr/bin/env python3
import datetime
import sys

def parse_timestamp(field: str):
    try:
        return datetime.datetime.fromisoformat(field)
    except ValueError:
        return None


def find_timestamp(fields):
    for field in fields:
        ts = parse_timestamp(field)
        if ts is not None:
            return ts
    return None


def refine_file(input_path: str, output_path: str):
    last_quarter = None
    kept_lines = []

    with open(input_path, "r", encoding="utf-8") as infile:
        for line_number, raw_line in enumerate(infile, start=1):
            line = raw_line.strip()
            if not line:
                continue

            fields = line.split(",")
            if len(fields) != 55:
                print(
                    f"Warning: line {line_number} has {len(fields)} fields "
                    f"(expected 55).",
                    file=sys.stderr,
                )

            timestamp = find_timestamp(fields)
            if timestamp is None:
                print(
                    f"Warning: line {line_number} does not contain a parseable "
                    "ISO timestamp.",
                    file=sys.stderr,
                )
                continue

            quarter_microsecond = (timestamp.microsecond // 250000) * 250000
            quarter_key = timestamp.replace(microsecond=quarter_microsecond)
            if quarter_key != last_quarter:
                processed_fields = [field for idx, field in enumerate(fields) if idx not in {0, 7, 8, 9, 10, 11, 12}]
                kept_lines.append(",".join(processed_fields))
                last_quarter = quarter_key

    with open(output_path, "w", encoding="utf-8") as outfile:
        outfile.write("\n".join(kept_lines))
        if kept_lines:
            outfile.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Refine raw comma-separated text data by keeping "
                    "only the first record when the timestamp crosses into a new second."
    )
    parser.add_argument("input_file", help="Input .txt file with raw data")
    parser.add_argument("output_file", help="Output .txt file for refined data")
    args = parser.parse_args()

    refine_file(args.input_file, args.output_file)


if __name__ == "__main__":
    main()