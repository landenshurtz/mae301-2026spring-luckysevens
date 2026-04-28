import argparse
#!/usr/bin/env python3
import collections
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
    raw_52_window = collections.deque()

    with open(input_path, "r", encoding="utf-8") as infile:
        for line_number, raw_line in enumerate(infile, start=1):
            line = raw_line.strip()
            if not line:
                continue

            fields = line.split(",")
            if len(fields) != 57:
                print(
                    f"Warning: line {line_number} has {len(fields)} fields "
                    f"(expected 57).",
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

            if len(fields) > 52:
                raw_value = fields[52].strip()
                if raw_value:
                    try:
                        raw_52_window.append((timestamp, float(raw_value)))
                    except ValueError:
                        pass

            cutoff = timestamp - datetime.timedelta(seconds=1)
            while raw_52_window and raw_52_window[0][0] < cutoff:
                raw_52_window.popleft()

            quarter_microsecond = (timestamp.microsecond // 250000) * 250000
            quarter_key = timestamp.replace(microsecond=quarter_microsecond)
            if quarter_key != last_quarter:
                avg_52 = None
                if raw_52_window:
                    avg_52 = sum(value for _, value in raw_52_window) / len(raw_52_window)

                processed_fields = []
                for idx, field in enumerate(fields):
                    if idx in {
                        0, 1, 2+2, 4+2, 5+2, 6+2, 7+2, 8+2, 9+2,
                        17+2, 18+2, 22+2, 23+2, 24+2, 31+2, 32+2, 33+2, 34+2, 35+2, 36+2, 37+2, 38+2,
                        47+2, 49+2,
                    }:
                        continue
                    if idx == 52:
                        if avg_52 is not None:
                            processed_fields.append(f"{avg_52:.6f}")
                        elif field.strip() != "":
                            processed_fields.append(field)
                        continue
                    if field.strip() == "":
                        continue
                    processed_fields.append(field)

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