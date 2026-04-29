#!/usr/bin/env python3
import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def run_data_refiner(python_executable: str, script_path: Path, input_path: Path, output_path: Path) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".txt") as temp_file:
        temp_path = Path(temp_file.name)
        with input_path.open("r", encoding="utf-8", errors="ignore") as source_file:
            lines = source_file.readlines()[-1000:]
        temp_file.writelines(lines)

    try:
        subprocess.run(
            [python_executable, str(script_path), str(temp_path), str(output_path)],
            check=True,
            stderr=subprocess.DEVNULL,
        )
    finally:
        try:
            temp_path.unlink()
        except OSError:
            pass


def run_predict_latest(python_executable: str, script_path: Path, csv_path: str | None, model_path: str | None) -> None:
    command = [python_executable, str(script_path)]
    if csv_path:
        command.extend(["--csv", csv_path])
    if model_path:
        command.extend(["--model", model_path])

    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuously run dataRefiner followed by predict_latest every interval seconds."
    )
    parser.add_argument(
        "--refine-input",
        default="data_1.txt",
        help="Path to the raw data input file for dataRefiner.",
    )
    parser.add_argument(
        "--refine-output",
        default="refined_output.txt",
        help="Path to write dataRefiner output.",
    )
    parser.add_argument(
        "--csv",
        help="Optional CSV file path to pass to predict_latest. If omitted, predict_latest uses the latest CSV in the directory.",
    )
    parser.add_argument(
        "--model",
        help="Optional model path to pass to predict_latest.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.25,
        help="Seconds to wait between each dataRefiner + predict_latest cycle.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_refiner_script = script_dir / "dataRefiner.py"
    predict_script = script_dir / "predict_latest.py"

    if not data_refiner_script.exists():
        raise FileNotFoundError(f"Missing dataRefiner.py at {data_refiner_script}")
    if not predict_script.exists():
        raise FileNotFoundError(f"Missing predict_latest.py at {predict_script}")

    input_path = Path(args.refine_input)
    output_path = Path(args.refine_output)
    if not input_path.is_absolute():
        input_path = script_dir / input_path
    if not output_path.is_absolute():
        output_path = script_dir / output_path

    print("Starting continuous refine + predict loop.")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            start_time = time.time()

            try:
                run_data_refiner(sys.executable, data_refiner_script, input_path, output_path)
            except subprocess.CalledProcessError as exc:
                print(f"dataRefiner failed with exit code {exc.returncode}.")

            try:
                run_predict_latest(sys.executable, predict_script, args.csv, args.model)
            except subprocess.CalledProcessError as exc:
                print(f"predict_latest failed with exit code {exc.returncode}.")

            elapsed = time.time() - start_time
            sleep_time = max(0.0, args.interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nLoop canceled by user.")


if __name__ == "__main__":
    main()
