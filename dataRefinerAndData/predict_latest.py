#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List

from ai_model import INPUT_WINDOW, EXPECTED_COLUMN_COUNT, read_csv_rows, predict


def find_latest_csv(directory: Path) -> Path:
    csv_files = sorted(
        (path for path in directory.glob("*.csv") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    return csv_files[0]


def load_model_weights(model_path: Path) -> List[List[float]]:
    with model_path.open("r", encoding="utf-8") as model_file:
        content = json.load(model_file)
    weights = content.get("weights")
    if weights is None:
        raise ValueError(f"Model file {model_path} does not contain a 'weights' field")
    return weights


def build_features_from_rows(rows: List[List[float]], window_size: int) -> List[float]:
    if len(rows) < window_size:
        raise ValueError(
            f"Not enough rows to build input features: {len(rows)} available, {window_size} required"
        )
    window_rows = rows[-window_size:]
    return [value for row in window_rows for value in row]


def risk_level(first_value: float) -> str:
    if first_value < 10:
        return "Low Risk"
    if first_value < 30:
        return "Moderate Risk"
    if first_value < 50:
        return "High Risk"
    return "Severe Risk"


def format_prediction(output_values: List[float]) -> str:
    if not output_values:
        return "No prediction output available."

    risk = risk_level(output_values[0])
    #if len(output_values) == 1:
    #    return f"Predicted output: {output_values[0]:.6f}\nRisk level: {risk}"

    lines = []
    #for index, value in enumerate(output_values, start=1):
    #    lines.append(f"  output[{index}] = {value:.6f}")
    lines.append(f"Risk level: {risk}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load the latest CSV data, run the saved linear model, and print predictions."
    )
    parser.add_argument(
        "--csv",
        help="Optional path to a CSV file. If omitted, the most recently modified CSV in the script directory is used.",
    )
    parser.add_argument(
        "--model",
        default="trained_linear_model.json",
        help="Path to the saved JSON model weights file.",
    )
    parser.add_argument(
        "--polynomial-degree",
        type=int,
        default=1,
        help="Polynomial degree for feature expansion when using the saved model weights.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=INPUT_WINDOW,
        help="Number of most recent rows to use as input features.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    model_path = (base_dir / args.model).resolve() if not Path(args.model).is_absolute() else Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    csv_path = Path(args.csv).resolve() if args.csv else find_latest_csv(base_dir)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    rows = read_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"No valid numeric rows found in {csv_path}")
    if len(rows) < args.window_size:
        raise ValueError(
            f"CSV file contains {len(rows)} rows, but window size {args.window_size} is required"
        )

    features = build_features_from_rows(rows, args.window_size)
    weights = load_model_weights(model_path)
    prediction = predict(weights, features, polynomial_degree=args.polynomial_degree)
    print(format_prediction(prediction))


if __name__ == "__main__":
    main()
