import argparse
import csv
import json
import math
import random
from itertools import combinations_with_replacement
from pathlib import Path
from typing import List, Tuple

try:
    import numpy as np
except ImportError:
    np = None


DEFAULT_CSV_PATHS = [
    "run_001_out.csv",
    "run_002_out.csv",
    "run_003_out.csv",
    "run_004_out.csv",
    "run_005_out.csv",
    "run_006_out.csv",
    "run_007_out.csv",
    "run_008_out.csv",
    "run_009_out.csv",
    "run_010_out.csv",
    "run_011_out.csv",
    "run_012_out.csv",
    "run_013_out.csv",
    "run_014_out.csv",
    "run_015_out.csv",
    "run_016_out.csv",
    "run_017_out.csv",
    "run_018_out.csv",
    "run_019_out.csv",
    "run_020_out.csv",
    "run_021_out.csv",
    "run_022_out.csv",
    "run_023_out.csv",
    "run_024_out.csv",
    "run_025_out.csv",
    "run_026_out.csv",
    "run_027_out.csv",
    "run_028_out.csv",
    "run_029_out.csv",
    "run_030_out.csv",
    "run_031_out.csv",
    "run_032_out.csv",
    "run_033_out.csv",
    "run_034_out.csv",
    "run_035_out.csv",
    "run_036_out.csv",
    "run_037_out.csv",
    "run_038_out.csv",
    "run_039_out.csv",
    "run_040_out.csv",
    "run_041_out.csv",
    "run_042_out.csv",
    "run_043_out.csv",
    "run_044_out.csv",
]
INPUT_WINDOW = 6
FORECAST_HORIZON = 20
LABEL_INDICES = (25, 26)
EXPECTED_COLUMN_COUNT = 27
POLY_FEATURE_COUNT_WARNING = 50000


def estimate_polynomial_feature_count(num_features: int, degree: int) -> int:
    if degree <= 1 or num_features <= 0:
        return num_features
    total = num_features
    for k in range(2, degree + 1):
        total += math.comb(num_features + k - 1, k)
    return total


def read_csv_rows(csv_path: Path) -> List[List[float]]:
    rows: List[List[float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.reader(csv_file)
        for row_number, row in enumerate(reader, start=1):
            if not row:
                continue
            if len(row) > EXPECTED_COLUMN_COUNT:
                extra = row[EXPECTED_COLUMN_COUNT:]
                if all(cell == "" for cell in extra):
                    row = row[:EXPECTED_COLUMN_COUNT]
                else:
                    print(
                        f"Warning: {csv_path.name} line {row_number} has {len(row)} columns "
                        f"(expected {EXPECTED_COLUMN_COUNT}), skipping.",
                    )
                    continue
            elif len(row) != EXPECTED_COLUMN_COUNT:
                print(
                    f"Warning: {csv_path.name} line {row_number} has {len(row)} columns "
                    f"(expected {EXPECTED_COLUMN_COUNT}), skipping.",
                )
                continue
            try:
                rows.append([float(value) for value in row])
            except ValueError:
                print(
                    f"Warning: {csv_path.name} line {row_number} contains nonnumeric data, skipping.",
                )
    return rows


def make_instances(
    rows: List[List[float]],
    input_window: int,
    forecast_horizon: int,
    label_indices: Tuple[int, int],
    combine_outputs: bool = False,
) -> List[Tuple[List[float], List[float]]]:
    instances: List[Tuple[List[float], List[float]]] = []
    total_window = input_window + forecast_horizon
    if len(rows) < total_window:
        return instances

    for start in range(0, len(rows) - total_window + 1):
        input_slice = rows[start : start + input_window]
        last_input_row = input_slice[-1]
        future_row = rows[start + input_window + forecast_horizon - 1]

        features = [value for row in input_slice for value in row]
        output = [
            future_row[label_indices[0]] - last_input_row[label_indices[0]],
            future_row[label_indices[1]] - last_input_row[label_indices[1]],
        ]
        if combine_outputs:
            output = [output[0] + output[1]]
        instances.append((features, output))

    return instances


def split_train_test_by_file(
    file_instances: List[List[Tuple[List[float], List[float]]]],
    test_fraction: float,
) -> Tuple[List[List[float]], List[List[float]], List[List[float]], List[List[float]]]:
    train_X: List[List[float]] = []
    train_y: List[List[float]] = []
    test_X: List[List[float]] = []
    test_y: List[List[float]] = []

    for instances in file_instances:
        count = len(instances)
        if count == 0:
            continue
        test_count = max(1, int(round(count * test_fraction)))
        # Shuffle the instances to randomize selection
        shuffled_instances = instances[:]
        random.shuffle(shuffled_instances)
        # Take the first test_count as test, rest as train
        for i, (features, output) in enumerate(shuffled_instances):
            if i < test_count:
                test_X.append(features)
                test_y.append(output)
            else:
                train_X.append(features)
                train_y.append(output)

    return train_X, train_y, test_X, test_y


def filter_features(
    X: List[List[float]], y: List[List[float]], exclude_indices: List[int]
) -> Tuple[List[List[float]], List[List[float]]]:
    """Filter out specified feature indices from X and y."""
    if not exclude_indices:
        return X, y
    exclude_set = set(exclude_indices)
    X_filtered = [[val for j, val in enumerate(row) if j not in exclude_set] for row in X]
    return X_filtered, y


def generate_polynomial_features(features: List[float], degree: int) -> List[float]:
    """Generate polynomial features up to specified degree using itertools.
    
    For degree 3, generates: x_i, x_i*x_j, x_i*x_j*x_k, x_i^2, x_i^2*x_j, x_i^3, etc.
    This is much more efficient than recursive generation.
    """
    if degree < 1:
        return features
    
    poly_features: List[float] = features[:]
    
    # Generate all combinations of feature indices with replacement up to degree
    num_features = len(features)
    
    for current_degree in range(2, degree + 1):
        # combinations_with_replacement generates all sorted tuples of indices
        for combo in combinations_with_replacement(range(num_features), current_degree):
            product = 1.0
            for idx in combo:
                product *= features[idx]
            poly_features.append(product)
    
    return poly_features


def transpose(matrix: List[List[float]]) -> List[List[float]]:
    if not matrix:
        return []
    return [list(column) for column in zip(*matrix)]


def dot_product(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def matrix_multiply(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    return [[dot_product(row, col) for col in transpose(b)] for row in a]


def solve_linear_system(a: List[List[float]], b: List[float]) -> List[float]:
    n = len(a)
    augmented = [row[:] + [b_val] for row, b_val in zip(a, b)]

    for i in range(n):
        pivot_row = max(range(i, n), key=lambda r: abs(augmented[r][i]))
        if abs(augmented[pivot_row][i]) < 1e-10:
            raise ValueError("Matrix is singular or nearly singular")
        augmented[i], augmented[pivot_row] = augmented[pivot_row], augmented[i]

        pivot = augmented[i][i]
        augmented[i] = [value / pivot for value in augmented[i]]

        for j in range(n):
            if j == i:
                continue
            factor = augmented[j][i]
            augmented[j] = [
                current - factor * pivot_value
                for current, pivot_value in zip(augmented[j], augmented[i])
            ]

    return [row[-1] for row in augmented]


def fit_linear_regression_numpy(X: List[List[float]], y: List[List[float]], alpha: float = 1e-2) -> Tuple[List[List[float]], List[List[float]]]:
    X_arr = np.array(X, dtype=float)
    y_arr = np.array(y, dtype=float)
    X_design = np.hstack([np.ones((X_arr.shape[0], 1), dtype=float), X_arr])
    Gram = X_design.T @ X_design
    Gram[np.diag_indices_from(Gram)] += alpha
    target = X_design.T @ y_arr

    try:
        beta = np.linalg.solve(Gram, target)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(Gram) @ target

    if y_arr.ndim == 1:
        beta = beta.reshape(-1, 1)
        y_arr = y_arr.reshape(-1, 1)

    weights = [beta[:, i].tolist() for i in range(beta.shape[1])]

    predictions = X_design @ beta
    residuals = y_arr - predictions
    degrees_of_freedom = max(X_design.shape[0] - X_design.shape[1], 1)
    sigma_squared = np.sum(residuals**2, axis=0) / degrees_of_freedom
    inv_Gram = np.linalg.pinv(Gram)
    diag = np.diag(inv_Gram)
    std_errors = [np.sqrt(np.maximum(sigma_squared[i] * diag, 0.0)).tolist() for i in range(beta.shape[1])]
    return weights, std_errors


def fit_linear_regression(X: List[List[float]], y: List[List[float]], alpha: float = 1e-2) -> Tuple[List[List[float]], List[List[float]]]:
    if np is not None:
        return fit_linear_regression_numpy(X, y, alpha)

    if not X or not y:
        raise ValueError("Cannot fit model without data")

    X_design = [[1.0] + row for row in X]
    X_design_T = transpose(X_design)
    num_features = len(X_design_T)
    gram_matrix = [
        [dot_product(col_i, col_j) for col_j in X_design_T]
        for col_i in X_design_T
    ]

    weights: List[List[float]] = []
    std_errors: List[List[float]] = []
    for target_column in transpose(y):
        target_vector = [dot_product(col, target_column) for col in X_design_T]
        regularized = [row[:] for row in gram_matrix]
        for diag_index in range(num_features):
            regularized[diag_index][diag_index] += alpha
        beta = solve_linear_system(regularized, target_vector)
        weights.append(beta)

        # Compute residuals
        predictions = [dot_product(beta, row) for row in X_design]
        residuals = [actual - pred for actual, pred in zip(target_column, predictions)]
        residual_sum_squares = sum(r**2 for r in residuals)
        degrees_of_freedom = len(X) - num_features
        if degrees_of_freedom <= 0:
            sigma_squared = 0
        else:
            sigma_squared = residual_sum_squares / degrees_of_freedom

        # Compute (X^T X)^-1
        # Since we have gram_matrix which is X^T X, and we solved for beta, but to get inverse is hard.
        # Approximate standard errors using the diagonal of gram_matrix inverse approximation
        # For simplicity, use the diagonal elements as approximation for variance
        se = []
        for i in range(num_features):
            if gram_matrix[i][i] > 0:
                se.append(math.sqrt(sigma_squared / gram_matrix[i][i]))
            else:
                se.append(0.0)
        std_errors.append(se)

    return weights, std_errors


def predict(weights: List[List[float]], features: List[float], polynomial_degree: int = 1) -> List[float]:
    """Predict using weights. If polynomial_degree > 1, generate polynomial features first."""
    if polynomial_degree > 1:
        features_expanded = generate_polynomial_features(features, polynomial_degree)
    else:
        features_expanded = features
    x_design = [1.0] + features_expanded
    return [dot_product(weight_vector, x_design) for weight_vector in weights]


def mean_absolute_error(y_true: List[List[float]], y_pred: List[List[float]]) -> List[float]:
    if not y_true:
        return [0.0]
    num_outputs = len(y_true[0])
    errors = [0.0] * num_outputs
    for actual, predicted in zip(y_true, y_pred):
        for idx in range(num_outputs):
            errors[idx] += abs(actual[idx] - predicted[idx])
    return [errors[idx] / len(y_true) for idx in range(num_outputs)]


def mean_squared_error(y_true: List[List[float]], y_pred: List[List[float]]) -> List[float]:
    if not y_true:
        return [0.0]
    num_outputs = len(y_true[0])
    errors = [0.0] * num_outputs
    for actual, predicted in zip(y_true, y_pred):
        for idx in range(num_outputs):
            diff = actual[idx] - predicted[idx]
            errors[idx] += diff * diff
    return [errors[idx] / len(y_true) for idx in range(num_outputs)]


def analyze_feature_importance(weights: List[List[float]], std_errors: List[List[float]], input_window: int, expected_column_count: int, combine_outputs: bool) -> List[int]:
    print("\nFeature Importance Analysis with Statistical Significance:")
    
    if combine_outputs:
        # Single output
        feature_weights = weights[0][1:]
        feature_se = std_errors[0][1:]
        
        irrelevant_features = []
        for idx in range(len(feature_weights)):
            beta = feature_weights[idx]
            se = feature_se[idx]
            ci_lower = beta - 1.96 * se
            ci_upper = beta + 1.96 * se
            if ci_lower <= 0 <= ci_upper:
                irrelevant_features.append(idx)
        
        print(f"Features deemed irrelevant (95% CI includes 0): {len(irrelevant_features)}")
        for feature_idx in irrelevant_features[:20]:
            row_idx = feature_idx // expected_column_count
            col_idx = feature_idx % expected_column_count
            print(f"  Feature {feature_idx} (row {row_idx}, col {col_idx})")
        if len(irrelevant_features) > 20:
            print(f"  ... and {len(irrelevant_features) - 20} more")
        
        significant_features = [idx for idx in range(len(feature_weights)) if idx not in irrelevant_features]
        print(f"\nFeatures deemed relevant: {len(significant_features)}")
    else:
        # Two outputs
        feature_weights_dropped = weights[0][1:]
        feature_weights_returned = weights[1][1:]
        feature_se_dropped = std_errors[0][1:]
        feature_se_returned = std_errors[1][1:]
        
        irrelevant_features = []
        for idx in range(len(feature_weights_dropped)):
            # For dropped
            beta_d = feature_weights_dropped[idx]
            se_d = feature_se_dropped[idx]
            ci_lower_d = beta_d - 1.96 * se_d
            ci_upper_d = beta_d + 1.96 * se_d
            significant_d = not (ci_lower_d <= 0 <= ci_upper_d)
            
            # For returned
            beta_r = feature_weights_returned[idx]
            se_r = feature_se_returned[idx]
            ci_lower_r = beta_r - 1.96 * se_r
            ci_upper_r = beta_r + 1.96 * se_r
            significant_r = not (ci_lower_r <= 0 <= ci_upper_r)
            
            if not significant_d and not significant_r:
                irrelevant_features.append(idx)
        
        print(f"Features deemed irrelevant (95% CI includes 0 for both outputs): {len(irrelevant_features)}")
        for feature_idx in irrelevant_features[:20]:  # Show first 20
            row_idx = feature_idx // expected_column_count
            col_idx = feature_idx % expected_column_count
            print(f"  Feature {feature_idx} (row {row_idx}, col {col_idx})")
        if len(irrelevant_features) > 20:
            print(f"  ... and {len(irrelevant_features) - 20} more")
        
        # Also show some significant ones
        significant_features = [idx for idx in range(len(feature_weights_dropped)) if idx not in irrelevant_features]
        print(f"\nFeatures deemed relevant: {len(significant_features)}")
    
    return irrelevant_features


def save_model(model_path: Path, weights: List[List[float]]) -> None:
    model_path.write_text(json.dumps({"weights": weights}, indent=2), encoding="utf-8")


def build_dataset(csv_files: List[Path], input_window: int, forecast_horizon: int, combine_outputs: bool) -> Tuple[List[List[float]], List[List[float]], List[List[float]], List[List[float]]]:
    all_instances: List[List[Tuple[List[float], List[float]]]] = []
    for csv_file in csv_files:
        rows = read_csv_rows(csv_file)
        instances = make_instances(rows, input_window, forecast_horizon, LABEL_INDICES, combine_outputs)
        all_instances.append(instances)
        print(f"Loaded {len(rows)} rows from {csv_file.name}, produced {len(instances)} cases.")

    return split_train_test_by_file(all_instances, test_fraction=0.2)


def report_metrics(name: str, y_true: List[List[float]], y_pred: List[List[float]], combine_outputs: bool) -> None:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mean_true = [sum(col) / len(col) for col in transpose(y_true)]
    mean_pred = [sum(col) / len(col) for col in transpose(y_pred)]
    print(f"{name} examples: {len(y_true)}")
    if combine_outputs:
        print(f"  MAE total: {mae[0]:.6f}")
        print(f"  MSE total: {mse[0]:.6f}")
        print(f"  Mean expected total: {mean_true[0]:.6f}")
        print(f"  Mean predicted total: {mean_pred[0]:.6f}")
    else:
        print(f"  MAE dropped: {mae[0]:.6f}, returned: {mae[1]:.6f}")
        print(f"  MSE dropped: {mse[0]:.6f}, returned: {mse[1]:.6f}")
        print(f"  Mean expected dropped: {mean_true[0]:.6f}, returned: {mean_true[1]:.6f}")
        print(f"  Mean predicted dropped: {mean_pred[0]:.6f}, returned: {mean_pred[1]:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a polynomial ridge regression model using all variables from the previous 5 lines "
                    "to predict increases in total dropped and returned packets after 5 seconds."
    )
    parser.add_argument(
        "--data-files",
        nargs="*",
        default=DEFAULT_CSV_PATHS,
        help="CSV files to use for training and testing.",
    )
    parser.add_argument(
        "--save-model",
        default="trained_linear_model.json",
        help="Optional JSON file path to save trained model weights.",
    )
    parser.add_argument(
        "--combine-outputs",
        action="store_true",
        help="Combine dropped and returned outputs into a single sum output for simplified univariate regression.",
    )
    parser.add_argument(
        "--drop-irrelevant",
        action="store_true",
        help="Drop statistically irrelevant features and retrain the model.",
    )
    parser.add_argument(
        "--exclude-columns",
        type=str,
        default="",
        help="Comma-separated list of feature indices to exclude (e.g., '0,5,10,15').",
    )
    parser.add_argument(
        "--polynomial-degree",
        type=int,
        default=1,
        help="Polynomial degree for feature expansion (1=linear, 2=quadratic, 3=cubic, etc.).",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1e-2,
        help="Ridge regularization strength added to the diagonal of X^T X.",
    )
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parent
    csv_paths = [base_path / file_name for file_name in args.data_files]
    train_X, train_y, test_X, test_y = build_dataset(csv_paths, INPUT_WINDOW, FORECAST_HORIZON, args.combine_outputs)

    if not train_X or not test_X:
        raise RuntimeError("Insufficient data to build training/testing datasets.")

    # Apply column exclusion if specified
    exclude_indices: List[int] = []
    if args.exclude_columns.strip():
        try:
            exclude_indices = [int(idx.strip()) for idx in args.exclude_columns.split(",")]
            print(f"Excluding feature indices: {exclude_indices}")
            train_X, train_y = filter_features(train_X, train_y, exclude_indices)
            test_X, test_y = filter_features(test_X, test_y, exclude_indices)
            print(f"Filtered to {len(train_X[0])} features from original {len(train_X[0]) + len(exclude_indices)}")
        except ValueError:
            print("Warning: Invalid --exclude-columns format. Expected comma-separated integers.")

    # Apply polynomial transformation if degree > 1
    if args.polynomial_degree > 1:
        print(f"Expanding features to polynomial degree {args.polynomial_degree}...")
        original_feature_count = len(train_X[0]) if train_X else 0
        expected_feature_count = estimate_polynomial_feature_count(original_feature_count, args.polynomial_degree)
        if expected_feature_count > POLY_FEATURE_COUNT_WARNING:
            print(
                f"Warning: estimated polynomial expansion will create {expected_feature_count} features. "
                "This may be very slow or may not finish. Reduce --polynomial-degree or exclude more features."
            )
        train_X = [generate_polynomial_features(x, args.polynomial_degree) for x in train_X]
        test_X = [generate_polynomial_features(x, args.polynomial_degree) for x in test_X]
        expanded_feature_count = len(train_X[0]) if train_X else 0
        print(f"Expanded from {original_feature_count} to {expanded_feature_count} features")

    weights, std_errors = fit_linear_regression(train_X, train_y, alpha=args.ridge_alpha)
    print("Trained polynomial ridge regression model with feature dimension", len(train_X[0]) + 1)

    irrelevant_features = analyze_feature_importance(weights, std_errors, INPUT_WINDOW, EXPECTED_COLUMN_COUNT, args.combine_outputs)

    if args.drop_irrelevant and irrelevant_features:
        print(f"\nRetraining model after dropping {len(irrelevant_features)} irrelevant features...")
        # Filter out irrelevant features from train_X and test_X
        def filter_irrelevant_features(X: List[List[float]], irrelevant: List[int]) -> List[List[float]]:
            return [[val for j, val in enumerate(row) if j not in irrelevant] for row in X]
        
        train_X_filtered = filter_irrelevant_features(train_X, irrelevant_features)
        test_X_filtered = filter_irrelevant_features(test_X, irrelevant_features)
        
        weights, std_errors = fit_linear_regression(train_X_filtered, train_y, alpha=args.ridge_alpha)
        print("Retrained simplified polynomial ridge regression model with feature dimension", len(train_X_filtered[0]) + 1)
        
        # Update train_X and test_X for evaluation
        train_X, test_X = train_X_filtered, test_X_filtered

    if args.save_model:
        save_model(base_path / args.save_model, weights)
        print(f"Saved trained model weights to {args.save_model}")

    train_predictions = [predict(weights, x, polynomial_degree=1) for x in train_X]
    test_predictions = [predict(weights, x, polynomial_degree=1) for x in test_X]

    report_metrics("Training", train_y, train_predictions, args.combine_outputs)
    report_metrics("Testing", test_y, test_predictions, args.combine_outputs)

    sample_index = min(9, len(test_X) - 1)
    print("\nSample test predictions:")
    if args.combine_outputs:
        print("(total):")
        for i in range(sample_index + 1):
            print(
                f"  expected={test_y[i][0]:.4f}, predicted={test_predictions[i][0]:.4f}"
            )
    else:
        print("(dropped, returned):")
        for i in range(sample_index + 1):
            print(
                f"  expected={test_y[i]}, predicted={[round(value, 4) for value in test_predictions[i]]}"
            )


if __name__ == "__main__":
    main()
