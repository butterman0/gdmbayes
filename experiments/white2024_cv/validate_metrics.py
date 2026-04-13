#!/usr/bin/env python3
"""Validate CV metrics against White et al. (2024) Table 1 benchmarks.

Usage:
    python validate_metrics.py <metrics_csv> [--max-rmse-ratio 1.5]

Exits 0 if all configs pass, 1 if any fail, 2 if metrics file missing/incomplete.
Prints per-config validation results to stdout.
"""
import argparse
import csv
import os
import sys

BENCHMARKS_CSV = os.path.join(
    os.path.dirname(__file__), "..", "..", "benchmarks", "white2024_table1.csv"
)

# Map our config_tag → White et al. (dataset, spatial_effect, variance)
# so we can look up the benchmark RMSE for comparison.


def load_benchmarks():
    """Load White et al. Table 1 into a dict keyed by (dataset, spatial_effect, variance)."""
    benchmarks = {}
    with open(BENCHMARKS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["dataset"], row["spatial_effect"], row["variance"])
            benchmarks[key] = {
                "RMSE": float(row["RMSE"]) if row["RMSE"] else None,
                "MAE": float(row["MAE"]) if row["MAE"] else None,
                "CRPS": float(row["CRPS"]) if row["CRPS"] else None,
            }
    return benchmarks


# Map dataset names in our metrics CSV to White et al. dataset names
DATASET_MAP = {
    "Panama": "Panama",
    "SW Australia": "SW Australia",
    "GCFR": "GCFR Family",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_csv", help="Path to CV metrics CSV")
    parser.add_argument(
        "--max-rmse-ratio", type=float, default=1.5,
        help="Max ratio of our RMSE to White's RMSE before flagging (default 1.5 = 50%% tolerance)"
    )
    parser.add_argument(
        "--min-configs", type=int, default=9,
        help="Minimum number of completed configs expected (default 9)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.metrics_csv):
        print(f"MISSING: {args.metrics_csv}")
        sys.exit(2)

    benchmarks = load_benchmarks()

    rows = []
    with open(args.metrics_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) < args.min_configs:
        print(f"INCOMPLETE: {len(rows)}/{args.min_configs} configs in {args.metrics_csv}")
        sys.exit(2)

    all_pass = True
    bad_configs = []

    for row in rows:
        our_dataset = row["dataset"]
        white_dataset = DATASET_MAP.get(our_dataset, our_dataset)
        spatial = row["spatial_effect"]
        variance = row["variance"]
        tag = row["config_tag"]
        our_rmse = float(row["RMSE_CV"])
        our_mae = float(row["MAE_CV"])
        our_crps = float(row["CRPS_CV"])

        key = (white_dataset, spatial, variance)
        bench = benchmarks.get(key)

        if bench is None or bench["RMSE"] is None:
            # No benchmark (e.g., polynomial for Panama/SW) — just check sanity
            # RMSE should be less than Naive baseline
            naive_key = (white_dataset, "", "")
            naive = benchmarks.get(naive_key)
            if naive and naive["RMSE"] and our_rmse > naive["RMSE"] * 2:
                print(f"FAIL  {tag:40s} RMSE={our_rmse:.4f} (>2x Naive={naive['RMSE']:.4f})")
                all_pass = False
                bad_configs.append(tag)
            else:
                print(f"OK    {tag:40s} RMSE={our_rmse:.4f} MAE={our_mae:.4f} CRPS={our_crps:.4f} (no benchmark)")
            continue

        ratio = our_rmse / bench["RMSE"]
        status = "OK" if ratio <= args.max_rmse_ratio else "FAIL"
        if status == "FAIL":
            all_pass = False
            bad_configs.append(tag)

        print(
            f"{status:5s} {tag:40s} "
            f"RMSE={our_rmse:.4f} (White={bench['RMSE']:.4f}, ratio={ratio:.2f})  "
            f"MAE={our_mae:.4f} (White={bench['MAE']:.4f})  "
            f"CRPS={our_crps:.4f}" + (f" (White={bench['CRPS']:.4f})" if bench['CRPS'] else "")
        )

    if all_pass:
        print(f"\nPASSED: All {len(rows)} configs within tolerance.")
        sys.exit(0)
    else:
        print(f"\nFAILED: {len(bad_configs)} configs exceeded tolerance: {bad_configs}")
        sys.exit(1)


if __name__ == "__main__":
    main()
