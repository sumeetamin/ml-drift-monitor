"""Compare numeric reference and current batches with PSI, KS, and missingness."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def population_stability_index(reference, current, bins=10):
    """Calculate PSI using quantile bins fitted only on reference values."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    reference_share = np.histogram(reference, bins=edges)[0] / len(reference)
    current_share = np.histogram(current, bins=edges)[0] / len(current)
    epsilon = 1e-6
    reference_share = np.clip(reference_share, epsilon, None)
    current_share = np.clip(current_share, epsilon, None)
    return float(np.sum((current_share - reference_share) * np.log(current_share / reference_share)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--columns", help="Comma-separated numeric columns; defaults to numeric intersection")
    parser.add_argument("--psi-threshold", type=float, default=0.2)
    parser.add_argument("--reference-label", default="reference")
    parser.add_argument("--current-label", default="current")
    parser.add_argument("--source-sha256")
    parser.add_argument("--dataset")
    parser.add_argument("--source-url")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    reference, current = pd.read_csv(args.reference), pd.read_csv(args.current)
    columns = args.columns.split(",") if args.columns else sorted(
        set(reference.select_dtypes(include="number")) & set(current.select_dtypes(include="number"))
    )
    absent = sorted(set(columns) - set(reference.columns) | (set(columns) - set(current.columns)))
    if absent:
        raise SystemExit(f"Columns missing from one or both files: {absent}")

    report = {
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "reference_label": args.reference_label,
        "current_label": args.current_label,
        "psi_threshold": args.psi_threshold,
        "columns": {},
    }
    for column in columns:
        x = pd.to_numeric(reference[column], errors="coerce").dropna().to_numpy()
        y = pd.to_numeric(current[column], errors="coerce").dropna().to_numpy()
        if len(x) < 20 or len(y) < 20:
            continue
        psi = population_stability_index(x, y)
        ks = ks_2samp(x, y)
        report["columns"][column] = {
            "psi": psi,
            "ks_statistic": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "reference_missing_rate": float(reference[column].isna().mean()),
            "current_missing_rate": float(current[column].isna().mean()),
            "alert": psi >= args.psi_threshold,
        }
    report["drifted_columns"] = [name for name, value in report["columns"].items() if value["alert"]]
    if args.source_sha256:
        report["source_sha256"] = args.source_sha256
    if args.dataset:
        report["dataset"] = args.dataset
    if args.source_url:
        report["source_url"] = args.source_url
    report["limitations"] = (
        "One temporal comparison is descriptive. PSI thresholds are heuristics; KS p-values are not corrected "
        "for multiple comparisons. Distribution shift alone does not establish model quality degradation or its cause."
    )

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "drift_report.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame([{"column": name, **values} for name, values in report["columns"].items()]).to_csv(
        output / "drift_report.csv", index=False
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
