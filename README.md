# ML Drift Monitor

Compare numeric reference and current batches with PSI, two-sample KS, and missingness-change summaries.

> **Portfolio prototype.** Source code is provided; no benchmark results are claimed until you run it on the cited data and record the output.

## Use case

Compare numeric reference and current batches with PSI, two-sample KS, and missingness-change summaries. The CLI consumes local files, emits reproducible reports, and keeps data out of the repository.

## Data and provenance

Bring a frozen training/reference batch and a current production batch from your authorized pipeline.. Two CSV files with matching numeric feature columns. Do not mix different populations or schema versions without documenting the change. Source files are not bundled. Respect source terms and cite the dataset when publishing results.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
# Download/prepare the source data locally, then run:
python main.py --reference data/reference.csv --current data/current.csv --out outputs
```

## Outputs

Generated results are saved under `outputs/`; file names and metrics are printed or documented by the CLI. The repository excludes downloaded data and artifacts.

## Evaluation and operating notes

Drift signals do not prove model degradation or reveal its cause. Freeze a representative reference set, correct for multiple comparisons, then validate performance against labeled outcomes.

For production use, add a frozen reference period, out-of-time evaluation, monitoring ownership, and a documented rollback or escalation path. This project is a local baseline, not a deployed service.

## License

No code license file is included yet. Third-party data remains subject to its source terms.
