# ML Drift Monitor

Compare a frozen reference distribution with a later batch using reference-quantile PSI, two-sample Kolmogorov–Smirnov statistics, and missingness rates. The output flags PSI at a configurable threshold; it does not claim that model quality changed.

## Real-data benchmark

Uses the UCI Bike Sharing hourly dataset (dataset 275). Rows from 2011 form the reference (8,645 rows); 2012 forms the later comparison (8,734 rows). The original raw file SHA-256 is \`f52d7db575d4d8dd4633ab1e39d501e8cc77fc9e2f3352802afbfd0e8cb5bafa\`. The run monitors continuous weather variables and hourly ride count \`cnt\`; categorical calendar codes and the \`yr\` split marker are excluded.

| Column | PSI | KS statistic | PSI alert at 0.20 |
|---|---:|---:|---|
| temp | 0.036 | 0.055 | No |
| atemp | 0.033 | 0.046 | No |
| hum | 0.048 | 0.065 | No |
| windspeed | 0.004 | 0.019 | No |
| cnt (target) | 0.244 | 0.216 | Yes |

The target distribution shifted above the heuristic PSI threshold while the four continuous weather inputs did not. This signals a distribution change for investigation; it does not establish why demand changed or whether a deployed model's error increased. The KS p-values are unadjusted and are not the alert rule. See [aggregate benchmark](reports/benchmark.json).

## Reproduce

Download UCI's [Bike Sharing dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), then split its hourly \`hour.csv\` by calendar year and run:

\`\`\`bash
python main.py --reference reference_2011.csv --current current_2012.csv \\
  --columns temp,atemp,hum,windspeed,cnt \\
  --reference-label "2011 training year" --current-label "2012 later year" \\
  --psi-threshold 0.2 --out outputs
\`\`\`

Use a frozen, representative reference and compare compatible schemas/populations. PSI cutoffs are heuristics; validate any alert with domain owners and labeled outcomes. A production monitor should track data quality, model performance, ownership, and rollback separately. Raw source data and row-level results are not committed.
