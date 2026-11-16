"""Data quality audit (whole dataset, incremental) and exploratory analysis (sample)."""
import numpy as np
import pandas as pd

from . import config, plots
from .data_loader import iter_row_groups, sample_train
from .utils import load_json, save_json


def data_quality_audit():
    """Scan the whole Parquet file group by group and collect quality statistics.

    Nothing is loaded fully into memory: counters and per-class sums are
    updated incrementally, row hashes are kept to count duplicated rows.
    """
    audit = {"rows": 0, "missing": 0, "infinite": 0, "bad_labels": 0,
             "label_counts": {"0": 0, "1": 0}, "negative_values": {}}
    acc = {}                       # (feature, label) -> running n, sum, sum of squares, min, max
    hashes = []

    for _, df in iter_row_groups():
        audit["rows"] += len(df)
        audit["missing"] += int(df.isna().sum().sum())
        audit["infinite"] += int(np.isinf(df[config.FEATURES].to_numpy()).sum())
        audit["bad_labels"] += int((~df[config.LABEL].isin([0, 1])).sum())
        hashes.append(pd.util.hash_pandas_object(df, index=False).to_numpy())

        for feature in config.NON_NEGATIVE:
            count = int((df[feature] < 0).sum())
            if count:
                audit["negative_values"][feature] = audit["negative_values"].get(feature, 0) + count

        for label in (0, 1):
            part = df.loc[df[config.LABEL] == label, config.FEATURES].astype(np.float64)
            audit["label_counts"][str(label)] += len(part)
            for feature in config.FEATURES:
                values = part[feature]
                a = acc.setdefault((feature, label),
                                   {"n": 0, "sum": 0.0, "sumsq": 0.0, "min": np.inf, "max": -np.inf})
                a["n"] += len(values)
                a["sum"] += float(values.sum())
                a["sumsq"] += float((values ** 2).sum())
                a["min"] = min(a["min"], float(values.min()))
                a["max"] = max(a["max"], float(values.max()))

    all_hashes = np.concatenate(hashes)
    audit["duplicate_rows"] = int(len(all_hashes) - len(np.unique(all_hashes)))
    audit["signal_share"] = round(audit["label_counts"]["1"] / audit["rows"], 4)

    stats = []
    for (feature, label), a in acc.items():
        mean = a["sum"] / a["n"] if a["n"] else 0.0
        variance = max(a["sumsq"] / a["n"] - mean ** 2, 0.0) if a["n"] else 0.0
        stats.append({"feature": feature, "label": label, "n": a["n"], "mean": mean,
                      "std": variance ** 0.5, "min": a["min"], "max": a["max"]})

    save_json(audit, config.RESULTS_DIR / "data_quality_audit.json")
    save_json(stats, config.RESULTS_DIR / "feature_stats.json")
    return audit, stats


def get_feature_stats():
    """Per-class feature statistics; computed by the audit if not saved yet."""
    stats = load_json(config.RESULTS_DIR / "feature_stats.json")
    if stats is None:
        _, stats = data_quality_audit()
    return stats


def outlier_report(sample):
    """Share of values outside the 1.5*IQR whiskers for every feature."""
    rows = []
    for feature in config.FEATURES:
        q1, q3 = sample[feature].quantile([0.25, 0.75])
        iqr = q3 - q1
        outside = ((sample[feature] < q1 - 1.5 * iqr) | (sample[feature] > q3 + 1.5 * iqr)).mean()
        n_unique = sample[feature].nunique()
        rows.append({"feature": feature,
                     "level": "high" if feature in config.HIGH_LEVEL else "low",
                     "type": "discrete" if n_unique <= 10 else "continuous",
                     "unique_values": int(n_unique), "q1": q1, "q3": q3,
                     "outlier_share": round(float(outside), 4)})
    return pd.DataFrame(rows)


def run_eda():
    """Audit + descriptive statistics + figures. Returns a short text summary."""
    print("Data quality audit over the whole file ...")
    audit, _ = data_quality_audit()
    print(f"  rows: {audit['rows']:,} | missing: {audit['missing']} | infinite: {audit['infinite']}"
          f" | duplicate rows: {audit['duplicate_rows']:,} | signal share: {audit['signal_share']:.2%}")
    if audit["negative_values"]:
        print(f"  negative values in non-negative columns: {audit['negative_values']}")

    sample = sample_train(config.EDA_SAMPLE)
    print(f"EDA on a random sample of {len(sample):,} train rows ...")
    sample[config.FEATURES].describe().T.to_csv(config.RESULTS_DIR / "descriptive_stats.csv")
    outliers = outlier_report(sample)
    outliers.to_csv(config.RESULTS_DIR / "outlier_report.csv", index=False)

    plots.plot_class_balance({int(k): v for k, v in audit["label_counts"].items()})
    plots.plot_distributions(sample, config.HIGH_LEVEL, "distributions_high_level.png",
                             "High-level features (invariant masses)")
    plots.plot_distributions(sample, config.LOW_LEVEL, "distributions_low_level.png",
                             "Low-level features (kinematics of detected particles)")
    corr = plots.plot_correlation(sample)
    plots.plot_boxplots(sample)
    plots.interactive_distributions(sample)

    label_corr = corr[config.LABEL].drop(config.LABEL).sort_values(key=abs, ascending=False)
    label_corr.rename("correlation_with_label").to_csv(
        config.RESULTS_DIR / "correlation_with_label.csv", index_label="feature")
    print("  strongest linear correlation with the label:")
    for feature, value in label_corr.head(5).items():
        print(f"    {feature:<28}{value:+.3f}")
    top_outliers = outliers.sort_values("outlier_share", ascending=False).head(3)
    print("  features with most IQR outliers: "
          + ", ".join(f"{r.feature} ({r.outlier_share:.1%})" for r in top_outliers.itertuples()))
    return audit
