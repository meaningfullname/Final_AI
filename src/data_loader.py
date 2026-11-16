"""Downloading the dataset, chunked CSV -> Parquet conversion and out-of-core reading."""
import urllib.error
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import config
from .utils import Timer, file_mb, memory_mb, save_json


# ---------------------------------------------------------------- download
def find_raw_file():
    """Return the path of the raw CSV (gzipped or plain) or None."""
    for path in (config.CSV_GZ_PATH, config.CSV_PATH):
        if path.exists():
            return path
    return None


def download_dataset():
    """Download higgs.zip from UCI (about 2.6 GB) and extract HIGGS.csv.gz."""
    if find_raw_file() is not None:
        print(f"Raw file already exists: {find_raw_file()}")
        return find_raw_file()

    if not config.ZIP_PATH.exists():
        print(f"Downloading {config.DATA_URL} ...")
        try:
            with urllib.request.urlopen(config.DATA_URL) as response, \
                    open(config.ZIP_PATH, "wb") as out:
                done, next_report = 0, 100
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    out.write(block)
                    done += len(block)
                    if done / 1024 ** 2 >= next_report:
                        print(f"  {done / 1024 ** 2:,.0f} MB downloaded")
                        next_report += 100
        except (urllib.error.URLError, OSError) as error:
            config.ZIP_PATH.unlink(missing_ok=True)   # do not keep a broken file
            print(f"Download failed: {error}")
            print("Download the archive manually from "
                  "https://archive.ics.uci.edu/dataset/280/higgs "
                  f"and put HIGGS.csv.gz into {config.DATA_DIR}")
            return None

    print("Extracting the archive ...")
    with zipfile.ZipFile(config.ZIP_PATH) as archive:
        members = [m for m in archive.namelist() if m.lower().endswith((".csv.gz", ".csv"))]
        if not members:
            print("No CSV file found inside the archive.")
            return None
        member = members[0]
        target = config.CSV_GZ_PATH if member.lower().endswith(".gz") else config.CSV_PATH
        with archive.open(member) as src, open(target, "wb") as dst:
            while True:
                block = src.read(1024 * 1024)
                if not block:
                    break
                dst.write(block)
    print(f"Saved {target} ({file_mb(target):,.0f} MB)")
    return target


# ---------------------------------------------------------------- conversion
def optimize_dtypes(chunk):
    """float64 -> float32 for features, int8 for the label (halves the memory)."""
    chunk[config.FEATURES] = chunk[config.FEATURES].astype(np.float32)
    chunk[config.LABEL] = chunk[config.LABEL].astype(np.int8)
    return chunk


def convert_to_parquet(max_rows=None):
    """Read the CSV in chunks, optimise dtypes and write one Parquet file.

    Each chunk becomes one Parquet row group, so later we can read any part
    of the data without loading the whole file into memory.
    """
    raw = find_raw_file()
    if raw is None:
        raise FileNotFoundError("Raw HIGGS file not found. Run the download step first.")

    report = {"rows": 0, "chunks": 0, "memory_before_mb": 0.0, "memory_after_mb": 0.0}
    writer = None
    with Timer() as timer:
        reader = pd.read_csv(raw, header=None, names=config.COLUMNS,
                             chunksize=config.CHUNK_SIZE, nrows=max_rows)
        for chunk in reader:
            report["memory_before_mb"] += memory_mb(chunk)
            chunk = optimize_dtypes(chunk)
            report["memory_after_mb"] += memory_mb(chunk)
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(config.PARQUET_PATH, table.schema,
                                          compression="snappy")
            writer.write_table(table)
            report["rows"] += len(chunk)
            report["chunks"] += 1
            print(f"  chunk {report['chunks']}: {report['rows']:,} rows converted")
    if writer is not None:
        writer.close()

    report["seconds"] = round(timer.seconds, 1)
    report["raw_file_mb"] = round(file_mb(raw), 1)
    report["parquet_file_mb"] = round(file_mb(config.PARQUET_PATH), 1)
    report["memory_before_mb"] = round(report["memory_before_mb"], 1)
    report["memory_after_mb"] = round(report["memory_after_mb"], 1)
    report["memory_saved_pct"] = round(
        100 * (1 - report["memory_after_mb"] / report["memory_before_mb"]), 1)
    save_json(report, config.RESULTS_DIR / "conversion_report.json")
    return report


def benchmark_formats():
    """Baseline vs two approaches for loading the same rows (time and memory).

    baseline : CSV with default dtypes (float64)
    approach1: CSV with optimised dtypes
    approach2: Parquet
    """
    raw = find_raw_file()
    n = min(config.BENCHMARK_ROWS, num_rows())
    dtypes = {c: np.float32 for c in config.FEATURES}
    results = []

    with Timer() as t:
        df = pd.read_csv(raw, header=None, names=config.COLUMNS, nrows=n)
    results.append({"method": "CSV, default dtypes (baseline)", "rows": n,
                    "seconds": round(t.seconds, 2), "memory_mb": round(memory_mb(df), 1)})
    baseline_sum = float(df[config.FEATURES[0]].sum())

    with Timer() as t:
        df = pd.read_csv(raw, header=None, names=config.COLUMNS, nrows=n, dtype=dtypes)
    results.append({"method": "CSV, float32 dtypes", "rows": n,
                    "seconds": round(t.seconds, 2), "memory_mb": round(memory_mb(df), 1)})

    with Timer() as t:
        df = read_rows(0, n)
    results.append({"method": "Parquet", "rows": n,
                    "seconds": round(t.seconds, 2), "memory_mb": round(memory_mb(df), 1)})

    # the two formats must contain the same data
    same = bool(np.isclose(baseline_sum, float(df[config.FEATURES[0]].sum()), rtol=1e-4))
    out = pd.DataFrame(results)
    out["same_result_as_baseline"] = same
    out.to_csv(config.RESULTS_DIR / "format_benchmark.csv", index=False)
    return out


# ---------------------------------------------------------------- reading
def num_rows():
    return pq.ParquetFile(config.PARQUET_PATH).metadata.num_rows


def split_boundaries():
    """Return (n_train, n_total). Rows [0, n_train) are train, the rest is test.

    The split is positional and fixed, so the test rows are never seen in training.
    """
    total = num_rows()
    if total > 10 * config.TEST_ROWS:
        n_test = config.TEST_ROWS
    else:                                   # only a part of the file was converted
        n_test = int(total * config.TEST_FRACTION_SMALL)
    return total - n_test, total


def iter_row_groups(columns=None):
    """Yield (start_row, DataFrame) for every row group of the Parquet file."""
    parquet = pq.ParquetFile(config.PARQUET_PATH)
    start = 0
    for i in range(parquet.num_row_groups):
        df = parquet.read_row_group(i, columns=columns).to_pandas()
        yield start, df
        start += len(df)


def read_rows(start, stop, columns=None):
    """Read rows [start, stop) touching only the row groups that contain them."""
    parquet = pq.ParquetFile(config.PARQUET_PATH)
    parts, offset = [], 0
    for i in range(parquet.num_row_groups):
        size = parquet.metadata.row_group(i).num_rows
        lo, hi = max(start, offset), min(stop, offset + size)
        if lo < hi:
            df = parquet.read_row_group(i, columns=columns).to_pandas()
            part = df.iloc[lo - offset: hi - offset]
            part.index = range(lo, hi)        # global row number = event id
            parts.append(part)
        offset += size
        if offset >= stop:
            break
    return pd.concat(parts)


def sample_train(n, seed=config.RANDOM_STATE):
    """Random sample of n train rows, built group by group (never loads all rows)."""
    n_train, _ = split_boundaries()
    n = min(n, n_train)
    rng = np.random.default_rng(seed)
    parts = []
    for start, df in iter_row_groups():
        if start >= n_train:
            break
        df.index = range(start, start + len(df))
        df = df.iloc[: max(0, n_train - start)]           # cut off test rows
        take = min(len(df), int(np.ceil(n * len(df) / n_train)))
        parts.append(df.iloc[np.sort(rng.choice(len(df), size=take, replace=False))])
    sample = pd.concat(parts)
    return sample.sample(n=min(n, len(sample)), random_state=seed)   # shuffle + trim


def load_test():
    n_train, total = split_boundaries()
    return read_rows(n_train, total)
