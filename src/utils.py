"""Small helpers shared by all modules: timing, JSON files, program state."""
import json
import time
from datetime import datetime

from . import config


class Timer:
    """Context manager that measures wall-clock time in seconds."""

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start
        return False


def ensure_dirs():
    """Create all output folders if they do not exist yet."""
    for folder in (config.DATA_DIR, config.RESULTS_DIR, config.FIGURES_DIR,
                   config.SQL_RESULTS_DIR, config.MODELS_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def memory_mb(df):
    """Memory used by a DataFrame in megabytes."""
    return float(df.memory_usage(deep=True).sum()) / 1024 ** 2


def file_mb(path):
    """File size in megabytes (0 if the file does not exist)."""
    return path.stat().st_size / 1024 ** 2 if path.exists() else 0.0


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path, default=None):
    """Read a JSON file; return `default` if it is missing or broken."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def mark_done(step):
    """Remember in results/state.json that a pipeline step was completed."""
    state = load_json(config.STATE_PATH, default={})
    state[step] = datetime.now().isoformat(timespec="seconds")
    save_json(state, config.STATE_PATH)


def is_done(step):
    return step in load_json(config.STATE_PATH, default={})


def print_table(df, max_rows=30):
    """Print a DataFrame without truncating columns."""
    import pandas as pd
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", "{:.4f}".format):
        print(df.head(max_rows).to_string(index=False))
