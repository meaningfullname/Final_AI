"""SQLite storage: schema, loading of events and experiment results, running queries."""
import sqlite3
from datetime import datetime

import pandas as pd

from . import config
from .data_loader import read_rows, split_boundaries
from .eda import get_feature_stats
from .sql_queries import QUERIES

MODEL_INFO = {      # name -> (family, is_baseline)
    "dummy": ("baseline: always predicts the class prior", 1),
    "logistic_regression": ("linear model", 0),
    "random_forest": ("bagging of decision trees", 0),
    "lightgbm": ("gradient boosting of decision trees", 0),
}

_FEATURE_COLUMNS = ",\n    ".join(f"{name} REAL" for name in config.FEATURES)
SCHEMA = f"""
CREATE TABLE IF NOT EXISTS features (
    feature_id INTEGER PRIMARY KEY,
    name       TEXT UNIQUE NOT NULL,
    level      TEXT NOT NULL,             -- 'low' or 'high'
    object     TEXT NOT NULL              -- lepton, missing_energy, jet1..4, invariant_mass
);
CREATE TABLE IF NOT EXISTS feature_stats (
    feature_id INTEGER NOT NULL REFERENCES features(feature_id),
    label      INTEGER NOT NULL,          -- 0 background, 1 signal
    n INTEGER, mean REAL, std REAL, min REAL, max REAL,
    PRIMARY KEY (feature_id, label)
);
CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY,         -- row number in the original file
    label    INTEGER NOT NULL,
    split    TEXT NOT NULL,               -- 'train' or 'test'
    {_FEATURE_COLUMNS}
);
CREATE TABLE IF NOT EXISTS models (
    model_id    INTEGER PRIMARY KEY,
    name        TEXT UNIQUE NOT NULL,
    family      TEXT NOT NULL,
    is_baseline INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id       INTEGER NOT NULL REFERENCES models(model_id),
    kind           TEXT NOT NULL,         -- 'learning_curve' or 'feature_set'
    feature_set    TEXT NOT NULL,         -- 'all', 'low_level', 'high_level'
    train_size     INTEGER NOT NULL,
    is_final       INTEGER NOT NULL,      -- 1 = largest run of the model (saved to disk)
    roc_auc REAL, pr_auc REAL, accuracy REAL, f1 REAL,
    train_time_s REAL, predict_time_s REAL, model_size_mb REAL,
    created_at     TEXT
);
CREATE TABLE IF NOT EXISTS predictions (
    experiment_id INTEGER NOT NULL REFERENCES experiments(experiment_id),
    event_id      INTEGER NOT NULL REFERENCES events(event_id),
    proba         REAL NOT NULL,          -- predicted probability of signal
    PRIMARY KEY (experiment_id, event_id)
);
CREATE TABLE IF NOT EXISTS feature_importance (
    experiment_id   INTEGER NOT NULL REFERENCES experiments(experiment_id),
    feature_id      INTEGER NOT NULL REFERENCES features(feature_id),
    importance_mean REAL, importance_std REAL,
    PRIMARY KEY (experiment_id, feature_id)
);
"""


def connect():
    connection = sqlite3.connect(config.DB_PATH)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def feature_object(name):
    """Physical object a feature belongs to (used in the `features` table)."""
    if name in config.HIGH_LEVEL:
        return "invariant_mass"
    if name.startswith("jet"):
        return name.split("_")[0]
    if name.startswith("missing"):
        return "missing_energy"
    return "lepton"


def db_test_event_range():
    """Row range of the test events that are stored in SQLite."""
    n_train, total = split_boundaries()
    return n_train, min(total, n_train + config.DB_TEST_EVENTS)


def build_database():
    """Create the schema and load features, feature statistics, models and events."""
    with connect() as con:
        con.executescript(SCHEMA)
        # reference tables are rebuilt from scratch every time
        for table in ("predictions", "feature_importance", "experiments",
                      "feature_stats", "events", "features", "models"):
            con.execute(f"DELETE FROM {table}")

        feature_ids = {}
        for feature_id, name in enumerate(config.FEATURES, start=1):
            level = "high" if name in config.HIGH_LEVEL else "low"
            con.execute("INSERT INTO features VALUES (?, ?, ?, ?)",
                        (feature_id, name, level, feature_object(name)))
            feature_ids[name] = feature_id

        for row in get_feature_stats():
            con.execute("INSERT INTO feature_stats VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (feature_ids[row["feature"]], row["label"], row["n"], row["mean"],
                         row["std"], row["min"], row["max"]))

        for model_id, (name, (family, is_baseline)) in enumerate(MODEL_INFO.items(), start=1):
            con.execute("INSERT INTO models VALUES (?, ?, ?, ?)",
                        (model_id, name, family, is_baseline))

        n_train, _ = split_boundaries()
        test_start, test_stop = db_test_event_range()
        parts = {"train": read_rows(0, min(n_train, config.DB_TRAIN_EVENTS)),
                 "test": read_rows(test_start, test_stop)}
        for split, df in parts.items():
            df = df.copy()
            df.insert(0, "event_id", df.index)
            df.insert(2, "split", split)
            df.to_sql("events", con, if_exists="append", index=False, chunksize=20_000)
        con.execute("CREATE INDEX IF NOT EXISTS idx_events_split ON events(split)")
        counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                  for t in ("features", "feature_stats", "models", "events")}
    return counts


def ensure_database():
    """Build the database if it does not exist yet (used by the training steps)."""
    if not config.DB_PATH.exists():
        build_database()
        return
    with connect() as con:
        con.executescript(SCHEMA)
        empty = con.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
    if empty:
        build_database()


def clear_experiments(kind):
    """Delete previous experiments of one kind together with their child rows."""
    with connect() as con:
        ids = "SELECT experiment_id FROM experiments WHERE kind = ?"
        con.execute(f"DELETE FROM predictions WHERE experiment_id IN ({ids})", (kind,))
        con.execute(f"DELETE FROM feature_importance WHERE experiment_id IN ({ids})", (kind,))
        con.execute("DELETE FROM experiments WHERE kind = ?", (kind,))


def insert_experiment(record):
    """Insert one experiment (dict with model name and metrics); return its id."""
    with connect() as con:
        model_id = con.execute("SELECT model_id FROM models WHERE name = ?",
                               (record["model"],)).fetchone()[0]
        cursor = con.execute(
            """INSERT INTO experiments (model_id, kind, feature_set, train_size, is_final,
                   roc_auc, pr_auc, accuracy, f1, train_time_s, predict_time_s,
                   model_size_mb, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (model_id, record["kind"], record["feature_set"], record["train_size"],
             record.get("is_final", 0), record["roc_auc"], record["pr_auc"],
             record["accuracy"], record["f1"], record["train_time_s"],
             record["predict_time_s"], record["model_size_mb"],
             datetime.now().isoformat(timespec="seconds")))
        return cursor.lastrowid


def mark_final(experiment_id):
    with connect() as con:
        con.execute("UPDATE experiments SET is_final = 1 WHERE experiment_id = ?",
                    (experiment_id,))


def insert_predictions(experiment_id, event_ids, probabilities):
    rows = [(experiment_id, int(e), float(p)) for e, p in zip(event_ids, probabilities)]
    with connect() as con:
        con.executemany("INSERT OR REPLACE INTO predictions VALUES (?, ?, ?)", rows)


def insert_importance(experiment_id, importance):
    with connect() as con:
        ids = dict(con.execute("SELECT name, feature_id FROM features").fetchall())
        con.execute("DELETE FROM feature_importance WHERE experiment_id = ?", (experiment_id,))
        con.executemany(
            "INSERT INTO feature_importance VALUES (?, ?, ?, ?)",
            [(experiment_id, ids[r.feature], float(r.importance_mean), float(r.importance_std))
             for r in importance.itertuples()])


def run_query(sql, params=()):
    """Run a SELECT and return the result as a pandas DataFrame."""
    with connect() as con:
        return pd.read_sql_query(sql, con, params=params)


def run_all_queries():
    """Run every analytical query, save each result to CSV, return {name: DataFrame}."""
    results = {}
    for name, (description, sql) in QUERIES.items():
        try:
            df = run_query(sql)
        except (sqlite3.Error, pd.errors.DatabaseError) as error:
            print(f"[{name}] failed: {error}")
            continue
        df.to_csv(config.SQL_RESULTS_DIR / f"{name}.csv", index=False)
        results[name] = (description, df)
    return results
