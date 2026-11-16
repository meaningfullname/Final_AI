"""Models: preprocessing pipeline, tuning, learning curves, comparison, interpretation."""
import pickle

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config, database, plots
from .data_loader import load_test, sample_train, split_boundaries
from .utils import Timer, load_json, save_json

MODEL_NAMES = ["dummy", "logistic_regression", "random_forest", "lightgbm"]
BEST_PARAMS_PATH = config.RESULTS_DIR / "best_params.json"
BEST_MODEL_PATH = config.MODELS_DIR / "best_model.json"
PREDICTIONS_PATH = config.RESULTS_DIR / "test_predictions.parquet"

# hyperparameter search spaces (keys use the pipeline step name "clf")
SEARCH_SPACES = {
    "logistic_regression": {"clf__C": [0.01, 0.1, 1.0, 10.0]},
    "random_forest": {"clf__max_depth": [8, 12, 16, 20],
                      "clf__min_samples_leaf": [5, 20, 50],
                      "clf__max_features": ["sqrt", 0.5]},
    "lightgbm": {"clf__num_leaves": [31, 63, 127],
                 "clf__learning_rate": [0.05, 0.1],
                 "clf__n_estimators": [300, 600],
                 "clf__min_child_samples": [20, 100],
                 "clf__colsample_bytree": [0.7, 1.0]},
}
SEARCH_ITERATIONS = {"logistic_regression": 4, "random_forest": 4, "lightgbm": 8}


# ---------------------------------------------------------------- building
def build_preprocessor(features):
    """ColumnTransformer: impute + scale continuous columns, only impute b-tag columns.

    The transformer is part of the model pipeline, so its statistics (medians,
    means, standard deviations) are learned from training rows only.
    """
    continuous = [f for f in features if f not in config.BTAG_FEATURES]
    btag = [f for f in features if f in config.BTAG_FEATURES]
    transformers = [("continuous",
                     Pipeline([("impute", SimpleImputer(strategy="median")),
                               ("scale", StandardScaler())]),
                     continuous)]
    if btag:
        transformers.append(("btag", SimpleImputer(strategy="most_frequent"), btag))
    return ColumnTransformer(transformers)


def build_model(name, features=config.FEATURES, params=None):
    """Return an untrained pipeline (preprocessing + classifier) for a model name."""
    if name == "dummy":
        classifier = DummyClassifier(strategy="prior")
    elif name == "logistic_regression":
        classifier = LogisticRegression(max_iter=300)
    elif name == "random_forest":
        classifier = RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_leaf=20,
                                            n_jobs=-1, random_state=config.RANDOM_STATE)
    elif name == "lightgbm":
        classifier = LGBMClassifier(n_estimators=400, learning_rate=0.1, num_leaves=63,
                                    n_jobs=-1, random_state=config.RANDOM_STATE, verbose=-1)
    else:
        raise ValueError(f"Unknown model: {name}")
    pipeline = Pipeline([("prep", build_preprocessor(features)), ("clf", classifier)])
    if params:
        pipeline.set_params(**params)
    return pipeline


def evaluate(model, X, y):
    """Quality metrics on a held-out set plus the prediction time."""
    with Timer() as timer:
        proba = model.predict_proba(X)[:, 1]
    predicted = (proba >= 0.5).astype(int)
    metrics = {"roc_auc": float(roc_auc_score(y, proba)),
               "pr_auc": float(average_precision_score(y, proba)),
               "accuracy": float(accuracy_score(y, predicted)),
               "f1": float(f1_score(y, predicted)),
               "predict_time_s": timer.seconds}
    return metrics, proba


# ---------------------------------------------------------------- tuning
def tune_hyperparameters():
    """Randomized search with 3-fold cross-validation on a train subsample."""
    sample = sample_train(config.TUNING_ROWS, seed=config.RANDOM_STATE + 1)
    X, y = sample[config.FEATURES], sample[config.LABEL]
    best = {}
    for name, space in SEARCH_SPACES.items():
        print(f"  tuning {name} on {len(sample):,} rows ...")
        search = RandomizedSearchCV(build_model(name), space, n_iter=SEARCH_ITERATIONS[name],
                                    cv=3, scoring="roc_auc", random_state=config.RANDOM_STATE)
        with Timer() as timer:
            search.fit(X, y)
        best[name] = {"params": search.best_params_,
                      "cv_roc_auc": round(float(search.best_score_), 4),
                      "seconds": round(timer.seconds, 1)}
        print(f"    best CV AUC {search.best_score_:.4f} with {search.best_params_}")
    save_json(best, BEST_PARAMS_PATH)
    return best


def load_best_params():
    """{model name: params} from the saved search; empty dict means defaults."""
    saved = load_json(BEST_PARAMS_PATH, default={})
    return {name: entry["params"] for name, entry in saved.items()}


# ---------------------------------------------------------------- experiments
def run_experiments():
    """Train every model on growing training sets and test on the fixed test set.

    Gives the learning curves; the largest run of each model is the final one
    (saved to disk, predictions stored in SQLite).
    """
    database.ensure_database()
    database.clear_experiments("learning_curve")
    params = load_best_params()

    n_train, _ = split_boundaries()
    sizes = [s for s in config.LEARNING_CURVE_SIZES if s <= n_train] or [n_train]
    train = sample_train(max(sizes))          # shuffled once; smaller sets are its prefixes
    test = load_test()
    X_test, y_test = test[config.FEATURES], test[config.LABEL]
    print(f"  train pool: {len(train):,} rows | test: {len(test):,} rows")

    records, final = [], {}
    for size in sizes:
        subset = train.iloc[:size]
        for name in MODEL_NAMES:
            if name == "dummy" and size != sizes[-1]:
                continue                       # the baseline does not depend on train size
            if name == "random_forest" and size > config.RF_MAX_TRAIN:
                continue
            model = build_model(name, params=params.get(name))
            with Timer() as timer:
                model.fit(subset[config.FEATURES], subset[config.LABEL])
            metrics, proba = evaluate(model, X_test, y_test)
            record = {"model": name, "kind": "learning_curve", "feature_set": "all",
                      "train_size": size, "train_time_s": timer.seconds,
                      "model_size_mb": len(pickle.dumps(model)) / 1024 ** 2, **metrics}
            record["experiment_id"] = database.insert_experiment(record)
            records.append(record)
            final[name] = (record, model, proba)
            print(f"  {name:<20} rows={size:>9,}  AUC={metrics['roc_auc']:.4f}  "
                  f"train={timer.seconds:6.1f}s  size={record['model_size_mb']:.2f} MB")

    # the last (largest) run of every model is its final version
    _, db_stop = database.db_test_event_range()
    in_db = test.index < db_stop
    predictions = pd.DataFrame({config.LABEL: y_test})
    for name, (record, model, proba) in final.items():
        record["is_final"] = 1
        database.mark_final(record["experiment_id"])
        database.insert_predictions(record["experiment_id"], test.index[in_db], proba[in_db])
        joblib.dump(model, config.MODELS_DIR / f"{name}.joblib")
        predictions[name] = proba.astype(np.float32)
    predictions.to_parquet(PREDICTIONS_PATH)

    results = pd.DataFrame(records)
    results.to_csv(config.RESULTS_DIR / "experiments.csv", index=False)
    best = max((r for r, _, _ in final.values()), key=lambda r: r["roc_auc"])
    save_json({"model": best["model"], "experiment_id": best["experiment_id"],
               "train_size": best["train_size"], "roc_auc": best["roc_auc"],
               "features": config.FEATURES}, BEST_MODEL_PATH)
    plots.plot_learning_curves(results)
    return results


def run_feature_sets():
    """Compare low-level, high-level and all features with the same model (LightGBM)."""
    database.ensure_database()
    database.clear_experiments("feature_set")
    params = load_best_params().get("lightgbm")
    train = sample_train(config.FEATURE_SET_TRAIN, seed=config.RANDOM_STATE + 2)
    test = load_test()

    records = []
    for set_name, features in config.FEATURE_SETS.items():
        model = build_model("lightgbm", features=features, params=params)
        with Timer() as timer:
            model.fit(train[features], train[config.LABEL])
        metrics, _ = evaluate(model, test[features], test[config.LABEL])
        record = {"model": "lightgbm", "kind": "feature_set", "feature_set": set_name,
                  "train_size": len(train), "train_time_s": timer.seconds,
                  "model_size_mb": len(pickle.dumps(model)) / 1024 ** 2, **metrics}
        database.insert_experiment(record)
        records.append(record)
        print(f"  {set_name:<11} ({len(features):>2} features)  AUC={metrics['roc_auc']:.4f}")
    results = pd.DataFrame(records)
    results.to_csv(config.RESULTS_DIR / "feature_sets.csv", index=False)
    plots.plot_feature_sets(results)
    return results


# ---------------------------------------------------------------- interpretation
def load_best_model():
    """Return (info dict, fitted pipeline) of the best final model."""
    info = load_json(BEST_MODEL_PATH)
    if info is None:
        raise FileNotFoundError("No trained model found. Run the training step first.")
    return info, joblib.load(config.MODELS_DIR / f"{info['model']}.joblib")


def interpret_best_model():
    """Permutation importance, ROC curves, confusion matrix and error analysis."""
    info, model = load_best_model()
    test = load_test()
    predictions = pd.read_parquet(PREDICTIONS_PATH)
    y_true = predictions[config.LABEL].to_numpy()
    proba = predictions[info["model"]].to_numpy()
    predicted = (proba >= 0.5).astype(int)

    plots.plot_roc_curves(y_true, {name: predictions[name] for name in MODEL_NAMES
                                   if name in predictions})
    matrix = plots.plot_confusion(y_true, predicted, info["model"])

    # permutation importance on a random part of the test set
    part = test.sample(n=min(config.IMPORTANCE_ROWS, len(test)), random_state=config.RANDOM_STATE)
    result = permutation_importance(model, part[config.FEATURES], part[config.LABEL],
                                    scoring="roc_auc", n_repeats=5,
                                    random_state=config.RANDOM_STATE)
    importance = pd.DataFrame({"feature": config.FEATURES,
                               "importance_mean": result.importances_mean,
                               "importance_std": result.importances_std})
    importance = importance.sort_values("importance_mean", ascending=False)
    importance.to_csv(config.RESULTS_DIR / "permutation_importance.csv", index=False)
    database.insert_importance(info["experiment_id"], importance)
    plots.plot_importance(importance, info["model"])

    # error analysis: what do wrongly classified events look like?
    outcome = np.select(
        [(y_true == 1) & (predicted == 1), (y_true == 0) & (predicted == 0),
         (y_true == 0) & (predicted == 1)],
        ["true_positive", "true_negative", "false_positive"], default="false_negative")
    table = test[config.HIGH_LEVEL].copy()
    table["outcome"] = outcome
    table["proba"] = proba
    errors = table.groupby("outcome").mean()
    errors.insert(0, "events", table.groupby("outcome").size())
    errors.to_csv(config.RESULTS_DIR / "error_analysis.csv")

    wrong = predicted != y_true
    confident_wrong = wrong & ((proba >= 0.9) | (proba <= 0.1))
    summary = {"model": info["model"],
               "confusion_matrix": matrix.tolist(),
               "error_rate": round(float(wrong.mean()), 4),
               "confident_error_share_of_errors": round(float(confident_wrong.sum() / max(wrong.sum(), 1)), 4),
               "top_features": importance["feature"].head(5).tolist()}
    save_json(summary, config.RESULTS_DIR / "interpretation_summary.json")
    return summary, importance, errors
