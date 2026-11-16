"""Higgs signal vs background classification - console application.

Run `python main.py` for the interactive menu or `python main.py --all`
to execute the whole pipeline without questions.
"""
import argparse
import sys

from src import config, data_loader, database, eda, models, predict
from src.utils import ensure_dirs, is_done, load_json, mark_done, print_table

MENU = [
    ("1", "download", "Download the dataset (2.6 GB)"),
    ("2", "convert", "Convert CSV to Parquet in chunks + memory report"),
    ("3", "eda", "Data quality audit and EDA figures"),
    ("4", "database", "Build the SQLite database"),
    ("5", "train", "Tune and train models, learning curves"),
    ("6", "interpret", "Feature-set comparison, importance, error analysis"),
    ("7", "sql", "Run the analytical SQL queries"),
    ("8", "predict", "Predict a single event"),
    ("9", "all", "Run steps 1-7 one after another"),
    ("0", "exit", "Exit"),
]


# ---------------------------------------------------------------- steps
def step_download(options):
    return data_loader.download_dataset() is not None


def step_convert(options):
    report = data_loader.convert_to_parquet(max_rows=options.get("rows"))
    print(f"Converted {report['rows']:,} rows in {report['chunks']} chunks, {report['seconds']} s")
    print(f"Memory of the data: {report['memory_before_mb']:,.0f} MB (float64) -> "
          f"{report['memory_after_mb']:,.0f} MB (float32/int8), saved {report['memory_saved_pct']}%")
    print(f"File size: raw {report['raw_file_mb']:,.0f} MB -> Parquet {report['parquet_file_mb']:,.0f} MB")
    print("Reading benchmark (baseline vs two approaches):")
    print_table(data_loader.benchmark_formats())
    return True


def step_eda(options):
    eda.run_eda()
    return True


def step_database(options):
    if options.get("interactive") and config.DB_PATH.exists():
        answer = input("The database will be rebuilt and saved experiments removed. Continue? (y/n): ")
        if answer.strip().lower() != "y":
            return False
    counts = database.build_database()
    print("Rows loaded: " + ", ".join(f"{table}={n:,}" for table, n in counts.items()))
    print("ER diagram: docs/er_diagram.md")
    return True


def step_train(options):
    if options.get("tune", True):
        print("Hyperparameter search (3-fold CV on a train subsample) ...")
        models.tune_hyperparameters()
    else:
        print("Tuning skipped: using saved parameters or defaults.")
    print("Training on growing training sets ...")
    results = models.run_experiments()
    final = results[results["is_final"] == 1]
    print("\nFinal models (test set):")
    print_table(final[["model", "train_size", "roc_auc", "pr_auc", "accuracy", "f1",
                       "train_time_s", "model_size_mb"]].sort_values("roc_auc", ascending=False))
    return True


def step_interpret(options):
    print("Feature sets (LightGBM) ...")
    models.run_feature_sets()
    print("Interpretation of the best model ...")
    summary, importance, errors = models.interpret_best_model()
    print(f"Best model: {summary['model']}, error rate {summary['error_rate']:.2%}")
    print("Top features: " + ", ".join(summary["top_features"]))
    print("Mean high-level features by outcome:")
    print_table(errors.reset_index())
    return True


def step_sql(options):
    results = database.run_all_queries()
    for name, (description, df) in results.items():
        print(f"\n[{name}] {description}")
        if df.empty:
            print("  (no rows yet - run the training steps first)")
        else:
            print_table(df)
    print(f"\n{len(results)} queries executed, results saved to results/sql/")
    return True


def step_predict(options):
    predict.prediction_menu()
    return False            # nothing to mark as done


STEPS = {"download": step_download, "convert": step_convert, "eda": step_eda,
         "database": step_database, "train": step_train, "interpret": step_interpret,
         "sql": step_sql, "predict": step_predict}
PIPELINE = ["download", "convert", "eda", "database", "train", "interpret", "sql"]
NEEDS_PARQUET = {"eda", "database", "train", "interpret", "sql", "predict"}


def run_step(name, options):
    """Run one step with error handling; return True if it finished successfully."""
    if name in NEEDS_PARQUET and not config.PARQUET_PATH.exists():
        print("Parquet file not found. Run steps 1 and 2 first.")
        return False
    print(f"\n=== {name.upper()} ===")
    try:
        finished = STEPS[name](options)
    except FileNotFoundError as error:
        print(f"File not found: {error}")
        return False
    except MemoryError:
        print("Not enough memory. Reduce the sizes in src/config.py or use --rows.")
        return False
    except KeyboardInterrupt:
        print("\nStep interrupted by the user.")
        return False
    if finished:
        mark_done(name)
    return bool(finished)


def run_pipeline(options):
    for name in PIPELINE:
        if name == "download" and data_loader.find_raw_file() is not None:
            print("\n=== DOWNLOAD ===\nRaw file already present, skipping.")
            continue
        if not run_step(name, options):
            print(f"Pipeline stopped at step '{name}'.")
            return False
    print("\nPipeline finished. See the results/ folder.")
    return True


# ---------------------------------------------------------------- menu
def show_menu():
    print("\n" + "=" * 62)
    print(" Higgs Signal vs Background Classification")
    print("=" * 62)
    state = load_json(config.STATE_PATH, default={})
    for key, name, title in MENU:
        if name in state:
            status = f"[done {state[name][:16].replace('T', ' ')}]"
        else:
            status = ""
        print(f" {key} - {title} {status}")


def interactive_menu():
    actions = {key: name for key, name, _ in MENU}
    options = {"interactive": True, "tune": True, "rows": None}
    while True:
        show_menu()
        choice = input("Choose an option: ").strip()
        if choice not in actions:
            print("Unknown option, please type a number from the menu.")
            continue
        name = actions[choice]
        if name == "exit":
            print("Bye!")
            break
        elif name == "all":
            run_pipeline({**options, "interactive": False})
        elif name == "convert":
            text = input("How many rows to convert? (Enter = whole file): ").strip()
            try:
                options["rows"] = int(text.replace("_", "").replace(",", "")) if text else None
            except ValueError:
                print("Not a number - the whole file will be converted.")
                options["rows"] = None
            run_step(name, options)
        elif name == "train":
            if is_done("train"):
                answer = input("Repeat the hyperparameter search? (y/n): ").strip().lower()
                options["tune"] = answer == "y"
            run_step(name, options)
        else:
            run_step(name, options)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="run the whole pipeline (steps 1-7)")
    parser.add_argument("--step", choices=PIPELINE, help="run a single step and exit")
    parser.add_argument("--rows", type=int, default=None,
                        help="convert only the first N rows (quick run)")
    parser.add_argument("--no-tuning", action="store_true", help="skip hyperparameter search")
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    ensure_dirs()
    options = {"interactive": False, "tune": not arguments.no_tuning, "rows": arguments.rows}
    if arguments.all:
        sys.exit(0 if run_pipeline(options) else 1)
    elif arguments.step:
        sys.exit(0 if run_step(arguments.step, options) else 1)
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
