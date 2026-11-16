"""Interactive prediction: the user picks or types an event and gets a prediction."""
import numpy as np
import pandas as pd

from . import config
from .data_loader import read_rows, split_boundaries
from .models import load_best_model


def predict_event(model, values):
    """Probability of signal for one event given as a list of 28 feature values."""
    frame = pd.DataFrame([values], columns=config.FEATURES, dtype=np.float32)
    return float(model.predict_proba(frame)[0, 1])


def parse_manual_input(text):
    """Turn '0.87, -0.63, ...' into a list of 28 floats (ValueError if it is not)."""
    values = [float(part) for part in text.replace(";", ",").split(",") if part.strip()]
    if len(values) != len(config.FEATURES):
        raise ValueError(f"expected {len(config.FEATURES)} numbers, got {len(values)}")
    return values


def show_prediction(proba, true_label=None):
    verdict = "SIGNAL (Higgs boson process)" if proba >= 0.5 else "BACKGROUND"
    print(f"  Probability of signal: {proba:.4f}  ->  {verdict}")
    if true_label is not None:
        truth = "signal" if true_label == 1 else "background"
        correct = (proba >= 0.5) == (true_label == 1)
        print(f"  True class: {truth}  ->  the model is {'right' if correct else 'wrong'}")


def prediction_menu():
    """Sub-menu: random test event, test event by number, or manual input."""
    try:
        info, model = load_best_model()
    except FileNotFoundError as error:
        print(error)
        return
    n_train, total = split_boundaries()
    print(f"Using model '{info['model']}' (test AUC {info['roc_auc']:.4f}).")

    while True:
        print("\n  1 - Random event from the test set"
              "\n  2 - Test event by its number"
              "\n  3 - Type 28 feature values manually"
              "\n  0 - Back")
        choice = input("  Choose: ").strip()
        try:
            if choice == "0":
                return
            elif choice in ("1", "2"):
                if choice == "1":
                    event_id = int(np.random.default_rng().integers(n_train, total))
                else:
                    event_id = int(input(f"  Event number ({n_train}..{total - 1}): "))
                    if not n_train <= event_id < total:
                        print("  This number is outside the test set.")
                        continue
                row = read_rows(event_id, event_id + 1).iloc[0]
                print(f"  Event {event_id}: "
                      + ", ".join(f"{f}={row[f]:.3f}" for f in config.HIGH_LEVEL))
                show_prediction(predict_event(model, row[config.FEATURES].tolist()),
                                int(row[config.LABEL]))
            elif choice == "3":
                print("  Order: " + ", ".join(config.FEATURES))
                values = parse_manual_input(input("  28 numbers separated by commas: "))
                show_prediction(predict_event(model, values))
            else:
                print("  Unknown option.")
        except ValueError as error:
            print(f"  Invalid input: {error}")
