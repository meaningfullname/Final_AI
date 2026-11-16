"""Project configuration: paths, column names and experiment settings."""
from pathlib import Path

# ---------- paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
SQL_RESULTS_DIR = RESULTS_DIR / "sql"
MODELS_DIR = BASE_DIR / "models"

DATA_URL = "https://archive.ics.uci.edu/static/public/280/higgs.zip"
ZIP_PATH = DATA_DIR / "higgs.zip"
CSV_GZ_PATH = DATA_DIR / "HIGGS.csv.gz"
CSV_PATH = DATA_DIR / "HIGGS.csv"          # also accepted if already unpacked
PARQUET_PATH = DATA_DIR / "higgs.parquet"
DB_PATH = DATA_DIR / "higgs.db"
STATE_PATH = RESULTS_DIR / "state.json"

# ---------- columns ----------
# The raw file has no header: column 0 is the label, then 28 features.
LOW_LEVEL = ["lepton_pt", "lepton_eta", "lepton_phi",
             "missing_energy_magnitude", "missing_energy_phi"]
for _i in range(1, 5):
    LOW_LEVEL += [f"jet{_i}_pt", f"jet{_i}_eta", f"jet{_i}_phi", f"jet{_i}_btag"]
HIGH_LEVEL = ["m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"]

FEATURES = LOW_LEVEL + HIGH_LEVEL
LABEL = "label"
COLUMNS = [LABEL] + FEATURES

# b-tag columns take only a few discrete values, so they are not scaled
BTAG_FEATURES = [f"jet{i}_btag" for i in range(1, 5)]
# quantities that cannot be negative (used in the data quality audit)
NON_NEGATIVE = (["lepton_pt", "missing_energy_magnitude"]
                + [f"jet{i}_pt" for i in range(1, 5)] + BTAG_FEATURES + HIGH_LEVEL)

FEATURE_SETS = {"low_level": LOW_LEVEL, "high_level": HIGH_LEVEL, "all": FEATURES}

# ---------- data processing ----------
CHUNK_SIZE = 500_000          # rows per CSV chunk = rows per Parquet row group
TEST_ROWS = 500_000           # the last 500K rows are the test set (dataset convention)
TEST_FRACTION_SMALL = 0.10    # used instead when only part of the file was converted
EDA_SAMPLE = 200_000
BENCHMARK_ROWS = 1_000_000    # rows used in the CSV vs Parquet reading benchmark

# ---------- database ----------
DB_TRAIN_EVENTS = 100_000     # events from the train part stored in SQLite
DB_TEST_EVENTS = 100_000      # events from the test part stored in SQLite (with predictions)

# ---------- modelling ----------
RANDOM_STATE = 42
TUNING_ROWS = 100_000         # subsample used for hyperparameter search (3-fold CV)
LEARNING_CURVE_SIZES = [10_000, 30_000, 100_000, 300_000, 1_000_000]
# To study even larger samples add e.g. 3_000_000 (training takes much longer).
RF_MAX_TRAIN = 300_000        # Random Forest is not trained above this size (time limit)
FEATURE_SET_TRAIN = 300_000   # train size for the low-level / high-level / all comparison
IMPORTANCE_ROWS = 20_000      # test rows used for permutation importance
