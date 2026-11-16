# Higgs Signal vs Background Classification

Final project for the course **Programming for Data Analysis and Database**
(MSc Applied Artificial Intelligence, Astana IT University). Topic 38, Cluster 7:
Machine Learning on Tabular Data.

**Author:** Nurislam Toktarov, group AAI-2602

## Task

The [UCI HIGGS dataset](https://archive.ics.uci.edu/dataset/280/higgs) contains
11 million simulated particle collisions. Each row has a label
(1 = a process that produces a Higgs boson, "signal"; 0 = "background") and
28 numeric features: 21 low-level measurements of the detected particles and
7 high-level features (invariant masses) derived from them by physicists.

The project answers three questions:

1. How to process a file that does not fit comfortably in memory (2.6 GB compressed)?
2. How much training data is actually needed? (learning curves, subsample justification)
3. Which model gives the best trade-off between quality (ROC-AUC), training time and model size?

## Project structure

```
main.py               console menu and command-line interface
src/config.py         paths, column names, experiment settings
src/utils.py          timer, JSON files, saved program state
src/data_loader.py    download, chunked CSV -> Parquet, out-of-core reading and sampling
src/eda.py            data quality audit and exploratory analysis
src/plots.py          all figures (matplotlib + interactive Plotly)
src/database.py       SQLite schema, loading, query runner
src/sql_queries.py    12 analytical SQL queries
src/models.py         pipelines, tuning, learning curves, interpretation
src/predict.py        interactive prediction for a single event
docs/er_diagram.md    ER diagram of the database
```

## How to run

Python 3.10 or newer.

```bash
git clone <repository url>
cd higgs-classification
pip install -r requirements.txt
python main.py            # interactive menu
```

Without the menu:

```bash
python main.py --all                    # whole pipeline on the full dataset
python main.py --all --rows 1000000     # quick run on the first 1M rows
python main.py --step sql               # a single step
python main.py --all --no-tuning        # skip the hyperparameter search
```

### Google Colab

```
!git clone <repository url>
%cd higgs-classification
!pip install -q -r requirements.txt
!python main.py --all
```

The dataset is downloaded automatically by step 1. If the download fails, get
`higgs.zip` from the dataset page, and put `HIGGS.csv.gz` into the `data/` folder.

### Menu

```
1 - Download the dataset (2.6 GB)
2 - Convert CSV to Parquet in chunks + memory report
3 - Data quality audit and EDA figures
4 - Build the SQLite database
5 - Tune and train models, learning curves
6 - Feature-set comparison, importance, error analysis
7 - Run the analytical SQL queries
8 - Predict a single event
9 - Run steps 1-7 one after another
0 - Exit
```

Completed steps are remembered in `results/state.json` and marked in the menu.

## Method

**Data.** The CSV is read in chunks of 500K rows. Features are converted from
float64 to float32 and the label to int8, and every chunk is written as one
Parquet row group. Later steps read only the row groups they need.

**Split without leakage.** The last 500,000 rows are the test set (the split used
by the dataset authors). Training samples are drawn only from the rows before
them. Imputation and scaling are steps of the scikit-learn pipeline, so their
statistics are learned from training rows only. Hyperparameters are chosen by
3-fold cross-validation inside the training part; the test set is used once per
trained model.

**Models.**

| Model | Role |
|---|---|
| DummyClassifier (class prior) | baseline |
| Logistic Regression | linear model |
| Random Forest | bagging of trees |
| LightGBM | gradient boosting |

**Comparisons.**

- Loading: CSV with default dtypes (baseline) vs CSV with float32 vs Parquet (time, memory).
- Models: ROC-AUC, PR-AUC, accuracy, F1, training time, prediction time, model size.
- Learning curves: the same models on 10K, 30K, 100K, 300K and 1M training rows.
- Feature sets: low-level only vs high-level only vs all 28 features.

**Interpretation.** Permutation importance, confusion matrix, ROC curves,
calibration by deciles (SQL), analysis of events that every model gets wrong (SQL).

## Database

SQLite file `data/higgs.db` with 7 tables. See [docs/er_diagram.md](docs/er_diagram.md).
The 12 queries in `src/sql_queries.py` use JOIN, GROUP BY / HAVING, subqueries,
CTEs and window functions (RANK, LAG, NTILE, MAX OVER). Results are loaded into
pandas and saved to `results/sql/`.

## Outputs

| Path | Content |
|---|---|
| `results/conversion_report.json` | rows, memory before and after dtype optimisation, file sizes |
| `results/format_benchmark.csv` | CSV vs Parquet reading time and memory |
| `results/data_quality_audit.json` | missing values, duplicates, invalid values, class balance |
| `results/descriptive_stats.csv`, `outlier_report.csv` | descriptive statistics and IQR outliers |
| `results/best_params.json` | best hyperparameters and cross-validation AUC |
| `results/experiments.csv` | every training run with metrics, time and size |
| `results/feature_sets.csv` | low-level vs high-level vs all features |
| `results/permutation_importance.csv`, `error_analysis.csv` | interpretation |
| `results/figures/` | 10 PNG figures and 2 interactive Plotly HTML files |
| `results/sql/` | result of every SQL query |
| `models/` | trained models (joblib) |

## Results

To be filled in after the full run: final metrics table, learning-curve figure
and the main conclusions.

## Limitations

- The data is simulated, not recorded by a real detector.
- Random Forest is trained on at most 300K rows because of training time
  (`RF_MAX_TRAIN` in `src/config.py`).
- Only a 200K-event sample is stored in SQLite; the full data stays in Parquet.

## References

- Baldi, P., Sadowski, P., Whiteson, D. Searching for exotic particles in
  high-energy physics with deep learning. *Nature Communications* 5, 4308 (2014).
- Whiteson, D. HIGGS dataset. UCI Machine Learning Repository (2014).
  https://archive.ics.uci.edu/dataset/280/higgs
