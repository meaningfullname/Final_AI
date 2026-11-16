# ER diagram of the SQLite database (`data/higgs.db`)

```mermaid
erDiagram
    features ||--o{ feature_stats : "described by"
    features ||--o{ feature_importance : "scored in"
    models ||--o{ experiments : "trained in"
    experiments ||--o{ predictions : "produces"
    experiments ||--o{ feature_importance : "explains"
    events ||--o{ predictions : "receives"

    features {
        INTEGER feature_id PK
        TEXT name
        TEXT level "low or high"
        TEXT object "lepton, jet1..4, missing_energy, invariant_mass"
    }
    feature_stats {
        INTEGER feature_id PK, FK
        INTEGER label PK "0 background, 1 signal"
        INTEGER n
        REAL mean
        REAL std
        REAL min
        REAL max
    }
    events {
        INTEGER event_id PK "row number in the original file"
        INTEGER label
        TEXT split "train or test"
        REAL lepton_pt "... 28 feature columns"
    }
    models {
        INTEGER model_id PK
        TEXT name
        TEXT family
        INTEGER is_baseline
    }
    experiments {
        INTEGER experiment_id PK
        INTEGER model_id FK
        TEXT kind "learning_curve or feature_set"
        TEXT feature_set "all, low_level, high_level"
        INTEGER train_size
        INTEGER is_final
        REAL roc_auc
        REAL pr_auc
        REAL accuracy
        REAL f1
        REAL train_time_s
        REAL predict_time_s
        REAL model_size_mb
        TEXT created_at
    }
    predictions {
        INTEGER experiment_id PK, FK
        INTEGER event_id PK, FK
        REAL proba "predicted probability of signal"
    }
    feature_importance {
        INTEGER experiment_id PK, FK
        INTEGER feature_id PK, FK
        REAL importance_mean
        REAL importance_std
    }
```

## Tables

| Table | Rows | Purpose |
|---|---|---|
| `features` | 28 | Reference table of features with their level and physical object |
| `feature_stats` | 56 | Mean, std, min, max of every feature per class, computed over the whole dataset |
| `events` | up to 200,000 | Sample of events: first 100K train rows and first 100K test rows |
| `models` | 4 | Reference table of models, including the baseline |
| `experiments` | one per training run | Metrics, time and model size of every run |
| `predictions` | final models x stored test events | Predicted probability of signal |
| `feature_importance` | 28 | Permutation importance of the best model |

Only a sample of events is stored in SQLite. The full 11M rows stay in Parquet,
which is the right format for column-wise numeric reads; SQLite holds the
relational part of the project (reference tables, experiments, predictions).
