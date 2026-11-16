"""Analytical SQL queries. Each entry: name -> (what it answers, SQL text)."""

QUERIES = {
    "q01_feature_separation": (
        "Which features differ most between signal and background? (two JOINs on feature_stats)",
        """
        SELECT f.name, f.level,
               ROUND(s.mean, 4) AS mean_signal,
               ROUND(b.mean, 4) AS mean_background,
               ROUND(ABS(s.mean - b.mean) / ((s.std + b.std) / 2.0), 4) AS separation
        FROM features f
        JOIN feature_stats s ON s.feature_id = f.feature_id AND s.label = 1
        JOIN feature_stats b ON b.feature_id = f.feature_id AND b.label = 0
        ORDER BY separation DESC
        LIMIT 10
        """),

    "q02_class_balance_by_split": (
        "Is the class balance the same in the train and test parts? (GROUP BY)",
        """
        SELECT split, COUNT(*) AS events,
               SUM(label) AS signal_events,
               ROUND(AVG(label), 4) AS signal_share
        FROM events
        GROUP BY split
        """),

    "q03_btag_signal_share": (
        "Signal share for every b-tag value of the leading jet (GROUP BY + HAVING)",
        """
        SELECT ROUND(jet1_btag, 2) AS jet1_btag, COUNT(*) AS events,
               ROUND(AVG(label), 4) AS signal_share
        FROM events
        GROUP BY ROUND(jet1_btag, 2)
        HAVING COUNT(*) >= 100
        ORDER BY jet1_btag
        """),

    "q04_model_leaderboard": (
        "Final models ranked by AUC with the gain over the baseline (RANK window + subquery)",
        """
        SELECT m.name AS model, e.train_size,
               ROUND(e.roc_auc, 4) AS roc_auc, ROUND(e.f1, 4) AS f1,
               ROUND(e.train_time_s, 1) AS train_time_s,
               ROUND(e.model_size_mb, 2) AS model_size_mb,
               RANK() OVER (ORDER BY e.roc_auc DESC) AS auc_rank,
               ROUND(e.roc_auc - (SELECT MAX(e2.roc_auc)
                                  FROM experiments e2
                                  JOIN models m2 ON m2.model_id = e2.model_id
                                  WHERE m2.is_baseline = 1), 4) AS gain_vs_baseline
        FROM experiments e
        JOIN models m ON m.model_id = e.model_id
        WHERE e.is_final = 1
        ORDER BY auc_rank
        """),

    "q05_learning_curve_gain": (
        "How much AUC does each increase of the training set add? (LAG window function)",
        """
        SELECT m.name AS model, e.train_size, ROUND(e.roc_auc, 4) AS roc_auc,
               ROUND(e.roc_auc - LAG(e.roc_auc) OVER (PARTITION BY m.name
                                                      ORDER BY e.train_size), 4) AS auc_gain,
               ROUND(e.train_time_s, 1) AS train_time_s
        FROM experiments e
        JOIN models m ON m.model_id = e.model_id
        WHERE e.kind = 'learning_curve' AND m.is_baseline = 0
        ORDER BY m.name, e.train_size
        """),

    "q06_sufficient_train_size": (
        "Smallest training size that is within 0.005 AUC of the model's best (CTE + window MAX)",
        """
        WITH curve AS (
            SELECT m.name AS model, e.train_size, e.roc_auc, e.train_time_s,
                   MAX(e.roc_auc) OVER (PARTITION BY m.name) AS best_auc
            FROM experiments e
            JOIN models m ON m.model_id = e.model_id
            WHERE e.kind = 'learning_curve' AND m.is_baseline = 0
        )
        SELECT model, MIN(train_size) AS sufficient_train_size,
               ROUND(MAX(best_auc), 4) AS best_auc
        FROM curve
        WHERE best_auc - roc_auc <= 0.005
        GROUP BY model
        ORDER BY best_auc DESC
        """),

    "q07_confusion_matrix": (
        "Confusion matrix of every final model at threshold 0.5 (JOIN of 4 tables + GROUP BY)",
        """
        SELECT m.name AS model,
               SUM(CASE WHEN ev.label = 1 AND p.proba >= 0.5 THEN 1 ELSE 0 END) AS true_positive,
               SUM(CASE WHEN ev.label = 0 AND p.proba <  0.5 THEN 1 ELSE 0 END) AS true_negative,
               SUM(CASE WHEN ev.label = 0 AND p.proba >= 0.5 THEN 1 ELSE 0 END) AS false_positive,
               SUM(CASE WHEN ev.label = 1 AND p.proba <  0.5 THEN 1 ELSE 0 END) AS false_negative
        FROM predictions p
        JOIN experiments e ON e.experiment_id = p.experiment_id
        JOIN models m      ON m.model_id = e.model_id
        JOIN events ev     ON ev.event_id = p.event_id
        WHERE e.is_final = 1
        GROUP BY m.name
        """),

    "q08_calibration_deciles": (
        "Are probabilities of the best model reliable? Deciles of predicted probability (CTE + NTILE)",
        """
        WITH best AS (
            SELECT experiment_id FROM experiments
            WHERE is_final = 1 ORDER BY roc_auc DESC LIMIT 1
        ),
        ranked AS (
            SELECT p.proba, ev.label,
                   NTILE(10) OVER (ORDER BY p.proba) AS decile
            FROM predictions p
            JOIN best b    ON b.experiment_id = p.experiment_id
            JOIN events ev ON ev.event_id = p.event_id
        )
        SELECT decile, COUNT(*) AS events,
               ROUND(AVG(proba), 4) AS mean_predicted,
               ROUND(AVG(label), 4) AS actual_signal_share
        FROM ranked
        GROUP BY decile
        ORDER BY decile
        """),

    "q09_hard_events": (
        "Error analysis: events that every trained model gets wrong vs the rest (CTE + HAVING)",
        """
        WITH per_event AS (
            SELECT p.event_id,
                   SUM(CASE WHEN (p.proba >= 0.5) <> ev.label THEN 1 ELSE 0 END) AS wrong_models,
                   COUNT(*) AS models
            FROM predictions p
            JOIN experiments e ON e.experiment_id = p.experiment_id
            JOIN models m      ON m.model_id = e.model_id
            JOIN events ev     ON ev.event_id = p.event_id
            WHERE e.is_final = 1 AND m.is_baseline = 0
            GROUP BY p.event_id
            HAVING COUNT(*) >= 2
        )
        SELECT CASE WHEN pe.wrong_models = pe.models THEN 'wrong in all models'
                    WHEN pe.wrong_models = 0 THEN 'correct in all models'
                    ELSE 'models disagree' END AS event_group,
               COUNT(*) AS events,
               ROUND(AVG(ev.label), 4) AS signal_share,
               ROUND(AVG(ev.m_bb), 4) AS mean_m_bb,
               ROUND(AVG(ev.m_wwbb), 4) AS mean_m_wwbb,
               ROUND(AVG(ev.m_wbb), 4) AS mean_m_wbb
        FROM per_event pe
        JOIN events ev ON ev.event_id = pe.event_id
        GROUP BY event_group
        ORDER BY events DESC
        """),

    "q10_importance_by_level": (
        "Total permutation importance of low-level vs high-level features (JOIN + GROUP BY)",
        """
        SELECT f.level, COUNT(*) AS features,
               ROUND(SUM(fi.importance_mean), 4) AS total_importance,
               ROUND(AVG(fi.importance_mean), 4) AS mean_importance
        FROM feature_importance fi
        JOIN features f ON f.feature_id = fi.feature_id
        GROUP BY f.level
        ORDER BY total_importance DESC
        """),

    "q11_top_features": (
        "Top-10 features of the best model with their rank inside their level (window RANK)",
        """
        SELECT f.name, f.level, f.object,
               ROUND(fi.importance_mean, 4) AS importance,
               RANK() OVER (PARTITION BY f.level ORDER BY fi.importance_mean DESC) AS rank_in_level
        FROM feature_importance fi
        JOIN features f ON f.feature_id = fi.feature_id
        ORDER BY fi.importance_mean DESC
        LIMIT 10
        """),

    "q12_feature_sets": (
        "AUC with low-level, high-level and all features vs the full set (subquery)",
        """
        SELECT e.feature_set, e.train_size, ROUND(e.roc_auc, 4) AS roc_auc,
               ROUND(e.roc_auc - (SELECT e2.roc_auc FROM experiments e2
                                  WHERE e2.kind = 'feature_set' AND e2.feature_set = 'all'), 4)
                   AS diff_vs_all_features,
               ROUND(e.train_time_s, 1) AS train_time_s
        FROM experiments e
        WHERE e.kind = 'feature_set'
        ORDER BY e.roc_auc DESC
        """),
}
