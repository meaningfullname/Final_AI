"""All figures of the project (matplotlib for static, Plotly for interactive)."""
import matplotlib
matplotlib.use("Agg")           # no display needed: figures are saved to files
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve

from . import config

CLASS_NAMES = {0: "background", 1: "signal"}
COLORS = {0: "#4C72B0", 1: "#DD8452"}


def _save(fig, name):
    path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path.relative_to(config.BASE_DIR)}")


# ---------------------------------------------------------------- EDA
def plot_class_balance(label_counts):
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = sorted(label_counts)
    values = [label_counts[k] for k in labels]
    bars = ax.bar([CLASS_NAMES[k] for k in labels], values, color=[COLORS[k] for k in labels])
    total = sum(values)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}\n({value / total:.1%})",
                ha="center", va="bottom")
    ax.set_ylabel("Number of events")
    ax.set_title("Class balance (whole dataset)")
    ax.set_ylim(0, max(values) * 1.2)
    _save(fig, "class_balance.png")


def plot_distributions(sample, features, name, title):
    """Histograms of the given features for signal and background."""
    cols = 4
    rows = int(np.ceil(len(features) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, feature in zip(axes, features):
        hi = sample[feature].quantile(0.99)
        lo = sample[feature].quantile(0.01)
        bins = np.linspace(lo, hi, 50)
        for label in (0, 1):
            ax.hist(sample.loc[sample[config.LABEL] == label, feature], bins=bins,
                    alpha=0.55, density=True, color=COLORS[label], label=CLASS_NAMES[label])
        ax.set_title(feature)
        ax.set_ylabel("density")
    for ax in axes[len(features):]:
        ax.axis("off")
    axes[0].legend()
    fig.suptitle(title)
    _save(fig, name)


def plot_correlation(sample):
    corr = sample[[config.LABEL] + config.FEATURES].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.columns, fontsize=7)
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    ax.set_title("Correlation matrix (label + 28 features)")
    _save(fig, "correlation_heatmap.png")
    return corr


def plot_boxplots(sample):
    features = [f for f in config.FEATURES if f in config.NON_NEGATIVE
                and f not in config.BTAG_FEATURES]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot([sample[f] for f in features], showfliers=True,
               flierprops={"markersize": 1, "alpha": 0.3})
    ax.set_xticks(range(1, len(features) + 1))
    ax.set_xticklabels(features, rotation=45, ha="right")
    ax.set_ylabel("value (dataset units)")
    ax.set_title("Outliers: momentum and mass features have long right tails")
    _save(fig, "boxplots_outliers.png")


def interactive_distributions(sample):
    """Plotly histogram with a dropdown to switch between high-level features."""
    sample = sample.sample(n=min(len(sample), 30_000), random_state=config.RANDOM_STATE)
    fig = go.Figure()
    for i, feature in enumerate(config.HIGH_LEVEL):
        hi = float(sample[feature].quantile(0.99))
        for label in (0, 1):
            values = sample.loc[sample[config.LABEL] == label, feature]
            fig.add_trace(go.Histogram(x=values[values <= hi], name=CLASS_NAMES[label],
                                       opacity=0.6, nbinsx=60, histnorm="probability density",
                                       marker_color=COLORS[label], visible=(i == 0)))
    buttons = []
    for i, feature in enumerate(config.HIGH_LEVEL):
        visible = [j // 2 == i for j in range(2 * len(config.HIGH_LEVEL))]
        buttons.append({"label": feature, "method": "update",
                        "args": [{"visible": visible}, {"xaxis.title.text": feature}]})
    fig.update_layout(barmode="overlay", title="High-level features: signal vs background",
                      xaxis_title=config.HIGH_LEVEL[0], yaxis_title="density",
                      updatemenus=[{"buttons": buttons, "x": 1.0, "xanchor": "right", "y": 1.15}])
    path = config.FIGURES_DIR / "interactive_distributions.html"
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"  saved {path.relative_to(config.BASE_DIR)}")


# ---------------------------------------------------------------- models
def plot_learning_curves(experiments):
    """AUC on the fixed test set vs number of training rows (static + interactive)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    interactive = go.Figure()
    for model, group in experiments.groupby("model"):
        group = group.sort_values("train_size")
        if len(group) < 2:
            ax.axhline(group["roc_auc"].iloc[0], linestyle="--", color="gray", label=model)
            continue
        ax.plot(group["train_size"], group["roc_auc"], marker="o", label=model)
        interactive.add_trace(go.Scatter(
            x=group["train_size"], y=group["roc_auc"], mode="lines+markers", name=model,
            customdata=group[["train_time_s", "model_size_mb"]],
            hovertemplate="rows=%{x:,}<br>AUC=%{y:.4f}<br>train time=%{customdata[0]:.1f}s"
                          "<br>size=%{customdata[1]:.1f} MB"))
    ax.set_xscale("log")
    ax.set_xlabel("Training rows (log scale)")
    ax.set_ylabel("ROC-AUC on the test set")
    ax.set_title("Learning curves")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, "learning_curves.png")
    interactive.update_layout(title="Learning curves (hover for time and model size)",
                              xaxis_type="log", xaxis_title="Training rows",
                              yaxis_title="ROC-AUC on the test set")
    path = config.FIGURES_DIR / "interactive_learning_curves.html"
    interactive.write_html(path, include_plotlyjs="cdn")
    print(f"  saved {path.relative_to(config.BASE_DIR)}")


def plot_roc_curves(y_true, probabilities):
    """`probabilities` is a dict: model name -> predicted probability of signal."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for model, proba in probabilities.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        ax.plot(fpr, tpr, label=model)
    ax.plot([0, 1], [0, 1], linestyle=":", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves on the test set")
    ax.legend()
    _save(fig, "roc_curves.png")


def plot_confusion(y_true, y_pred, model_name):
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{value:,}\n{value / matrix.sum():.1%}", ha="center", va="center",
                color="white" if value > matrix.max() / 2 else "black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["background", "signal"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["background", "signal"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix: {model_name} (threshold 0.5)")
    _save(fig, "confusion_matrix.png")
    return matrix


def plot_importance(importance, model_name):
    """`importance` is a DataFrame with columns feature, importance_mean, importance_std."""
    data = importance.sort_values("importance_mean")
    colors = ["#DD8452" if f in config.HIGH_LEVEL else "#4C72B0" for f in data["feature"]]
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(data["feature"], data["importance_mean"], xerr=data["importance_std"], color=colors)
    ax.set_xlabel("Drop of ROC-AUC when the feature is shuffled")
    ax.set_title(f"Permutation importance: {model_name}\n(orange = high-level, blue = low-level)")
    _save(fig, "permutation_importance.png")


def plot_feature_sets(results):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(results["feature_set"], results["roc_auc"], color="#4C72B0")
    for bar, value in zip(bars, results["roc_auc"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom")
    ax.set_ylim(0.5, min(1.0, results["roc_auc"].max() + 0.05))
    ax.set_ylabel("ROC-AUC on the test set")
    ax.set_title("Which features carry the information?")
    _save(fig, "feature_sets.png")
