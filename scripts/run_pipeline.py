"""End-to-end pipeline runner: load → train → evaluate → save plots.

Used both for local development and to refresh the docs/ screenshots.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from m3_loan_default import data, explain, model

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)


def save_confusion_matrix(cm: list[list[int]], out_path: Path) -> Path:
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_arr, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Default", "Default"])
    ax.set_yticklabels(["No Default", "Default"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — UCI Credit Default")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center",
                    color="white" if cm_arr[i, j] > cm_arr.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> dict:
    X, y = data.load()
    print(f"Loaded: X={X.shape}, default_rate={y.mean():.4f}")

    result = model.train(X, y, run_cv=True)
    print("Metrics:", json.dumps(result.metrics, indent=2))

    save_confusion_matrix(result.metrics["confusion_matrix"], DOCS / "confusion_matrix.png")

    shap_values = explain.compute_shap_values(result.model, result.X_test.head(1000))
    explain.summary_plot(shap_values, result.X_test.head(1000),
                         out_path=DOCS / "shap_summary.png")
    explain.waterfall_plot(shap_values, row_idx=0,
                           out_path=DOCS / "shap_waterfall.png")

    top = explain.top_features(shap_values, result.X_test.head(1000), k=10)
    print("Top features:\n", top.to_string(index=False))

    metrics_path = DOCS / "metrics.json"
    metrics_path.write_text(json.dumps(result.metrics, indent=2))
    return result.metrics


if __name__ == "__main__":
    main()
