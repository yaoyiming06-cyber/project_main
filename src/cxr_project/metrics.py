"""Study-level ranking metrics used for model selection."""

import numpy as np
from sklearn.metrics import roc_auc_score

from .labels import LABEL_NAMES, NUM_LABELS


def compute_macro_auc(targets, probabilities):
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities)
    if targets.ndim != 2 or targets.shape[1] != NUM_LABELS:
        raise ValueError(f"targets must have shape [studies, {NUM_LABELS}]")
    if probabilities.shape != targets.shape:
        raise ValueError("probabilities must have the same shape as targets")
    aucs = []
    per_label = []
    for index, name in enumerate(LABEL_NAMES):
        valid = np.isin(targets[:, index], (0, 1))
        label_targets = targets[valid, index]
        if np.unique(label_targets).size < 2:
            auc = None
        else:
            auc = float(roc_auc_score(label_targets, probabilities[valid, index]))
            aucs.append(auc)
        per_label.append({
            "label_index": index,
            "label": name,
            "positive_studies": int((label_targets == 1).sum()),
            "negative_studies": int((label_targets == 0).sum()),
            "auroc": auc,
        })
    return {
        "macro_auc": float(np.mean(aucs)) if aucs else None,
        "auc_per_label": [item["auroc"] for item in per_label],
        "per_label": per_label,
        "missing_auc_labels": [item["label"] for item in per_label if item["auroc"] is None],
    }
