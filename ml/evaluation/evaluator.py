"""
DeepForensics Evaluation Metric Calculator

Calculates research-grade forensic classification metrics:
- Accuracy, Precision, Recall, F1
- ROC-AUC & PR-AUC
- Equal Error Rate (EER)
- Confusion Matrix (TP, FP, TN, FN)
- False Positive Rate (FPR) & False Negative Rate (FNR)
"""
from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix
)


def compute_equal_error_rate(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """
    Computes Equal Error Rate (EER) where FPR == FNR.
    Standard metric for biometric & deepfake verification benchmarks.
    """
    if len(np.unique(y_true)) < 2:
        return 0.0

    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    from sklearn.metrics import roc_curve

    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_probs, pos_label=1)
        fnr = 1 - tpr
        
        # EER is point where fpr == fnr
        try:
            eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        except Exception:
            diff = np.absolute(fnr - fpr)
            if len(diff) > 0 and not np.all(np.isnan(diff)):
                idx = np.nanargmin(diff)
                eer = float(fpr[idx])
            else:
                eer = 0.0
        return float(eer)
    except Exception:
        return 0.0


def compute_brier_score(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """Computes Brier score (mean squared probability error). Lower is better."""
    return float(np.mean((y_probs - y_true) ** 2)) if len(y_true) > 0 else 0.0


def compute_expected_calibration_error(y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) across n probability bins."""
    if len(y_true) == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]
        mask = (y_probs >= bin_lower) & (y_probs < bin_upper) if i < n_bins - 1 else (y_probs >= bin_lower) & (y_probs <= bin_upper)
        bin_count = np.sum(mask)

        if bin_count > 0:
            bin_acc = np.mean(y_true[mask] == (y_probs[mask] >= 0.5))
            bin_conf = np.mean(y_probs[mask])
            ece += (bin_count / total_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def evaluate_predictions(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Evaluate binary forensic predictions against ground truth labels.
    
    y_true: 1D array of ground truth integers (0=REAL, 1=FAKE)
    y_probs: 1D array of predicted probabilities P(FAKE) [0.0, 1.0]
    """
    y_pred = (y_probs >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    if len(np.unique(y_true)) >= 2:
        try:
            roc_auc = float(roc_auc_score(y_true, y_probs))
        except ValueError:
            roc_auc = 0.5

        try:
            prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_probs)
            pr_auc = float(auc(rec_curve, prec_curve))
        except Exception:
            pr_auc = 0.5
    else:
        roc_auc = 1.0 if acc == 1.0 else 0.5
        pr_auc = 1.0 if acc == 1.0 else 0.5

    eer = compute_equal_error_rate(y_true, y_probs)
    brier = compute_brier_score(y_true, y_probs)
    ece = compute_expected_calibration_error(y_true, y_probs)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    tpr = 1.0 - fnr
    tnr = 1.0 - fpr
    balanced_acc = float((tpr + tnr) / 2.0)

    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "eer": round(eer, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        }
    }
