"""
Evaluation Module — Skin Cancer Detection Pipeline
===================================================
Group Project: HAM10000 Two-Stage Hierarchical Classification

Stage 1: Binary (Benign vs Malignant)
Stage 2: Multi-class (mel vs bcc vs akiec — malignant subtypes)

Usage:
    from evaluation import *

    # After your teammates provide predictions:
    evaluate_stage1(y_true_binary, y_pred_binary, y_probs_binary)
    evaluate_stage2(y_true_multi, y_pred_multi, y_probs_multi)
    evaluate_pipeline(y_true_binary, y_pred_binary, y_true_multi, y_pred_multi)
    compare_models(mobilenet_results, vit_results)
    compare_efficiency(mobilenet_model, vit_model, input_size=(3, 224, 224))
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    recall_score, precision_score, f1_score, accuracy_score,
    roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# ---------------------------------------------------------------------------
# Class definitions
# ---------------------------------------------------------------------------

BINARY_CLASSES = ['Benign', 'Malignant']

# Stage 2 only classifies the malignant subtypes
MALIGNANT_CLASSES = ['akiec', 'bcc', 'mel']

# All 7 HAM10000 classes (for reference / full-dataset evaluation)
ALL_CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# Which of the 7 classes are malignant (used in pipeline evaluation)
MALIGNANT_SET = {'akiec', 'bcc', 'mel'}


# ============================= CORE METRICS ================================

def print_classification_report(y_true, y_pred, class_names, title="Classification Report"):
    """Print sklearn classification report with a clear header."""
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))


def print_per_class_recall(y_true, y_pred, class_names):
    """
    Print recall for every class on its own line.
    Recall is the PRIMARY metric — missing a cancer case is the worst outcome.
    """
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    print("-" * 40)
    print("  *** PER-CLASS RECALL (PRIMARY METRIC) ***")
    print("-" * 40)
    for name, r in zip(class_names, recalls):
        bar = "█" * int(r * 30)
        print(f"  {name:<12s}  {r:.4f}  {bar}")
    print(f"\n  Macro Recall:    {recalls.mean():.4f}")
    print(f"  Weighted Recall: {recall_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print("-" * 40)


def compute_f1_scores(y_true, y_pred):
    """Return macro and weighted F1 scores as a dict."""
    return {
        'Macro F1':    f1_score(y_true, y_pred, average='macro', zero_division=0),
        'Weighted F1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }


# =========================== CONFUSION MATRIX ==============================

def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix",
                          cmap='Blues', figsize=None):
    """Plot a confusion matrix heatmap."""
    n = len(class_names)
    if figsize is None:
        figsize = (5, 4) if n <= 3 else (8, 7)

    fig, ax = plt.subplots(figsize=figsize)
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=class_names,
        cmap=cmap, ax=ax,
        colorbar=True
    )
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


# =============================== ROC-AUC ===================================

def compute_roc_auc(y_true, y_probs, class_names):
    """
    Compute multi-class ROC-AUC (One-vs-Rest, macro averaged).

    Parameters
    ----------
    y_true   : array of integer labels (0, 1, ... n_classes-1)
    y_probs  : array of shape (n_samples, n_classes) — probability per class
    class_names : list of class name strings

    Returns
    -------
    macro_auc : float
    """
    n_classes = len(class_names)

    if n_classes == 2:
        # Binary case — just use the positive-class probabilities
        auc_score = roc_auc_score(y_true, y_probs[:, 1])
        print(f"ROC-AUC (binary): {auc_score:.4f}")
        return auc_score

    # Multi-class: binarize labels and compute OvR
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    macro_auc = roc_auc_score(y_bin, y_probs, average='macro', multi_class='ovr')
    print(f"ROC-AUC (macro, OvR): {macro_auc:.4f}")
    return macro_auc


def plot_roc_curves(y_true, y_probs, class_names, title="ROC Curves"):
    """Plot per-class ROC curves with macro-average."""
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(7, 6))

    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{name} (AUC = {auc_val:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


# ======================== STAGE EVALUATIONS ================================

def evaluate_stage1(y_true, y_pred, y_probs=None):
    """
    Full evaluation for Stage 1 (Binary: Benign vs Malignant).

    Parameters
    ----------
    y_true  : array of 0 (benign) or 1 (malignant)
    y_pred  : array of 0 or 1
    y_probs : array of shape (n_samples, 2) — optional, for ROC-AUC

    Returns
    -------
    results : dict of all computed metrics
    """
    print("\n" + "=" * 60)
    print("  STAGE 1 EVALUATION — Benign vs Malignant")
    print("=" * 60 + "\n")

    print_classification_report(y_true, y_pred, BINARY_CLASSES,
                                title="Stage 1 Classification Report")
    print_per_class_recall(y_true, y_pred, BINARY_CLASSES)

    f1s = compute_f1_scores(y_true, y_pred)
    print(f"\nMacro F1:    {f1s['Macro F1']:.4f}")
    print(f"Weighted F1: {f1s['Weighted F1']:.4f}")
    print(f"Accuracy:    {accuracy_score(y_true, y_pred):.4f}")

    results = {
        'Accuracy':       accuracy_score(y_true, y_pred),
        'Macro F1':       f1s['Macro F1'],
        'Weighted F1':    f1s['Weighted F1'],
        'Recall (Mal)':   recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        'Precision (Mal)':precision_score(y_true, y_pred, pos_label=1, zero_division=0),
    }

    if y_probs is not None:
        results['ROC-AUC'] = compute_roc_auc(y_true, y_probs, BINARY_CLASSES)

    plot_confusion_matrix(y_true, y_pred, BINARY_CLASSES,
                          title="Stage 1 — Confusion Matrix")
    return results


def evaluate_stage2(y_true, y_pred, y_probs=None):
    """
    Full evaluation for Stage 2 (Multi-class: malignant subtypes).

    Parameters
    ----------
    y_true  : array of integer labels (0=akiec, 1=bcc, 2=mel)
    y_pred  : array of integer labels
    y_probs : array of shape (n_samples, 3) — optional, for ROC-AUC
    """
    print("\n" + "=" * 60)
    print("  STAGE 2 EVALUATION — Malignant Subtypes")
    print("=" * 60 + "\n")

    print_classification_report(y_true, y_pred, MALIGNANT_CLASSES,
                                title="Stage 2 Classification Report")
    print_per_class_recall(y_true, y_pred, MALIGNANT_CLASSES)

    f1s = compute_f1_scores(y_true, y_pred)
    print(f"\nMacro F1:    {f1s['Macro F1']:.4f}")
    print(f"Weighted F1: {f1s['Weighted F1']:.4f}")
    print(f"Accuracy:    {accuracy_score(y_true, y_pred):.4f}")

    results = {
        'Accuracy':    accuracy_score(y_true, y_pred),
        'Macro F1':    f1s['Macro F1'],
        'Weighted F1': f1s['Weighted F1'],
    }

    # Per-class recall into results
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    for name, r in zip(MALIGNANT_CLASSES, recalls):
        results[f'Recall ({name})'] = r

    if y_probs is not None:
        results['ROC-AUC'] = compute_roc_auc(y_true, y_probs, MALIGNANT_CLASSES)
        plot_roc_curves(y_true, y_probs, MALIGNANT_CLASSES,
                        title="Stage 2 — ROC Curves (OvR)")

    plot_confusion_matrix(y_true, y_pred, MALIGNANT_CLASSES,
                          title="Stage 2 — Confusion Matrix", cmap='Oranges')
    return results


# ====================== PIPELINE EVALUATION ================================

def evaluate_pipeline(y_true_binary, y_pred_binary,
                      y_true_stage2, y_pred_stage2):
    """
    Evaluate the full two-stage pipeline.

    Measures how Stage 1 errors propagate into Stage 2.

    Parameters
    ----------
    y_true_binary : array of 0/1 — ground truth for ALL test images
    y_pred_binary : array of 0/1 — Stage 1 predictions for ALL test images
    y_true_stage2 : array — ground truth labels for images that Stage 1
                    predicted as malignant (i.e. y_pred_binary == 1)
    y_pred_stage2 : array — Stage 2 predictions for those same images

    Returns
    -------
    pipeline_results : dict
    """
    print("\n" + "=" * 60)
    print("  PIPELINE EVALUATION — Stage 1 → Stage 2")
    print("=" * 60 + "\n")

    total_malignant = (y_true_binary == 1).sum()
    total_benign    = (y_true_binary == 0).sum()

    # Stage 1 performance on malignant cases
    stage1_tp = ((y_true_binary == 1) & (y_pred_binary == 1)).sum()
    stage1_fn = ((y_true_binary == 1) & (y_pred_binary == 0)).sum()
    stage1_recall = stage1_tp / total_malignant if total_malignant > 0 else 0

    # Stage 1 leakage: malignant cases missed entirely
    leakage_rate = stage1_fn / total_malignant if total_malignant > 0 else 0

    print(f"Total malignant cases in test set:     {total_malignant}")
    print(f"Correctly flagged by Stage 1 (TP):     {stage1_tp}")
    print(f"Missed by Stage 1 (FN — LEAKAGE):      {stage1_fn}")
    print(f"Stage 1 Recall (malignant):            {stage1_recall:.4f}")
    print(f"Stage 1 Leakage Rate:                  {leakage_rate:.4f}")

    # Stage 2 conditional accuracy: among cases that reached Stage 2,
    # how many got the correct final malignant subtype?
    if len(y_true_stage2) > 0:
        stage2_correct = (y_true_stage2 == y_pred_stage2).sum()
        stage2_cond_acc = stage2_correct / len(y_true_stage2)
    else:
        stage2_correct = 0
        stage2_cond_acc = 0

    print(f"\nCases reaching Stage 2:                {len(y_true_stage2)}")
    print(f"Correctly classified by Stage 2:       {stage2_correct}")
    print(f"Stage 2 Conditional Accuracy:          {stage2_cond_acc:.4f}")

    # Overall pipeline recall: of ALL truly malignant cases, what % get
    # the correct final diagnosis (correct at BOTH stages)?
    # A case must be flagged malignant by Stage 1 AND get the right subtype
    # from Stage 2.
    #
    # y_true_stage2 contains ground truth for cases where y_pred_binary == 1.
    # Among those, only the ones that were truly malignant (y_true_binary == 1)
    # AND correctly subtyped count as pipeline successes.
    pipeline_correct = stage2_correct  # among TP cases from stage 1
    pipeline_recall = pipeline_correct / total_malignant if total_malignant > 0 else 0

    print(f"\n*** OVERALL PIPELINE RECALL:            {pipeline_recall:.4f} ***")
    print(f"    (of {total_malignant} malignant cases, {pipeline_correct} got the correct final diagnosis)")

    # Pipeline accuracy for ALL test images
    # Benign cases: correct if Stage 1 predicts benign
    benign_correct = ((y_true_binary == 0) & (y_pred_binary == 0)).sum()
    total_correct = benign_correct + pipeline_correct
    total_samples = len(y_true_binary)
    pipeline_accuracy = total_correct / total_samples if total_samples > 0 else 0

    print(f"\nPipeline Accuracy (all test images):   {pipeline_accuracy:.4f}")
    print(f"    ({benign_correct} benign correct + {pipeline_correct} malignant fully correct "
          f"out of {total_samples} total)")

    results = {
        'Stage 1 Recall (Mal)':    stage1_recall,
        'Stage 1 Leakage Rate':    leakage_rate,
        'Stage 2 Cond. Accuracy':  stage2_cond_acc,
        'Pipeline Recall':         pipeline_recall,
        'Pipeline Accuracy':       pipeline_accuracy,
    }

    print("\n" + "-" * 40)
    return results


# ========================= MODEL COMPARISON ================================

def compare_models(mobilenet_results, vit_results):
    """
    Side-by-side comparison of MobileNetV2 vs ViT.

    Parameters
    ----------
    mobilenet_results : dict — output from evaluate_stage1/stage2/pipeline
    vit_results       : dict — same structure

    Returns
    -------
    comparison_df : pandas DataFrame
    """
    df = pd.DataFrame({
        'MobileNetV2': mobilenet_results,
        'ViT':         vit_results,
    })
    df.index.name = 'Metric'

    print("\n" + "=" * 60)
    print("  MODEL COMPARISON — MobileNetV2 vs ViT")
    print("=" * 60)
    print(df.round(4).to_string())
    print()

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    width = 0.35
    bars1 = ax.bar(x - width/2, df['MobileNetV2'], width, label='MobileNetV2', color='steelblue')
    bars2 = ax.bar(x + width/2, df['ViT'],         width, label='ViT',         color='tomato')
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=30, ha='right')
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score')
    ax.set_title('MobileNetV2 vs ViT — Performance Comparison', fontsize=13)
    ax.legend()
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.show()

    return df


# ====================== EFFICIENCY COMPARISON ==============================

def compare_efficiency(model1, model2, model1_path=None, model2_path=None,
                       input_size=(3, 224, 224), device='cpu', n_runs=100):
    """
    Compare computational efficiency of two PyTorch models.

    Parameters
    ----------
    model1, model2       : PyTorch nn.Module (MobileNetV2, ViT)
    model1_path, model2_path : str — paths to saved .pt files (for file size)
    input_size           : tuple (C, H, W)
    device               : 'cpu' or 'cuda'
    n_runs               : number of forward passes for timing

    Returns
    -------
    efficiency_df : pandas DataFrame
    """
    import torch

    results = {}

    for name, model, path in [('MobileNetV2', model1, model1_path),
                               ('ViT',         model2, model2_path)]:
        model = model.to(device).eval()
        dummy = torch.randn(1, *input_size).to(device)

        # Parameter count
        params = sum(p.numel() for p in model.parameters())

        # Model file size
        if path and os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
        else:
            # Estimate from parameters (4 bytes per float32 param)
            size_mb = params * 4 / (1024 * 1024)

        # FLOPs (requires thop)
        flops = None
        try:
            from thop import profile
            flops, _ = profile(model, inputs=(dummy,), verbose=False)
        except ImportError:
            print(f"  [!] thop not installed — skipping FLOPs for {name}")
        except Exception as e:
            print(f"  [!] FLOPs failed for {name}: {e}")

        # Inference time
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = model(dummy)

            start = time.time()
            for _ in range(n_runs):
                _ = model(dummy)
            elapsed = (time.time() - start) / n_runs * 1000  # ms per image

        results[name] = {
            'Parameters':         f'{params:,}',
            'Model Size (MB)':    f'{size_mb:.1f}',
            'FLOPs':              f'{flops:,.0f}' if flops else 'N/A',
            'Inference (ms/img)': f'{elapsed:.1f}',
        }

    df = pd.DataFrame(results)
    df.index.name = 'Metric'

    print("\n" + "=" * 60)
    print("  EFFICIENCY COMPARISON")
    print("=" * 60)
    print(df.to_string())
    print()

    return df


# ============================ DEMO / TESTING ===============================

def run_demo():
    """
    Run a quick demo with dummy data to verify everything works.
    Call this to test the module before the real models are ready.
    """
    np.random.seed(42)
    n = 200

    # --- Stage 1: Binary ---
    y_true_bin = np.random.choice([0, 1], size=n, p=[0.7, 0.3])
    # Simulate a decent model (80% accuracy)
    y_pred_bin = y_true_bin.copy()
    flip_idx = np.random.choice(n, size=int(n * 0.2), replace=False)
    y_pred_bin[flip_idx] = 1 - y_pred_bin[flip_idx]
    y_probs_bin = np.random.dirichlet([2, 2], size=n)

    s1_results = evaluate_stage1(y_true_bin, y_pred_bin, y_probs_bin)

    # --- Stage 2: Multi-class (only for cases Stage 1 predicted malignant) ---
    flagged_mask = (y_pred_bin == 1)
    n_flagged = flagged_mask.sum()

    # Ground truth for flagged cases (0=akiec, 1=bcc, 2=mel)
    y_true_s2 = np.random.choice([0, 1, 2], size=n_flagged, p=[0.2, 0.3, 0.5])
    y_pred_s2 = y_true_s2.copy()
    flip_s2 = np.random.choice(n_flagged, size=int(n_flagged * 0.25), replace=False)
    y_pred_s2[flip_s2] = np.random.choice([0, 1, 2], size=len(flip_s2))
    y_probs_s2 = np.random.dirichlet([2, 2, 2], size=n_flagged)

    s2_results = evaluate_stage2(y_true_s2, y_pred_s2, y_probs_s2)

    # --- Pipeline ---
    pipeline_results = evaluate_pipeline(y_true_bin, y_pred_bin,
                                         y_true_s2, y_pred_s2)

    # --- Model comparison (fake second model) ---
    mobilenet_all = {**s1_results, **s2_results, **pipeline_results}
    vit_fake = {k: v * np.random.uniform(0.9, 1.1) for k, v in mobilenet_all.items()
                if isinstance(v, (int, float))}
    compare_models(mobilenet_all, vit_fake)

    print("\n✓ Demo complete — all functions working.\n")


if __name__ == '__main__':
    run_demo()
