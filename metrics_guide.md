# Evaluation Metrics Guide — Skin Cancer Detection Project

## Stage 1 (Benign vs Malignant)

| Metric | What you want | Why |
|---|---|---|
| **Recall (Malignant)** | >= 0.95 | Most critical — a missed cancer never reaches Stage 2. Even 0.90 is risky. |
| **Recall (Benign)** | >= 0.70 | Lower priority. False alarms (benign flagged as malignant) just send healthy people to Stage 2 unnecessarily. |
| **Precision (Malignant)** | >= 0.60 | How many flagged cases are actually malignant. Low precision = lots of false alarms but that's acceptable. |
| **Macro F1** | >= 0.75 | Overall balance across both classes. |
| **ROC-AUC** | >= 0.85 | Above 0.9 is excellent. Below 0.7 means the model is barely discriminating. |

---

## Stage 2 (mel / bcc / akiec)

| Metric | What you want | Why |
|---|---|---|
| **Recall (mel)** | >= 0.90 | Melanoma is the most dangerous — missing it is the worst Stage 2 outcome. |
| **Recall (bcc / akiec)** | >= 0.75 | Still important but slightly less aggressive than melanoma. |
| **Macro F1** | >= 0.70 | Gives equal weight to all 3 classes including rare ones. |
| **Weighted F1** | >= 0.75 | Accounts for class imbalance — useful alongside macro. |
| **ROC-AUC** | >= 0.80 | Multi-class OvR — above 0.85 is strong for a 3-class medical problem. |

---

## Pipeline

| Metric | What you want | Why |
|---|---|---|
| **Pipeline Recall** | >= 0.85 | Of ALL malignant cases in the test set, what % got the correct final diagnosis. This is the single most important number in the whole report. |
| **Stage 1 Leakage Rate** | <= 0.05 | % of malignant cases missed entirely by Stage 1. Even 5% means cancer going undetected. |
| **Stage 2 Conditional Accuracy** | >= 0.75 | Among cases that made it through Stage 1, how well does Stage 2 subtype them. |

---

## Efficiency

| Metric | MobileNet target | ViT target | Why it matters |
|---|---|---|---|
| **Parameters** | ~3-4M | ~85M | Fewer = lighter for mobile deployment |
| **Model Size** | ~14 MB | ~300+ MB | Matters for embedding in a phone app |
| **FLOPs** | ~300M | ~17B | Compute cost per image |
| **Inference time** | < 20ms | < 100ms | Real-time clinical use needs to be fast |

---

## The Key Trade-off (for the report)

If MobileNet gets close to ViT's recall (e.g., 0.92 vs 0.94) but runs 5x faster
with 25x fewer parameters, MobileNet is the better choice for real-world deployment.
That comparison is the story you're trying to tell.

---

## What You Need From Teammates

Tell Isam and Haruna to give you 4 arrays per model:

| Array | Shape | Description |
|---|---|---|
| y_true_binary | (n_test,) | Ground truth: 0 = benign, 1 = malignant (ALL test images) |
| y_pred_binary | (n_test,) | Stage 1 predictions: 0 or 1 |
| y_true_stage2 | (n_flagged,) | Ground truth for images Stage 1 predicted malignant (0=akiec, 1=bcc, 2=mel) |
| y_pred_stage2 | (n_flagged,) | Stage 2 predictions for those same images |

Plus optionally (for ROC-AUC curves):
- y_probs_binary  — shape (n_test, 2)     — Stage 1 probability outputs
- y_probs_stage2  — shape (n_flagged, 3)  — Stage 2 probability outputs

For efficiency comparison they also need to:
- Save trained models as .pt files
- Give you the model objects or saved paths
- Have thop installed: pip install thop

Also agree on:
1. Same test set — both models evaluated on identical images
2. Same preprocessing — images processed identically for fair comparison
3. Label mapping — confirm 0=akiec, 1=bcc, 2=mel for Stage 2
