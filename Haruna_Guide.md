# Haruna — Vision Transformer (ViT) & Attention Guide

> **Purpose:** Read this before the group meeting so you understand the architecture you're implementing and can explain your approach to the team.

---

## 1. What Is a Vision Transformer (ViT)?

ViT (Dosovitskiy et al., 2021) applies the **Transformer** architecture — originally built for language — directly to images. Instead of words, it processes **image patches**.

### How ViT Processes an Image

```
Input Image (224×224×3)
    ↓
Split into patches (e.g. 16×16 → 196 patches)
    ↓
Flatten each patch into a vector (16×16×3 = 768 dims)
    ↓
Add positional embeddings (so the model knows patch order)
    ↓
Prepend a [CLS] token (used for classification)
    ↓
Pass through Transformer Encoder (12 layers of self-attention)
    ↓
Take the [CLS] token output → Linear head → Prediction
```

### Key Concepts

| Concept | What It Means |
|---|---|
| **Patch Embedding** | The image is chopped into a grid of patches, each treated like a "word" |
| **Self-Attention** | Every patch attends to every other patch — captures global relationships |
| **[CLS] Token** | A special learnable token whose final representation is used for classification |
| **Positional Embedding** | Added to patches so the model knows spatial layout (transformers have no built-in position sense) |
| **Multi-Head Attention** | Multiple attention patterns run in parallel — different heads learn different relationships |

### Why ViT for Our Project

- ViT captures **global context** — it can relate distant parts of the image (e.g., lesion border irregularity)
- CNNs like MobileNetV2 are inherently local (small receptive fields)
- ViT serves as the **high-capacity baseline** to compare MobileNetV2 against

---

## 2. ViT vs MobileNetV2 — Key Differences

| Property | MobileNetV2 | ViT-Base |
|---|---|---|
| Parameters | ~3.4M | ~86M |
| FLOPs | ~300M | ~17B |
| Approach | Local convolutions | Global self-attention |
| Data Hungry? | No — works with small datasets | Yes — originally needs millions of images |
| Mobile Ready? | ✅ Designed for it | ❌ Too large without compression |
| Pretrained On | ImageNet | ImageNet-21K (often) |

**Key implication:** ViT needs more aggressive regularisation and augmentation on HAM10000 (only 10K images). This is expected — document it in your training notes.

---

## 3. Transfer Learning Strategy for ViT

### Recommended Approach

```
1. Load pretrained ViT (vit_b_16 from torchvision OR google/vit-base-patch16-224 from HuggingFace)
2. Freeze patch embedding + positional embedding
3. Freeze the first 8-10 of 12 transformer blocks
4. Replace classification head:
   - Stage 1: Linear(768, 2)   → benign vs malignant
   - Stage 2: Linear(768, 3)   → mel vs bcc vs akiec
5. Train the head + unfrozen layers for 10-15 epochs
6. Optionally unfreeze more layers and fine-tune with smaller LR
```

### Important ViT-Specific Training Tips

- **Use AdamW** (not plain Adam) — ViT was designed with weight decay
- **Cosine LR scheduler** — gradually reduce learning rate to avoid catastrophic forgetting
- **Learning rate:** 3e-5 to 5e-5 (much smaller than typical CNN LRs)
- **Mixed precision** (`torch.cuda.amp`) — ViT is memory-heavy, this halves GPU usage
- **Gradient accumulation** — if batch size > GPU memory, accumulate over 2-4 steps

---

## 4. Attention Visualisation (Your Version of Grad-CAM)

Standard Grad-CAM targets convolutional feature maps, but ViT doesn't have traditional convolutions. You have two options:

### Option A: Attention Rollout (Recommended)

1. Extract the attention weights from every transformer layer
2. Multiply attention matrices layer by layer (with residual connections)
3. Take the attention from the [CLS] token to all patches
4. Reshape back to a 14×14 grid → upsample to 224×224
5. Overlay on the original image

### Option B: Grad-CAM on the Last Projection

- Apply Grad-CAM to the final linear projection before the classification head
- Less standard but simpler to implement

### Libraries

- `pytorch-grad-cam` has ViT support — use `GradCAM` with `reshape_transform` for ViT
- Or implement Attention Rollout manually (< 30 lines of code)

---

## 5. Related Work — What to Research

Your report section is **Related Work (10%)**. You need to survey:

| Topic | What to Cover |
|---|---|
| **Dermoscopy + Deep Learning** | Key papers using CNNs for skin lesion classification (ISIC challenge papers) |
| **ViT in Medical Imaging** | Papers applying ViT to medical/pathology images — what worked, what didn't |
| **Grad-CAM / Explainability** | How visual explanations build trust in clinical AI |
| **HAM10000 Benchmarks** | What accuracy have others achieved on this dataset? What models did they use? |

Aim for **8–12 references** in this section. Your proposal already cites 5 — expand from there.

---

## 6. Key Points to Raise at the Meeting

1. **ViT will train slower** — expect 2–3× training time vs Isam's MobileNetV2. Plan accordingly.
2. **GPU memory** — if using Colab Free, you may need to reduce batch size to 8–16 for ViT
3. **Same data splits** — get the split indices from Isam, do NOT create your own
4. **Save checkpoints** — Caden needs your trained models to run evaluation
5. **Attention maps vs Grad-CAM** — discuss with the team which visualisation method to use for ViT in the report, for fair comparison with Isam's Grad-CAM
