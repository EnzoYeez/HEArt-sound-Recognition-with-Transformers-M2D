# Heart Murmur Classification (M2D + Transformers)
> **Ternary classification on the CirCor DigiScope dataset with pretrained audio representations and a lightweight Transformer head.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](#)
[![License](https://img.shields.io/badge/License-MIT-green)](#)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](#)
[![Issues](https://img.shields.io/badge/Issues-please%20report-orange)](#)

---

## TL;DR
This repo implements a **3-class heart murmur classifier** — *Absent / Present / Unknown* — by combining **pretrained Masked Modeling Duo (M2D)** audio embeddings with a **multi-head Transformer classifier**. It follows the course/project brief and reports results against **state-of-the-art (SOTA)** baselines on **CirCor DigiScope** with the recommended metrics **Weighted Accuracy (W.acc)** and **Unweighted Average Recall (UAR)**.

---

## 1) Project Overview
- **Goal.** Build a deep-learning system for **heart murmur classification** using auscultation audio; compare with SOTA and write a report.  
- **Our approach.**  
  - Use **pretrained M2D** as a fixed or fine-tuned backbone to extract robust audio embeddings.  
  - Encode **multi-site** recordings (AV/MV/PV/TV) into **site tokens**, apply a **4-layer Transformer encoder** with **attention pooling**, and **multi-head (K=3) classification** with **KL distillation** across heads.  
  - Train with **AdamW + cosine decay**, **SpecAugment**, and **Focal Loss + label smoothing**; evaluate with **W.acc** and **UAR**.  
- **Why M2D?** Recent results on CirCor show **M2D outperforms prior methods** and benefits from **ensembling** with other general-purpose audio models.

---

## 2) Dataset: CirCor DigiScope
We use the **CirCor DigiScope** dataset — a large pediatric heart-sound resource with **5,282 recordings from 1,568 patients**, annotated murmurs, and **multi-site auscultation** (AV/MV/PV/TV). Signals are recorded near **4 kHz**, with rich clinical metadata and expert-reviewed segmentations.

> **Task framing.** Public splits for murmur **3-class** classification (*Present / Absent / Unknown*) follow PhysioNet/Challenge conventions adopted in recent work.

---

## 3) Method
### 3.1 Preprocessing & Feature Extraction
- **Resample** audio to 16 kHz; trim/pad clips (e.g., 10 s) per recording.  
- Feed waveforms to **pretrained M2D** to obtain **frame-level embeddings**; average-pool to a **site-level token** per auscultation point; **mask missing sites**.

### 3.2 Classifier (M2DTransformer)
- Concatenate up to **4 site tokens** → linear projection (e.g., 3840→512) → **4-layer Transformer** (n=4 heads, FFN=1024).  
- **Attention pooling** produces a global vector.  
- **K=3 classification heads** + **inter-head KL** for **distilled ensembling**.

### 3.3 Losses & Training
- **Focal Loss** (+ label smoothing) to address class imbalance.  
- **Distillation loss** (KL) between each head and the averaged logits.  
- **AdamW**, **batch size 32**, cosine annealing, early stopping on **W.acc**.

---

## 4) Metrics
We report:
- **Weighted Accuracy (W.acc)** — places higher weight on clinically important classes (*Present*, *Unknown*).  
- **Unweighted Average Recall (UAR)** — macro-averaged recall over the three classes.  
Formal definitions and evaluation protocol follow recent literature.

---

## 5) Baselines & SOTA
**Pretrained general-purpose audio representations on CirCor (test):**  
- **M2D**: **W.acc ~0.83**, **UAR ~0.71** (Transformer, SSL)  
- **AST**: W.acc ~0.65, UAR ~0.67 (Transformer, SL)  
- **CNN14 / BYOL-A** show different class-recall profiles; **ensembling with M2D** improves balance (e.g., **AST+M2D** UAR ~0.73).

> We aim to match or exceed these references; please see **Results** below and the attached **lab report** for details.

---

## 6) Results (to be reproduced)
Insert your final numbers here after running the code:

| Model | W.acc | UAR | Notes |
|---|---:|---:|---|
| **M2DTransformer (ours)** | `TBD` | `TBD` | multi-site tokens, attention pooling, K=3 heads |
| **Ablation: no distill** | `TBD` | `TBD` | remove KL term |
| **Ablation: single head** | `TBD` | `TBD` | K=1 |
| **Ensemble (ours + M2D repr)** | `TBD` | `TBD` | late fusion |

> Our **full experimental design, architecture, and loss** choices are described in the report.

---

## 7) Quickstart

### 7.1 Environment
```bash
# 1) Create env
conda create -n murmurs python=3.10 -y
conda activate murmurs

# 2) Install deps (choose CUDA build for your GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy librosa soundfile tqdm scikit-learn matplotlib einops
```

### 7.2 Data
1. **Download** CirCor DigiScope data (see paper for details and official sources).  
2. **Organize** as:
```
data/
  circor/
    train/ ... wav
    val/   ... wav
    test/  ... wav
    metadata.csv
```
3. **(Optional) Precompute embeddings** with M2D for faster training.

### 7.3 Train & Evaluate
```bash
# Train
python tools/train.py   --data-root data/circor   --backbone m2d   --epochs 50 --bs 32   --lr 1e-4 --focal-gamma 1.5   --smoothing 0.1 --specaug True   --heads 3 --kl-alpha 0.3

# Evaluate
python tools/eval.py   --data-root data/circor   --ckpt runs/best.ckpt   --metrics wacc uar
```
> Scripts compute **W.acc/UAR** aligned to the literature.

---

---

## 8) Acknowledgements & References
- **Project Report:** *Heart Murmur Classification — Ternary Classification based on M2D and Transformers (Lab Report, May 29, 2025).*  
- **Project Brief / Requirements:** *Audio Pattern Recognition Project* — task description, dataset link, and reporting instructions.  
- **SOTA (Pretrained Audio Reps on CirCor):** *Exploring Pre-trained General-purpose Audio Representations for Heart Murmur Detection (2024)* — M2D results and protocol.  
- **Dataset Paper:** *The CirCor DigiScope Dataset: From Murmur Detection to Murmur Classification (JBHI 2022)* — data collection, annotation, and statistics.

---

## License
This project is released under the **MIT License**. See `LICENSE` for details.

---

<p align="center">
Made with ❤️ for clinical audio intelligence.
</p>
