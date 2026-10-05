# EcoSort: Intelligent Waste Management Assistant

**Module 8 Summative Lab: Neural Networks and Similar Models**

An AI assistant that helps Metro City residents dispose of waste correctly. A resident uploads a photo **or** types a description of an item. The assistant identifies the waste category and returns recycling instructions grounded in the city's waste policies, with the supporting policy documents cited.

The system combines three models:

| Component | Model | Task |
|---|---|---|
| Image classifier | EfficientNetB0 (transfer learning, fine-tuned) | Identify the waste material from a photo |
| Text classifier | DistilBERT (all layers fine-tuned) | Classify a written description of an item |
| Instruction generator | Flan-T5-small + retrieval (MiniLM + FAISS) | Generate recycling instructions grounded in policy documents (RAG) |

---

## Results

| Component | Metric | Result |
|---|---|---|
| **CNN** (713 test images) | Accuracy / macro-F1 | **87.4%** / 0.883 |
| **Text classifier** (741 test descriptions) | Accuracy / macro-F1 | **100%** / 1.000 |
| Text classifier (hand-written challenge set) | Accuracy on unambiguous items | **92.3%** (12 of 13) |
| **Retrieval** (item queries, no category named) | Hit@3 / MRR | **0.800** / 0.704 (TF-IDF baseline: 0.556 / 0.534) |
| **RAG generation** (27 evaluation queries) | Grounding / wrong-category leakage / fallback rate | **0.984** / **0.000** / 3.7% |
| Generator fine-tuning | ROUGE-L vs. zero-shot baseline | 0.235 → **0.789** |
| **Integrated assistant** (72 held-out requests) | Correct routing | **26 of 27** images, **45 of 45** descriptions |
| Integrated assistant | Edge cases passed | **11 of 11** |
| Integrated assistant | Response time (CPU) | ~7 s new answer, ~0.09 s cached |

The text classifier's perfect test score reflects how templated the synthetic descriptions are. Its vocabulary is only 305 words, and 19.7% of test items have a near-duplicate in training. The hand-written challenge set, with regional terms, typos and multi-material items, is the more realistic measure. See the notebook, Part 3.

---

## How it works

```
                 ┌──────────────────────────┐
  image  ──────▶ │ EfficientNetB0 CNN       │──┐  category + confidence + top-3
                 └──────────────────────────┘  │
                                               │   ┌───────────────────────────────┐
                                               ├──▶│ Retrieve policy chunks        │
                 ┌──────────────────────────┐  │   │ (MiniLM + FAISS, filtered by  │
  text   ──────▶ │ DistilBERT               │──┘   │  category)                    │
                 │ + out-of-distribution    │      │ Generate with Flan-T5         │──▶ instructions
                 │   check                  │      │ Verify each sentence against  │──▶ cited policy sources
                 └──────────────────────────┘      │ the policy; fall back to      │──▶ warnings / flags
                              ▲                    │ policy text if unverifiable   │
                              │                    └───────────────────────────────┘
                              └──── resident feedback (corrections, helpfulness) ◀──┘
```

**Key design choices**

- **Leak-free data splits.** Stratified 70/15/15 splits on file paths, with duplicates removed before splitting, so no image or description appears in more than one split.
- **Class-imbalance handling.** Balanced class weights (Plastic has 2.9× as many images as Textile Trash) plus data augmentation inside the model.
- **Model selection by experiment.** Four CNN configurations were compared (head width and depth, dropout, EfficientNetB0 vs. MobileNetV2), along with two DistilBERT learning-rate/dropout settings and a TF-IDF baseline.
- **Section-aware retrieval corpus.** 85 chunks: policy documents split at their section headings, plus each category's resident disposal instructions and common confusion cases from the descriptions dataset.
- **Factual-consistency safeguards.** Retrieval is filtered to the predicted category. Each generated sentence is kept only if it is supported by that category's policy text. If nothing can be verified, the assistant returns the policy text directly.
- **Uncertainty handling.** The assistant warns the resident when image confidence is low, or when a description is unlike anything in the training data.
- **Feedback loop.** Resident corrections are logged to CSV, applied immediately to repeated descriptions, and exported as new training data.

---

## Repository structure

```
.
├── waste_management_summative_final.ipynb   # Full project notebook (with outputs)
├── waste_descriptions.csv                   # 5,000 generated waste descriptions
├── waste_policy_documents.json              # 14 generated municipal policy documents
├── requirements.txt
├── .gitignore
└── README.md
```

The RealWaste images are **not** included in this repository because of their size. See *Data setup* below.

---

## Data setup

1. Download the **RealWaste** dataset (4,752 images, 9 classes) from the UCI Machine Learning Repository.
2. Unzip it and place the nine class folders inside a folder named `realwaste/` next to the notebook:

```
project/
├── waste_management_summative_final.ipynb
├── waste_descriptions.csv
├── waste_policy_documents.json
└── realwaste/
    ├── Cardboard/
    ├── Food Organics/
    ├── Glass/
    ├── Metal/
    ├── Miscellaneous Trash/
    ├── Paper/
    ├── Plastic/
    ├── Textile Trash/
    └── Vegetation/
```

The notebook finds the dataset folder automatically. The pretrained models (EfficientNetB0, DistilBERT, all-MiniLM-L6-v2 and Flan-T5-small) download automatically on the first run.

---

## Installation and running

Python 3.10 is recommended.

```bash
# Create and activate an environment
conda create -n ecosort_env python=3.10 -y
conda activate ecosort_env

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook waste_management_summative_final.ipynb
```

Then choose **Kernel → Restart & Run All**.

**Runtime:** a full run takes roughly **2–2.5 hours on a CPU**. The longest steps are the CNN experiments and training (about 50 minutes), DistilBERT fine-tuning (about 15 minutes), Flan-T5 fine-tuning (about 10 minutes) and the decoding-strategy experiments (about 30 minutes). To shorten the run, lower the epoch counts in the configuration cell (section 0.2). The notebook sets `TF_USE_LEGACY_KERAS=1` so the Hugging Face TensorFlow models work with Keras 2 (`tf-keras`).

---

## Notebook contents

| Part | Content |
|---|---|
| **0. Setup** | Imports, run configuration, random seeds |
| **1. Data exploration** | Full image audit (resolution, lighting, sharpness, background, integrity, duplicates). Text vocabulary and duplicate analysis. Policy structure. Data pipelines and splits. RAG corpus preparation |
| **2. CNN** | Augmentation, class weights, architecture experiments, head training, fine-tuning, learning curves, confusion matrices, error and confidence analysis |
| **3. Text classification** | WordPiece tokenisation, TF-IDF baseline, DistilBERT experiments and training curves, near-duplicate check, challenge set, prediction function |
| **4. RAG** | Embedding index, retrieval benchmark, zero-shot baselines, Flan-T5 fine-tuning, decoding experiments, sentence-level verification, final evaluation with readability metrics |
| **5. Integrated assistant** | Unified interface, error handling, caching, feedback loop, standard tests, end-to-end evaluation on 72 requests, edge cases |

Each part ends with an interpretation section that discusses the results.

---

## Limitations

- **Synthetic text data.** The descriptions are templated, so test accuracy overstates real-world performance. Misspellings (e.g. "PLASTIK WATTER BOTLE") and regional terms (e.g. "crisp packet") are misclassified, although the out-of-distribution check now flags them.
- **Single-facility images.** RealWaste photos come from one facility, so residents' phone photos (different backgrounds and lighting) may score lower. Visually similar pairs remain the main errors: Plastic ↔ Glass/Metal and Paper ↔ Cardboard.
- **Confident mistakes.** Confidence warnings catch uncertain predictions but not confident errors between similar materials.
- **Small generator.** Flan-T5-small sometimes reproduces correct policy facts from memory rather than from the retrieved text. Verification removes these sentences, which keeps answers grounded but makes them less complete.
- **Grounding metric.** Word-overlap checks can't detect negation or numeric errors.
- **Single jurisdiction.** Policies cover one city and may change, so every answer shows the policy's effective date.

## Next steps

- Always include each category's collection and preparation sections in the generator's context.
- Switch to deterministic beam-search decoding for reproducible answers.
- Calibrate classifier confidence and ask residents to confirm when the top two categories are close.
- Retrain on real resident photos and descriptions collected through the feedback loop.
- Evaluate a larger generator such as Flan-T5-base.

---

## Acknowledgements

- **RealWaste dataset:** S. Single, S. Iranmanesh and R. Raad, "RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning," *Information*, 2023. Available from the UCI Machine Learning Repository.
- **Waste descriptions and policy documents:** generated datasets provided with the course lab.
- **Pretrained models:** EfficientNetB0 (Keras Applications), DistilBERT (`distilbert-base-uncased`), `all-MiniLM-L6-v2` (sentence-transformers), `google/flan-t5-small` (Hugging Face), with FAISS for similarity search.

---

*Author: [Your Name]. Submitted for the Module 8 Summative Lab.*
