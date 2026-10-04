# TruthGuard 🛡️
### Evidence-Aware Fake News Detection for Multilingual & Code-Mixed Social Media

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Models](https://img.shields.io/badge/Models-IndicBERTv2%20%7C%20MuRIL%20%7C%20Gemini-green.svg)](https://huggingface.co/)
[![Architecture](https://img.shields.io/badge/Architecture-Hybrid%20Transformer%20%2B%20IR%20Pipeline-purple.svg)](#system-architecture)

**Author & Maintainer:** [Pranav Kalra](https://github.com/pranav-kalra22)

---

## 📌 Overview

**TruthGuard** is an advanced disinformation detection framework tailored for multilingual and code-mixed social media text (specifically Tamil-English / "Tanglish"). Code-mixed content on platforms like YouTube, WhatsApp, and Twitter presents unique challenges: phonetic transliteration, non-standard spelling, and colloquial linguistic blending break conventional NLP pipelines.

TruthGuard addresses these challenges with a **hybrid architecture** combining two complementary approaches:

1. **Evidence-Aware Verification Engine (`src/verification/`)**:
   - An Information Retrieval (IR) and LLM-grounded verification pipeline.
   - Extracts factual assertions from noisy social media text, generates multi-perspective search queries, retrieves live web evidence, ranks candidate sources (by relevance, domain authority, and freshness), and produces an explainable verdict (`REAL`, `FAKE`, `MISLEADING`, or `UNVERIFIED`) backed by direct evidence citations.
   - **Key Design Principle:** Retrieval modules *never* evaluate truth on their own—verdicts are strictly grounded in verifiable, cited sources.

2. **Fine-Tuned Baseline Classifiers (`src/fake_news_classifier/`)**:
   - Transformer sequence classification baselines utilizing **IndicBERTv2** and **MuRIL**.
   - Evaluated on Tamil news headline authenticity benchmarks, achieving **99.0%** (IndicBERTv2) and **98.6%** (MuRIL) test accuracy.

3. **Silver-Label Language Identification (LID) Pipeline (`src/`)**:
   - A token-level language annotation pipeline (`TA`, `EN`, `MIX`, `NE`, `O`) that converts raw comment streams into standardized CoNLL format for low-resource model fine-tuning.

---

## 🏗️ System Architecture

```
Social Media Comment / Claim (e.g., Tamil-English Tanglish)
                           │
       ┌───────────────────┴───────────────────┐
       ▼                                       ▼
[ APPROACH A: Fast Static Baseline ]   [ APPROACH B: Evidence-Aware Verification ]
  Transformer Sequence Classifier        6-Stage Retrieval & Grounding Engine
  (MuRIL / IndicBERTv2)
       │                                 [1] Claim Extraction (LLM)
       ▼                                 [2] Search Query Generation (LLM)
  Verdict: REAL / FAKE                   [3] Web Retrieval (Serper / Google API)
                                         [4] Webpage Content Extraction (Trafilatura)
                                         [5] Evidence Ranking (BM25 / TF-IDF + Authority + Freshness)
                                         [6] Grounded LLM Verification (Strict JSON Citing Sources)
                                               │
                                               ▼
                                         Verdict: REAL / FAKE / MISLEADING / UNVERIFIED
                                         (With Evidence IDs & Traceable Explanation)
```

---

## 📂 Project Structure

```
TruthGuard/
├── .env.example              # Configuration template for API keys
├── .gitignore                # Comprehensive security & artifact filters
├── LICENSE                   # MIT License
├── README.md                 # Project documentation
├── requirements.txt          # Core dependencies (IR pipeline & inference)
├── requirements-train.txt    # Training dependencies (PyTorch & HuggingFace)
├── data/
│   ├── raw/                  # Raw input datasets (Tamil news & comments)
│   ├── processed/            # Cleaned and deduplicated corpora
│   ├── annotated/            # Checkpointed annotations & token exports
│   ├── fake_news/            # Train, dev, and test dataset splits
│   └── lid_dataset/          # CoNLL-formatted LID splits
├── docs/                     # Detailed architectural specifications
├── presentation/             # Project presentation slide deck
├── prompts/                  # Prompt engineering templates
└── src/
    ├── config.py             # Global configurations and path management
    ├── utils.py              # Text cleaning and preprocessing utilities
    ├── preprocess.py         # YouTube comment cleaning pipeline
    ├── run_annotation.py     # Checkpointed silver-label annotation runner
    ├── tokens_to_csv.py      # Token extraction and flattening
    ├── convert_to_lid_format.py # CoNLL export generator
    ├── review_sample.py      # Quality assurance & human validation export
    ├── fake_news_classifier/ # MuRIL & IndicBERTv2 training & comparison
    │   ├── preprocess.py
    │   ├── train_classifier.py
    │   └── compare_classifiers.py
    └── verification/         # Evidence-Aware Verification Pipeline
        ├── claim_extraction.py
        ├── query_generation.py
        ├── web_search.py
        ├── webpage_extraction.py
        ├── evidence_ranking.py
        ├── llm_verifier.py
        ├── pipeline.py
        └── run_demo.py
```

---

## ⚡ Quickstart & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/pranav-kalra22/TruthGuard.git
cd TruthGuard
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Open `.env` and configure your credentials:
```env
# Free Gemini API key from https://aistudio.google.com/apikey
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-flash-lite-latest

# Free search API key from https://serper.dev (2,500 free queries)
SEARCH_PROVIDER=serper
SERPER_API_KEY=your_serper_api_key_here
```
> **Security Notice:** The `.env` file is strictly excluded by `.gitignore`. Never commit API keys or private credentials to git.

---

## 🚀 Running TruthGuard

### 1. Evidence-Aware Verification (Demo)

Run verification on any factual assertion or code-mixed social media claim:

```bash
# Example: Transliterated Tamil-English viral claim
python src/verification/run_demo.py --text "UNESCO Indian national anthem ah world best anthem nu declare pannitanga nu news viral aaguthu"

# Example: Banking / financial rumor
python src/verification/run_demo.py --text "RBI has released a new 5000 currency note with GPS tracking"

# Interactive mode (prompts for input in terminal)
python src/verification/run_demo.py
```

#### Sample Verification Output
```text
========================================================================
INPUT: UNESCO Indian national anthem ah world best anthem nu declare pannitanga nu news viral aaguthu
========================================================================

[1] CLEANED TEXT
    UNESCO Indian national anthem ah world best anthem nu declare pannitanga nu news viral aaguthu

[2] CLAIM EXTRACTION
    has_claim:        True
    normalized_claim: UNESCO has declared the Indian national anthem as the world's best anthem.
    reasoning:        The comment reports a viral news claim asserting that UNESCO ranked the anthem.

[3] GENERATED SEARCH QUERIES
    - UNESCO Indian national anthem best claim fact check
    - did UNESCO declare Jana Gana Mana best
    - UNESCO best national anthem hoax truth

[4] RETRIEVED URLs (8 sources identified)

[5] EXTRACTED EVIDENCE (8/8 article bodies parsed successfully)

[6] RANKED EVIDENCE (Ranked by Relevance + Domain Authority + Freshness)
    E1 (timesofindia.indiatimes.com)  final=0.5970
    E2 (vishvasnews.com)              final=0.5626
    E3 (bbc.com)                      final=0.5299

[7] FINAL VERDICT
    verdict:           FAKE
    confidence:        1.0
    evidence_relation: CONTRADICTS
    evidence_ids:      ['E1', 'E2', 'E3']
    explanation:       The claim that UNESCO declared the Indian national anthem as the world's best anthem is false. Multiple fact-checking sources and UNESCO officials confirm that UNESCO has never ranked national anthems, and the claim is an old internet hoax circulating since 2008.
```

---

### 2. Baseline Classifier Pipeline

Prepare datasets and benchmark sequence classifiers:

```bash
# 1. Preprocess headline splits (train/dev/test)
python src/fake_news_classifier/preprocess.py

# 2. Train baseline models (Requires GPU / PyTorch; install requirements-train.txt)
python src/fake_news_classifier/train_classifier.py --model indicbert-v2
python src/fake_news_classifier/train_classifier.py --model muril

# 3. Compare model performance
python src/fake_news_classifier/compare_classifiers.py
```

---

### 3. Language Identification (LID) Data Pipeline

Extract, clean, and structure code-mixed social media comments into CoNLL format:

```bash
# 1. Clean raw YouTube comments
python src/preprocess.py

# 2. Run silver-label annotation (resumable)
python src/run_annotation.py --limit 50

# 3. Flatten tokens to CSV
python src/tokens_to_csv.py

# 4. Generate train/dev/test CoNLL dataset
python src/convert_to_lid_format.py

# 5. Build human review validation spreadsheet
python src/review_sample.py --n 50
```

---

## 📊 Experimental Results

| Model / Pipeline | Approach | Accuracy | Macro F1 | Notes |
|---|---|---|---|---|
| **IndicBERTv2** | Sequence Classifier | **99.0%** | **0.990** | Evaluated on held-out test split |
| **MuRIL** | Sequence Classifier | **98.6%** | **0.986** | Multilingual BERT baseline |
| **TruthGuard IR Engine** | Evidence-Aware LLM | — | — | Zero-shot open-domain verification with cited evidence |

---

## 🔒 Security Best Practices

- **Zero Secret Leakage:** `.env` and all credential variants are strictly ignored via `.gitignore`.
- **Placeholder Templates:** Only `.env.example` with non-functional placeholders is tracked in version control.
- **Fail-Safe Retrieval:** If API credentials or search providers are unavailable, the pipeline terminates explicitly at the corresponding stage rather than fabricating information.

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Pranav Kalra**  
GitHub: [@pranav-kalra22](https://github.com/pranav-kalra22)
