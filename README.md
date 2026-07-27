# 🏥 Medical Transcriptions Classification

> An NLP pipeline that automatically classifies medical transcriptions into specialties, extracts clinical entities, and detects negation ("no fever" vs. "fever") - built to help hospitals triage discharge summaries and clinical notes faster.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![spaCy](https://img.shields.io/badge/spaCy-NLP-09a3d5.svg)](https://spacy.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-f89939.svg)](https://scikit-learn.org/)

**[🚀 Live Demo](https://medical-transcription-classifier.streamlit.app/)** &nbsp;|&nbsp; **[📓 Notebook](notebook/medical_transcriptions_classification.ipynb)** &nbsp;|&nbsp; **[📄 Full Report](report/Medical_Transcriptions_Classification_Report.docx)**

---

## 📌 Problem

Hospitals process thousands of discharge summaries and clinical notes daily. Senior doctors spend hours manually reading these to categorize patients by specialty and flag critical symptoms - a slow, expensive, error-prone process.

This project builds an automated NLP pipeline that:
1. **Cleans** raw medical text and expands clinical abbreviations
2. **Extracts** key medical entities (symptoms, body parts, descriptors)
3. **Detects negation** - understanding that "denies chest pain" ≠ "has chest pain"
4. **Classifies** each report into the correct medical specialty


---

## 📊 Results

| Metric | Value |
|---|---|
| **Classification Accuracy** | 73% |
| **Specialties Classified** | 4 (Surgery, Cardiovascular/Pulmonary, Orthopedic, Radiology) |
| **Dataset Size** | 2,103 transcriptions (876 after class balancing) |
| **Negation Detection** | 100% on test cases |

**Confusion Matrix Insight:** Surgery is most often confused with Orthopedic (23.6%) — both involve surgical procedures on bones/joints, sharing terms like *incision* and *procedure*.

---

## 🧠 How It Works

```
Raw Transcription
      │
      ▼
┌─────────────────────┐
│ 1. Preprocessing     │  Regex header removal → abbreviation expansion → lemmatization
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ 2. Entity Extraction │  POS tagging (spaCy) → nouns = symptoms, adjectives = descriptors
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ 3. Negation Analysis │  Dependency parsing → checks ancestors/children/siblings for
│                       │  negation cues (no, denies, without, absent...)
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ 4. Classification    │  TF-IDF (5,000 features) → Logistic Regression → specialty
└─────────────────────┘
```

---

## 🛠️ Tech Stack

- **Language:** Python 3.8+
- **NLP:** spaCy (`en_core_web_sm`) - POS tagging, dependency parsing, lemmatization
- **ML:** scikit-learn - TF-IDF vectorization, Logistic Regression, Naive Bayes
- **Data:** pandas, numpy
- **Visualization:** matplotlib, seaborn, wordcloud
- **Dataset:** [MTSamples (Medical Transcriptions)](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions) via Kaggle

---

## 🧪 Example: Negation Detection

```python
detect_negation("Patient denies chest pain.")
# → [{'Entity': 'pain', 'Status': 'Negated', 'Negation_Word': 'denies'}]

detect_negation("Patient has severe headache.")
# → [{'Entity': 'headache', 'Status': 'Present', 'Negation_Word': None}]
```

This distinction matters clinically - a keyword search for "pain" would incorrectly flag both sentences as positive for pain.

---

## 📈 Vocabulary Analysis (Top Nouns by Specialty)

| Rank | Cardiovascular / Pulmonary | Orthopedic |
|---|---|---|
| 1 | artery | pain |
| 2 | chest | incision |
| 3 | catheter | knee |
| 4 | pressure | position |
| 5 | heart | bone |

Distinct vocabularies confirm that specialty-specific terminology is a strong classification signal.

---
