"""
Medical Transcriptions Classification — Interactive Demo
Run locally:   streamlit run app/streamlit_app.py
Deploy free:   https://share.streamlit.io  (Streamlit Community Cloud)
"""

import re
import streamlit as st
import spacy

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Medical Transcription Classifier",
    page_icon="🏥",
    layout="centered"
)

# ----------------------------------------------------------------------------
# Load spaCy model (cached so it only loads once)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_model()

# ----------------------------------------------------------------------------
# Medical abbreviations
# ----------------------------------------------------------------------------
MEDICAL_ABBREV = {
    'pt': 'patient', 'pts': 'patients', 'hx': 'history', 'dx': 'diagnosis',
    'tx': 'treatment', 'rx': 'prescription', 'sx': 'symptoms', 'fx': 'fracture',
    'c/o': 'complains of', 'y/o': 'year old', 'yo': 'year old', 'w/': 'with',
    'w/o': 'without', 'h/o': 'history of', 's/p': 'status post', 'r/o': 'rule out',
    'sob': 'shortness of breath', 'cp': 'chest pain', 'bp': 'blood pressure',
    'hr': 'heart rate', 'wbc': 'white blood cell', 'rbc': 'red blood cell',
}

NEGATION_WORDS = {
    'no', 'not', 'never', 'none', 'nobody', 'nothing', 'neither', 'nowhere',
    'denies', 'denied', 'deny', 'without', 'lack', 'lacks', 'lacking',
    'absent', 'negative', 'free', 'unremarkable'
}

SPECIALTY_KEYWORDS = {
    "Cardiovascular / Pulmonary": ["heart", "artery", "chest", "catheter", "pressure",
                                     "cardiac", "coronary", "vessel", "valve", "lung"],
    "Orthopedic": ["bone", "joint", "knee", "fracture", "ligament", "hip",
                   "cartilage", "tendon", "spine", "incision"],
    "Radiology": ["image", "scan", "contrast", "evidence", "finding", "view",
                  "mri", "ct", "xray", "imaging"],
    "Surgery": ["incision", "suture", "procedure", "anesthesia", "operative",
                "room", "drape", "tissue", "surgical"],
}

# ----------------------------------------------------------------------------
# Pipeline functions
# ----------------------------------------------------------------------------
def expand_abbreviations(text):
    expanded = text
    for abbrev, full in MEDICAL_ABBREV.items():
        pattern = r'\b' + re.escape(abbrev) + r'\b'
        expanded = re.sub(pattern, full, expanded, flags=re.IGNORECASE)
    return expanded


def detect_negation(text):
    doc = nlp(text)
    results = []
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN']:
            is_negated, negation_source = False, None
            for ancestor in token.ancestors:
                if ancestor.text.lower() in NEGATION_WORDS:
                    is_negated, negation_source = True, ancestor.text
                    break
            if not is_negated:
                for child in token.children:
                    if child.text.lower() in NEGATION_WORDS or child.dep_ == 'neg':
                        is_negated, negation_source = True, child.text
                        break
            if not is_negated:
                for sibling in token.head.children:
                    if sibling.text.lower() in NEGATION_WORDS and sibling != token:
                        is_negated, negation_source = True, sibling.text
                        break
            results.append({
                'entity': token.text,
                'status': 'Negated' if is_negated else 'Present',
                'trigger': negation_source
            })
    return results


def extract_entities(text):
    doc = nlp(text)
    nouns = [t.lemma_.lower() for t in doc if t.pos_ in ['NOUN', 'PROPN'] and t.is_alpha and len(t.lemma_) > 2]
    adjectives = [t.lemma_.lower() for t in doc if t.pos_ == 'ADJ' and t.is_alpha]
    return nouns, adjectives


def predict_specialty_demo(text):
    """
    Lightweight keyword-overlap classifier for the live demo.
    (The full notebook uses a trained TF-IDF + Logistic Regression model —
    swap this out with a pickled model for production use.)
    """
    text_lower = text.lower()
    scores = {}
    for specialty, keywords in SPECIALTY_KEYWORDS.items():
        scores[specialty] = sum(text_lower.count(kw) for kw in keywords)

    total = sum(scores.values())
    if total == 0:
        return None, scores

    probs = {k: v / total for k, v in scores.items()}
    predicted = max(probs, key=probs.get)
    return predicted, probs


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("🏥 Medical Transcription Classifier")
st.markdown(
    "An NLP pipeline that cleans clinical text, detects negation, and "
    "routes a report to the likely medical specialty. "
    "[View the full project on GitHub](https://github.com/nagarjun1302/medical-transcription-classification)"
)

st.divider()

tab1, tab2, tab3 = st.tabs(["🔍 Try It Live", "🚫 Negation Detection", "📊 About the Project"])

# --- Tab 1: Live classification ---
with tab1:
    st.subheader("Paste a clinical note")

    example = st.selectbox(
        "Or pick an example:",
        [
            "— select an example —",
            "Patient underwent coronary artery bypass grafting. Post-operative recovery was uncomplicated. Patient is stable with good cardiac output and no chest pain.",
            "Patient presents with right knee pain following a fall. X-ray reveals a fracture of the distal femur. Orthopedic surgery recommended for internal fixation.",
            "CT scan of the abdomen shows no evidence of mass or free fluid. Findings are consistent with normal imaging study.",
        ]
    )

    default_text = "" if example.startswith("—") else example
    user_text = st.text_area("Clinical text:", value=default_text, height=150,
                              placeholder="e.g. Pt is a 45 y/o male c/o chest pain and sob...")

    if st.button("Analyze", type="primary"):
        if not user_text.strip():
            st.warning("Please enter some text first.")
        else:
            expanded = expand_abbreviations(user_text)
            nouns, adjectives = extract_entities(expanded)
            specialty, scores = predict_specialty_demo(expanded)

            st.markdown("#### 🏷️ Predicted Specialty")
            if specialty:
                st.success(f"**{specialty}**")
                cols = st.columns(len(scores))
                for col, (spec, score) in zip(cols, sorted(scores.items(), key=lambda x: -x[1])):
                    col.metric(spec.split(" / ")[0].split(" ")[0], f"{score:.0%}")
            else:
                st.info("Not enough specialty-specific vocabulary detected to make a confident prediction.")

            st.markdown("#### 🔬 Extracted Entities")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Nouns (symptoms / body parts)**")
                st.write(", ".join(sorted(set(nouns))[:20]) or "—")
            with c2:
                st.markdown("**Adjectives (descriptors)**")
                st.write(", ".join(sorted(set(adjectives))[:20]) or "—")

            st.markdown("#### 🚫 Negation Check")
            negations = detect_negation(user_text)
            negated = [n for n in negations if n['status'] == 'Negated']
            if negated:
                for n in negated:
                    st.markdown(f"❌ **{n['entity']}** — Negated (trigger: *{n['trigger']}*)")
            else:
                st.markdown("✅ No negated entities detected — symptoms appear present.")

# --- Tab 2: Negation playground ---
with tab2:
    st.subheader("Negation Detection Playground")
    st.markdown(
        "In medicine, *\"no fever\"* is the opposite of *\"fever\"*. "
        "This tool uses spaCy's dependency parser to check whether a symptom "
        "is asserted or negated in the sentence."
    )

    neg_examples = [
        "Patient denies chest pain.",
        "No fever or chills reported.",
        "Patient has severe headache.",
        "Heart sounds normal without murmur.",
        "Denies any recent trauma.",
    ]

    picked = st.selectbox("Try an example sentence:", ["— type your own below —"] + neg_examples)
    neg_input = st.text_input("Sentence:", value="" if picked.startswith("—") else picked)

    if neg_input:
        results = detect_negation(neg_input)
        if results:
            for r in results:
                icon = "❌" if r['status'] == 'Negated' else "✅"
                trigger = f" (trigger: *{r['trigger']}*)" if r['trigger'] else ""
                st.markdown(f"{icon} **{r['entity']}** — {r['status']}{trigger}")
        else:
            st.info("No noun entities detected in this sentence.")

# --- Tab 3: About ---
with tab3:
    st.subheader("About This Project")
    st.markdown(
        """
Hospitals process thousands of discharge summaries daily. This project builds
an NLP pipeline to automate the classification and initial understanding of
clinical text, so senior doctors spend less time on manual triage.

**Pipeline:**
1. **Preprocessing** — regex header removal, abbreviation expansion, lemmatization
2. **Entity Extraction** — POS tagging to pull out symptoms and descriptors
3. **Negation Analysis** — dependency parsing to catch "denies", "no", "without"
4. **Classification** — TF-IDF + Logistic Regression trained on MTSamples data

**Full project results (from the notebook, not this lightweight demo):**
- 73% classification accuracy across 4 specialties
- 100% accuracy on negation test cases
- Trained on 2,103 real medical transcriptions

> ⚠️ **Note:** This live demo uses a simplified keyword-based classifier for
> speed and to avoid shipping a large trained model file. The notebook contains
> the full TF-IDF + Logistic Regression pipeline with proper train/test evaluation.

**Dataset:** [MTSamples on Kaggle](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions)
        """
    )

st.divider()
st.caption("Built with spaCy, scikit-learn, and Streamlit · Not for clinical use")
