"""Streamlit Interactive Application for Spam Message Classifier.

CodeNova AI/ML Internship Program - Task 2: Spam Message Classifier
"""

import json
from pathlib import Path
from typing import Dict, List

import streamlit as st

# Configure page layout and aesthetics
st.set_page_config(
    page_title="SMS Spam Message Classifier | CodeNova",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Relative / project imports
from src.config import (
    COMPARISON_CSV_PATH,
    CONFUSION_MATRIX_PNG,
    DATA_AUDIT_JSON_PATH,
    DATASET_CITATION,
    DATASET_DOI,
    METRICS_JSON_PATH,
    ROC_PR_CURVES_PNG,
    SAVED_MODEL_PATH,
)
from src.inference import (
    ModelLoadError,
    ModelNotFoundError,
    PredictionResult,
    SpamClassifierService,
)

# Curated benchmark sample messages for interactive demonstration
SAMPLE_MESSAGES = {
    "-- Select a sample message --": "",
    "Spam: Urgent Prize Notification": (
        "URGENT! You have won a £1,000 cash prize or a holiday voucher! "
        "Claim by texting PRIZE to 88088 now. Terms apply. 150p/msg."
    ),
    "Spam: Bank Security Phishing Alert": (
        "ALERT: Your bank account has been temporarily locked due to suspicious login attempts. "
        "Verify your identity immediately at http://verify-secure-bank.co.uk to unlock."
    ),
    "Spam: Customer Call-back Promotion": (
        "FREE Ringtone! Reply YES to get top hits directly on your handset! "
        "Call 09061743810 now. Stop to cancel."
    ),
    "Ham: Casual Personal Coordination": (
        "Hey! Are you still free for lunch today at 1? Let me know if you want me to pick you up."
    ),
    "Ham: Work & Schedule Update": (
        "Hi doctor, just confirming our appointment for tomorrow at 10:30 AM. Thank you!"
    ),
    "Ham: Casual Message with 'Free' (Potential False Positive Trap)": (
        "Are you free tonight? We are planning to grab pizza and catch up on the project."
    ),
}


@st.cache_resource(show_spinner=False)
def get_inference_service() -> SpamClassifierService:
    """Loads and caches the inference pipeline."""
    return SpamClassifierService()


def load_metrics_report() -> Dict:
    """Loads the saved test evaluation metrics if available."""
    if METRICS_JSON_PATH.exists():
        try:
            with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def main() -> None:
    """Streamlit Application Main Controller."""
    st.title("🛡️ SMS Spam Message Classifier")
    st.caption("CodeNova AI/ML Internship Program — Task 2: Machine Learning & NLP Pipeline")

    # Sidebar: Model status, architecture, dataset metadata
    with st.sidebar:
        st.header("Project Architecture")
        st.markdown(
            """
            - **Vectorization**: TF-IDF (unigrams & bigrams, sublinear TF)
            - **Algorithms**: Naive Bayes vs. Logistic Regression
            - **Zero Leakage**: Strict Pipeline encapsulation & isolated test split
            - **Provenance**: Official UCI SMS Spam Collection
            """
        )

        st.divider()
        st.subheader("Dataset Attribution")
        st.markdown(f"**Citation:**\n*{DATASET_CITATION}*")
        st.markdown(f"[UCI Dataset Page]({DATASET_DOI})")

        st.divider()
        st.subheader("Model Status")
        if SAVED_MODEL_PATH.exists():
            st.success("✓ Trained Pipeline Ready")
        else:
            st.error("✗ Model artifact not found")
            st.info("Run `python train.py` in the terminal to train and save the pipeline.")

    # Check model availability safely without leaking server paths
    try:
        service = get_inference_service()
    except ModelNotFoundError:
        st.error(
            "⚠️ **Trained Model Pipeline Not Found**\n\n"
            "The model artifact has not been trained yet. Please train the model by executing:\n\n"
            "```bash\npython train.py\n```\n\n"
            "Once training is complete, refresh this application."
        )
        st.stop()
    except ModelLoadError:
        st.error(
            "⚠️ **Model Artifact Corrupted**\n\n"
            "The saved model could not be deserialized. Please recreate it by executing:\n\n"
            "```bash\npython train.py\n```"
        )
        st.stop()

    # Main Tabs
    tab_predict, tab_metrics, tab_limitations = st.tabs(
        ["🔍 Interactive Classifier", "📊 Test Set Metrics", "⚠️ Limitations & Trade-offs"]
    )

    # -------------------------------------------------------------
    # TAB 1: INTERACTIVE PREDICTION
    # -------------------------------------------------------------
    with tab_predict:
        st.markdown(
            "Enter any custom SMS or text message below, or select a pre-configured sample "
            "to test the classifier's prediction and probability score."
        )

        # Sample dropdown
        selected_sample_key = st.selectbox(
            "Choose a curated test sample:",
            list(SAMPLE_MESSAGES.keys()),
            index=0,
        )
        sample_text = SAMPLE_MESSAGES[selected_sample_key]

        # Text input area
        user_message = st.text_area(
            "Message Text to Analyze:",
            value=sample_text,
            height=120,
            placeholder="Type or paste your SMS message here...",
            help="Maximum 5,000 characters.",
        )

        char_count = len(user_message)
        word_count = len(user_message.split())
        st.caption(f"Length: {char_count} characters | {word_count} tokens")

        col1, col2 = st.columns([1, 4])
        with col1:
            predict_clicked = st.button("Classify Message", type="primary", width="stretch")

        if predict_clicked:
            if not user_message.strip():
                st.warning("Please enter a non-empty message to classify.")
            else:
                try:
                    result: PredictionResult = service.predict(user_message)

                    st.markdown("---")
                    st.subheader("Classification Result")

                    # Display verdict banner
                    if result.is_spam:
                        st.error(
                            f"🚨 **PREDICTED CLASS: SPAM**\n\n"
                            f"The classifier identified promotional, phishing, or financial spam cues."
                        )
                    else:
                        st.success(
                            f"✅ **PREDICTED CLASS: HAM (LEGITIMATE)**\n\n"
                            f"The classifier determined this message appears to be ordinary communication."
                        )

                    # Metrics Columns
                    m_col1, m_col2, m_col3 = st.columns(3)
                    with m_col1:
                        st.metric(
                            label="Confidence Score",
                            value=f"{result.confidence * 100:.1f}%",
                        )
                    with m_col2:
                        st.metric(
                            label="Spam Probability",
                            value=f"{result.spam_probability * 100:.1f}%",
                        )
                    with m_col3:
                        st.metric(
                            label="Inference Latency",
                            value=f"{result.processing_time_ms:.2f} ms",
                        )

                    # Visual Probability Meter
                    st.markdown("**Spam Probability Meter:**")
                    st.progress(min(max(result.spam_probability, 0.0), 1.0))

                    # Safety advisory banner
                    st.info(
                        "ℹ️ **Safety Note:** Statistical NLP classifiers can make mistakes. "
                        "Never rely solely on automated filters for critical security or personal decisions."
                    )

                except ValueError as val_err:
                    st.error(f"Validation error: {val_err}")
                except Exception as exc:
                    st.error("An unexpected error occurred during prediction.")

    # -------------------------------------------------------------
    # TAB 2: BENCHMARK & TEST METRICS
    # -------------------------------------------------------------
    with tab_metrics:
        st.subheader("Held-Out Test Set Performance")
        metrics = load_metrics_report()

        if metrics:
            st.markdown(
                f"Evaluation performed on **{metrics.get('sample_count', 0):,} isolated test messages** "
                f"(stratified 20% split, random seed = 42)."
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Overall Accuracy", f"{metrics.get('accuracy', 0)*100:.2f}%")
            c2.metric("Macro-F1 Score", f"{metrics.get('macro_f1', 0):.4f}")
            c3.metric(
                "Spam Precision",
                f"{metrics.get('spam_metrics', {}).get('precision', 0)*100:.2f}%",
            )
            c4.metric(
                "Spam Recall",
                f"{metrics.get('spam_metrics', {}).get('recall', 0)*100:.2f}%",
            )

            st.markdown("#### Confusion Matrix Breakdown")
            cm = metrics.get("confusion_matrix", {})
            cm_col1, cm_col2 = st.columns(2)
            with cm_col1:
                st.write(
                    f"- **True Negatives (Legitimate Ham caught)**: {cm.get('tn', 0):,}\n"
                    f"- **False Positives (Ham wrongly marked Spam)**: {cm.get('fp', 0):,}\n"
                    f"- **False Negatives (Spam missed as Ham)**: {cm.get('fn', 0):,}\n"
                    f"- **True Positives (Spam correctly caught)**: {cm.get('tp', 0):,}"
                )
            with cm_col2:
                if CONFUSION_MATRIX_PNG.exists():
                    st.image(str(CONFUSION_MATRIX_PNG), caption="Confusion Matrix Heatmap", width="stretch")

            if ROC_PR_CURVES_PNG.exists():
                st.markdown("#### ROC and Precision-Recall Curves")
                st.image(str(ROC_PR_CURVES_PNG), caption="Test ROC & PR Curves", width="stretch")
        else:
            st.info("Metrics report not found. Run `python train.py` to generate evaluation artifacts.")

    # -------------------------------------------------------------
    # TAB 3: LIMITATIONS & ERROR TRADE-OFFS
    # -------------------------------------------------------------
    with tab_limitations:
        st.subheader("Model Limitations & Asymmetric Error Costs")
        st.markdown(
            """
            ### 1. Asymmetric Cost of Errors
            In real-world spam detection, classification errors have strongly unequal consequences:
            - **False Positive (FP - High Cost):** A genuine message (such as a banking OTP, appointment confirmation, or urgent message from a family member) is wrongly sent to Spam. The user misses vital information.
            - **False Negative (FN - Low Cost):** A promotional spam SMS arrives in the inbox. The user experiences minor annoyance or must manually delete the message.

            **Design Decision:** Our training pipeline deliberately optimizes for **high Spam Precision** (minimizing False Positives) while maintaining robust overall Macro-F1.

            ### 2. Known Boundary Limitations
            - **Adversarial / Obfuscated Text:** Modern spammers substitute numbers or symbols for letters (e.g., `FR33`, `C@LL`). Classical n-gram TF-IDF models may fail to recognize heavily obfuscated vocabulary.
            - **Zero-Shot / Novel Spam Domains:** Spams referring to new cryptocurrencies, novel apps, or trending schemes outside the 2011 UCI vocabulary will have low TF-IDF weights.
            - **Short / Ambiguous Texts:** Single-word messages (e.g. `Free`, `Win`, `Call`) without context may trigger false positives or high uncertainty.
            - **Language Specificity:** The UCI collection consists almost exclusively of English-language SMS messages. Non-English SMS will not be reliably classified.
            """
        )


if __name__ == "__main__":
    main()
