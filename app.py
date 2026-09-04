"""Streamlit frontend for the Speech Emotion Recognition model.

Run locally:
    streamlit run app.py

Deploy: push this repo to GitHub, then connect it at https://share.streamlit.io (Streamlit
Community Cloud) pointing at `app.py` as the entry point -- it reads `requirements.txt` and
`models/cnn_lstm_hybrid.keras` needs to be present in the repo (see the "Deployment" note in
README.md for why the checkpoint is gitignored elsewhere in this project but needed here).

Audio capture uses `st.audio_input`, which records in the VIEWER'S browser -- this works when
deployed or run locally on a machine with a microphone, even though the environment this app was
*built* in has none.

Classification is explicit (a "Classify" button), not automatic as soon as audio_input/file_uploader
has a value. Streamlit reruns this whole script on *any* widget interaction anywhere on the page,
and both of those widgets keep returning their last value on every rerun until replaced -- so
auto-classifying on "value is not None" would silently re-run inference on reruns that have nothing
to do with the audio (e.g. switching tabs). The button also lets someone preview a recording before
paying for a model call. New audio (a fresh recording, or a different uploaded file) always
invalidates any prior result for that tab: results are only ever displayed for the exact bytes they
were computed from, checked via a stored fingerprint each rerun, so providing new data can never
leave a stale result on screen -- it disappears the instant the new clip is loaded, before the
button is even clicked.
"""
import hashlib
import tempfile
from pathlib import Path

import streamlit as st

from src.inference import load_model, predict_emotion

EMOJI = {
    "neutral": "😐", "calm": "😌", "happy": "😄", "sad": "😢",
    "angry": "😠", "fearful": "😨", "disgust": "🤢", "surprised": "😲",
}

st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️", layout="centered")


@st.cache_resource
def get_model():
    return load_model("models/cnn_lstm_hybrid.keras")


def run_prediction(audio_bytes: bytes) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        result = predict_emotion(tmp_path, get_model(), confidence_threshold=0.5)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return result


def show_result(result: dict):
    if result["label"] == "uncertain":
        st.warning(
            f"**Uncertain** — top guess was *{result['predicted_emotion']}* at only "
            f"{result['confidence']:.0%} confidence (below the 50% threshold)."
        )
    else:
        emoji = EMOJI.get(result["label"], "")
        st.success(f"### {emoji} {result['label'].capitalize()}  —  {result['confidence']:.0%} confidence")

    ranked = sorted(result["probabilities"].items(), key=lambda kv: -kv[1])
    st.bar_chart({label: prob for label, prob in ranked}, horizontal=True)


def classify_section(audio_bytes: bytes, state_key: str, button_label: str):
    """Renders a Classify button + result for one tab.

    A hash of `audio_bytes` is compared against the fingerprint the currently-stored result was
    computed from. They diverge the moment a new clip is recorded/uploaded (before Classify is
    clicked again), so the stale result is dropped immediately rather than lingering next to audio
    it no longer describes.
    """
    fingerprint = hashlib.md5(audio_bytes).hexdigest()
    result_key, fp_key = f"{state_key}_result", f"{state_key}_fingerprint"

    if st.session_state.get(fp_key) != fingerprint:
        st.session_state.pop(result_key, None)

    if st.button(button_label, key=f"{state_key}_button", type="primary", use_container_width=True):
        with st.spinner("Classifying..."):
            st.session_state[result_key] = run_prediction(audio_bytes)
            st.session_state[fp_key] = fingerprint

    if result_key in st.session_state:
        show_result(st.session_state[result_key])


st.title("🎙️ Speech Emotion Recognition")
st.caption(
    "CNN-LSTM hybrid trained on RAVDESS + TESS (89.9% test accuracy, macro-F1 0.895). "
    "Record or upload a short speech clip, then click Classify."
)

tab_record, tab_upload = st.tabs(["🎤 Record", "📁 Upload"])

with tab_record:
    audio = st.audio_input("Record a few seconds of speech")
    if audio is not None:
        classify_section(audio.getvalue(), state_key="record", button_label="🔍 Classify recording")
    else:
        st.caption("Record a clip above, then a Classify button will appear here.")

with tab_upload:
    uploaded = st.file_uploader("Upload a .wav file", type=["wav"])
    if uploaded is not None:
        st.audio(uploaded)
        classify_section(uploaded.getvalue(), state_key="upload", button_label="🔍 Classify upload")
    else:
        st.caption("Upload a .wav file above, then a Classify button will appear here.")

st.divider()
st.caption(
    "Model discriminates emotion across speakers seen during training — see the README for "
    "the project's full evaluation methodology and results."
)
