"""File-based inference demo.

Usage:
    python3 scripts/predict_emotion.py path/to/clip.wav [more_clips.wav ...]

Loads the trained CNN-LSTM hybrid checkpoint and prints the predicted emotion + full probability
distribution for each file given. Works on any .wav file — RAVDESS/TESS clips or a fresh recording
of your own (record on your phone/laptop, save as .wav, pass the path here).
"""
import sys
sys.path.insert(0, ".")

from src.inference import load_model, predict_emotion

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    model = load_model("models/cnn_lstm_hybrid.keras")

    for filepath in sys.argv[1:]:
        result = predict_emotion(filepath, model, confidence_threshold=0.5)
        print(f"\n{filepath}")
        if result["label"] == "uncertain":
            print(f"  uncertain  (top guess was {result['predicted_emotion']} at only {result['confidence']:.1%} confidence, below the 50% threshold)")
        else:
            print(f"  predicted: {result['label']}  (confidence: {result['confidence']:.1%})")
        ranked = sorted(result["probabilities"].items(), key=lambda kv: -kv[1])
        for label, prob in ranked:
            bar = "#" * int(prob * 30)
            print(f"    {label:<10} {prob:>6.1%} {bar}")
