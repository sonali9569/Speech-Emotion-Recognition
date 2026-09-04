"""Model architectures: SVM baseline (see src/baseline.py), LSTM-only, CNN-only, and the CNN-LSTM
hybrid headline model. Built with the Keras FUNCTIONAL API throughout
(`keras.Model(inputs=..., outputs=...)`), not `Sequential` -- the hybrid in particular needs to
reshape/permute a CNN's 4D output into an LSTM's 3D sequence input mid-model, which `Sequential`
can't express.

Masking (two different mechanisms, chosen per-model -- not the same trick reused blindly):
  - LSTMClassifier: the input IS the raw (T, 120) feature sequence, nothing computed in front of it
    -- so the padded frames can be explicitly zeroed (see src/data/keras_prep.py) and Keras's own
    `layers.Masking(mask_value=0.)` auto-detects and skips them. This is the literal, simplest case.
  - CNNLSTMHybrid: a CNN sits between the raw input and the LSTM. Conv2D/BatchNorm layers have bias
    and learned shifts, so a zeroed input column does NOT survive as an exact-zero column after the
    CNN -- `layers.Masking`'s auto-detection would silently fail here. Instead, the real (unpadded)
    length is passed as a second model input and used to build an explicit boolean mask
    (`tf.sequence_mask`), passed directly to the LSTM layer's `mask=` argument -- a first-class,
    documented Keras mechanism for exactly this situation.
"""
from __future__ import annotations

import tensorflow as tf
import keras
from keras import layers


@keras.saving.register_keras_serializable(package="ser")
class SequenceMaskLayer(layers.Layer):
    """Builds a boolean (batch, maxlen) mask from a per-sample length input, for passing directly
    to an LSTM's `mask=` argument. A named, registered layer (not a raw `layers.Lambda` closure) --
    Keras 3 refuses to deserialize an arbitrary Python lambda baked into a saved model by default
    (a real arbitrary-code-execution guard), so a proper layer class is what actually loads back
    cleanly with plain `keras.models.load_model()`, no `safe_mode=False` escape hatch needed."""

    def __init__(self, maxlen: int, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen

    def call(self, lengths):
        return tf.sequence_mask(lengths, maxlen=self.maxlen)

    def get_config(self):
        config = super().get_config()
        config.update({"maxlen": self.maxlen})
        return config


@keras.saving.register_keras_serializable(package="ser")
class MaskedAttentionPooling(layers.Layer):
    """Learned attention over an LSTM's per-timestep outputs, restricted to real (unmasked)
    timesteps -- see the module docstring for why masking matters here."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True  # declares that `call`'s `mask=` kwarg is handled explicitly
        self.score_dense = layers.Dense(1)

    def call(self, inputs, mask=None):
        # inputs: (batch, T, hidden)
        scores = tf.squeeze(self.score_dense(inputs), axis=-1)  # (batch, T)
        if mask is not None:
            mask = tf.cast(mask, scores.dtype)
            scores = scores + (1.0 - mask) * -1e9  # push masked positions' scores to ~-inf
        weights = tf.nn.softmax(scores, axis=1)
        weights = tf.expand_dims(weights, axis=-1)  # (batch, T, 1)
        return tf.reduce_sum(inputs * weights, axis=1)  # (batch, hidden)


def build_lstm_classifier(
    input_size: int = 120,
    seq_len: int = 79,
    hidden_size: int = 128,
    num_classes: int = 8,
    dropout: float = 0.3,
    use_attention: bool = False,
) -> keras.Model:
    """2-layer LSTM over an MFCC(+delta+delta2) sequence; classifies from either the last real
    hidden state (use_attention=False) or an attention-weighted sum over real frames."""
    inputs = keras.Input(shape=(seq_len, input_size), name="mfcc_sequence")
    x = layers.Masking(mask_value=0.0)(inputs)  # auto-detects zeroed (padded) frames -- see module docstring
    x = layers.LSTM(hidden_size, return_sequences=True, dropout=0.0)(x)
    x = layers.Dropout(dropout)(x)  # between-layer dropout, applied between the two stacked LSTM layers
    x = layers.LSTM(hidden_size, return_sequences=use_attention)(x)  # (batch,T,hidden) if attention else (batch,hidden)

    if use_attention:
        pooled = MaskedAttentionPooling()(x)
    else:
        pooled = x

    x = layers.Dropout(dropout)(pooled)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="lstm_classifier")


def build_cnn_classifier(n_mels: int = 128, seq_len: int = 79, num_classes: int = 8, dropout: float = 0.4) -> keras.Model:
    """3-block Conv2D+BatchNorm+ReLU+MaxPool over a mel-spectrogram image, global-average-pooled
    to a feature vector. Input is channels-LAST (Keras default): (n_mels, T, 1)."""
    inputs = keras.Input(shape=(n_mels, seq_len, 1), name="mel_spectrogram")
    x = layers.Conv2D(32, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)  # collapses each feature map to a single value per channel

    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="cnn_classifier")


def build_cnn_lstm_hybrid(
    n_mels: int = 128,
    seq_len: int = 79,
    lstm_hidden: int = 128,
    num_classes: int = 8,
    dropout: float = 0.3,
    use_attention: bool = False,
) -> keras.Model:
    """The headline model. Pools frequency only (kernel (2,1)) so the time axis (seq_len) survives
    every conv block unchanged, then reshapes into an LSTM sequence -- the CNN extracts local
    spectro-temporal texture at each instant, the LSTM models how that texture evolves across the
    utterance. Takes a second input, `valid_length`, to build the explicit LSTM mask (see module
    docstring)."""
    mel_input = keras.Input(shape=(n_mels, seq_len, 1), name="mel_spectrogram")
    length_input = keras.Input(shape=(), dtype=tf.int32, name="valid_length")

    x = layers.Conv2D(16, 3, padding="same")(mel_input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 1))(x)  # freq only: 128 -> 64, time stays seq_len

    x = layers.Conv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 1))(x)  # freq: 64 -> 32

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 1))(x)  # freq: 32 -> 16
    # x: (batch, freq=16, time=seq_len, channels=64) -- Keras is channels-last, so time is axis 2 here.

    x = layers.Permute((2, 1, 3))(x)                       # -> (batch, time, freq, channels)
    x = layers.Reshape((seq_len, 16 * 64))(x)               # -> (batch, time, freq*channels) = LSTM input

    mask = SequenceMaskLayer(maxlen=seq_len, name="build_mask")(length_input)

    lstm_out = layers.LSTM(lstm_hidden, return_sequences=use_attention)(x, mask=mask)

    if use_attention:
        pooled = MaskedAttentionPooling()(lstm_out, mask=mask)
    else:
        pooled = lstm_out

    x = layers.Dropout(dropout)(pooled)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs=[mel_input, length_input], outputs=outputs, name="cnn_lstm_hybrid")
