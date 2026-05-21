"""
stt.py - Local speech-to-text using OpenAI Whisper.

Auto-detects NVIDIA GPU (CUDA) via PyTorch.
Uses GPU acceleration if available, otherwise falls back to CPU.
Language is auto-detected by default.
Requires ffmpeg on system PATH for audio decoding.
"""

import logging
import os
import tempfile

import whisper

logger = logging.getLogger(__name__)

# ---- Configuration ----
MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")
LANGUAGE = os.getenv("WHISPER_LANGUAGE", None)  # None = auto-detect

# ---- GPU auto-detection via PyTorch ----
_device: str | None = None


def _detect_device() -> str:
    """Detect best compute device via PyTorch. Returns "cuda" or "cpu"."""
    global _device
    if _device is not None:
        return _device
    try:
        import torch
        if torch.cuda.is_available():
            _device = "cuda"
            logger.info("CUDA GPU detected: %s", torch.cuda.get_device_name(0))
        else:
            _device = "cpu"
            logger.info("No CUDA GPU found, using CPU")
    except Exception:
        _device = "cpu"
    return _device


# ---- Lazy-loaded model singleton ----
_model = None


def get_model():
    global _model
    if _model is not None:
        return _model
    device = _detect_device()
    logger.info("Loading Whisper model '%s' on %s...", MODEL_SIZE, device)
    _model = whisper.load_model(MODEL_SIZE, device=device)
    logger.info("Model loaded successfully")
    return _model


def transcribe(audio_bytes: bytes) -> str:
    """Transcribe audio bytes to text using local Whisper."""
    model = get_model()

    logger.info("Transcribing %d bytes, language=%s", len(audio_bytes), LANGUAGE or "auto")

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        result = model.transcribe(
            tmp_path,
            language=LANGUAGE,
            verbose=False,
            fp16=(_detect_device() == "cuda"),
        )
    finally:
        os.unlink(tmp_path)

    detected_lang = result.get("language", "unknown")
    logger.info("Detected language: %s", detected_lang)

    text = result.get("text", "").strip()
    logger.info("Transcription complete: %d chars", len(text))
    return text
