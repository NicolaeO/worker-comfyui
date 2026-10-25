"""Text-to-speech jobs for the worker: Qwen3-TTS voice cloning, outside ComfyUI.

A job with {"input": {"tts": {...}}} is handled here instead of by ComfyUI:

    {"tts": {
        "text": "Line to speak",                       # required
        "language": "English",                          # optional
        "ref_audio": "<base64 wav>",                    # the voice to clone (or ref_audio_path on the volume)
        "ref_text": "transcript of ref_audio",          # optional; without it only the timbre is cloned
        "x_vector_only": true,                          # timbre-only clone (default when ref_text is missing)
        "seed": 1,
        "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    }}

Returns {"images": [{"filename": "voice.wav", "type": "base64", "data": ...}]}, the same shape as image
jobs, so clients can treat it like any other output. Models are read from the network volume's Hugging
Face cache (/runpod-volume/hf) when it exists, so nothing is downloaded on a warm volume.
"""
import base64
import io
import os
import random
import tempfile
import time

DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
VOLUME_HF = "/runpod-volume/hf"
_models = {}


def _setup_cache():
    if os.path.isdir(VOLUME_HF):
        os.environ.setdefault("HF_HOME", VOLUME_HF)


def _model(name):
    if name not in _models:
        _setup_cache()
        import torch
        from qwen_tts import Qwen3TTSModel
        t = time.time()
        _models[name] = Qwen3TTSModel.from_pretrained(name, device_map="cuda:0", dtype=torch.bfloat16)
        print(f"worker-comfyui - TTS model {name} loaded in {time.time() - t:.0f}s")
    return _models[name]


def _seed(seed):
    import numpy
    import torch
    random.seed(seed)
    numpy.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run(spec):
    text = (spec.get("text") or "").strip()
    if not text:
        return {"error": "tts.text is required"}
    if len(text) > 1500:
        return {"error": "tts.text is too long (max 1500 characters)"}

    ref_path, tmp = spec.get("ref_audio_path"), None
    if spec.get("ref_audio"):
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(base64.b64decode(spec["ref_audio"].split(",", 1)[-1]))
        tmp.close()
        ref_path = tmp.name
    if not ref_path or not os.path.isfile(ref_path):
        return {"error": "tts needs ref_audio (base64) or an existing ref_audio_path"}

    try:
        import soundfile as sf
        model = _model(spec.get("model") or DEFAULT_MODEL)
        ref_text = (spec.get("ref_text") or "").strip()
        x_vector_only = bool(spec.get("x_vector_only", not ref_text))
        if x_vector_only:
            prompt = model.create_voice_clone_prompt(ref_audio=ref_path, x_vector_only_mode=True)
        else:
            prompt = model.create_voice_clone_prompt(ref_audio=ref_path, ref_text=ref_text)
        _seed(int(spec.get("seed", 1)))
        t = time.time()
        wavs, sr = model.generate_voice_clone(text=text, language=spec.get("language", "English"),
                                              voice_clone_prompt=prompt)
        buf = io.BytesIO()
        sf.write(buf, wavs[0], sr, format="WAV")
        seconds = len(wavs[0]) / sr
        print(f"worker-comfyui - TTS: {seconds:.1f}s of audio in {time.time() - t:.1f}s")
        return {"images": [{"filename": "voice.wav", "type": "base64",
                            "data": base64.b64encode(buf.getvalue()).decode()}],
                "audio_seconds": round(seconds, 2)}
    except Exception as err:  # report, don't crash the worker
        import traceback
        traceback.print_exc()
        return {"error": f"TTS failed: {err.__class__.__name__}: {err}"}
    finally:
        if tmp:
            os.unlink(tmp.name)
