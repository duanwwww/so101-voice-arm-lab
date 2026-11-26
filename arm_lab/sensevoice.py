"""Pinned, local SenseVoice Small INT8 speech recognition for short commands."""
from io import BytesIO
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile

import httpx
import numpy as np


MODEL_ID = "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
MODEL_REVISION = "2365baeacb507f821a0c8120fcee3d484dba7a07"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "outputs" / "models" / "sensevoice-small-int8-2024-07-17"
FILES = {
    "model.int8.onnx": (239233841, "c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51"),
    "tokens.txt": (315894, "f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc"),
    "LICENSE": (71, "221c6df10b0931a5629adad671ea48fb7747e034c414b6d2bfa275bc3dd4ea17"),
}
MAX_AUDIO_SECONDS = 30
_ORT_DLL_HANDLES = []


def load_windows_runtime():
    """Bind sherpa to the locked environment's ORT, before another DLL wins."""
    if os.name != "nt" or _ORT_DLL_HANDLES:
        return
    import ctypes

    spec = importlib.util.find_spec("onnxruntime")
    if spec is None or spec.origin is None:
        raise ImportError("请安装 voice 依赖中的 onnxruntime。")
    folder = Path(spec.origin).parent / "capi"
    # Windows can otherwise resolve an unrelated application's older ONNX DLL
    # from PATH, terminating the native extension before Python can catch it.
    search_directory = os.add_dll_directory(str(folder))
    try:
        library = ctypes.WinDLL(str(folder / "onnxruntime.dll"))
    except OSError:
        search_directory.close()
        raise
    _ORT_DLL_HANDLES.extend((search_directory, library))


def model_ready(path=DEFAULT_MODEL_DIR):
    """Cheap availability check for the state endpoint; loading verifies hashes."""
    path = Path(path)
    return all((path / name).is_file() and (path / name).stat().st_size == size
               for name, (size, _) in FILES.items())


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def prepare_model(path=DEFAULT_MODEL_DIR, *, allow_download=False):
    """Verify the fixed export and atomically download missing or corrupt files."""
    path = Path(path)
    for name, (size, expected) in FILES.items():
        target = path / name
        if target.is_file() and target.stat().st_size == size and sha256(target) == expected:
            continue
        if not allow_download:
            raise ValueError(f"SenseVoice 权重缺少或校验失败：{name}，请先批准下载。")
        path.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{name}"
        with tempfile.TemporaryDirectory(prefix=".sensevoice-", dir=path) as work:
            temporary = Path(work) / name
            downloaded = 0
            with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(90, connect=20)) as client:
                with client.stream("GET", url) as response, temporary.open("wb") as handle:
                    response.raise_for_status()
                    for chunk in response.iter_bytes():
                        downloaded += len(chunk)
                        if downloaded > size:
                            raise ValueError(f"SenseVoice 下载大小异常：{name}")
                        handle.write(chunk)
            if downloaded != size or sha256(temporary) != expected:
                raise ValueError(f"SenseVoice 下载校验失败：{name}")
            temporary.replace(target)
    return {"model": MODEL_ID, "revision": MODEL_REVISION,
            "files": {name: {"bytes": size, "sha256": digest}
                      for name, (size, digest) in FILES.items()}}


def decode_audio(audio):
    """Decode browser WebM/Opus or WAV bytes to 16 kHz mono float32 samples."""
    import av

    chunks = []
    count = 0
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
    with av.open(BytesIO(audio)) as clip:
        if not clip.streams.audio:
            raise ValueError("录音没有音轨。")
        for frame in clip.decode(audio=0):
            for converted in resampler.resample(frame):
                values = converted.to_ndarray().reshape(-1)
                count += values.size
                if count > 16000 * MAX_AUDIO_SECONDS:
                    raise ValueError(f"单条录音请控制在 {MAX_AUDIO_SECONDS} 秒以内。")
                chunks.append(values)
        for converted in resampler.resample(None):
            chunks.append(converted.to_ndarray().reshape(-1))
    samples = np.concatenate(chunks).astype(np.float32) if chunks else np.empty(0, np.float32)
    if not samples.size or samples.size > 16000 * MAX_AUDIO_SECONDS or not np.isfinite(samples).all():
        raise ValueError("录音时长或音频数据无效。")
    return samples


class SenseVoiceRecognizer:
    def __init__(self, path=DEFAULT_MODEL_DIR, *, num_threads=4, use_itn=True):
        load_windows_runtime()
        import sherpa_onnx

        self.path = Path(path)
        self.manifest = prepare_model(self.path)
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(self.path / "model.int8.onnx"), tokens=str(self.path / "tokens.txt"),
            num_threads=num_threads, provider="cpu", language="zh", use_itn=use_itn,
            debug=False)

    def transcribe(self, audio):
        samples = decode_audio(audio)
        stream = self.recognizer.create_stream()
        stream.accept_waveform(16000, samples)
        self.recognizer.decode_stream(stream)
        text = stream.result.text.strip()
        if not text:
            raise ValueError("这段录音没有识别到文字，请重新录音。")
        return text


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="下载或校验固定版本的本地 SenseVoice 权重")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--approve-download", action="store_true", help="批准从固定发布源下载模型文件")
    args = parser.parse_args()
    print(json.dumps(prepare_model(args.model_dir, allow_download=args.approve_download), ensure_ascii=False, indent=2))
