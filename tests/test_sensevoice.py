import hashlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import wave
from unittest.mock import MagicMock, patch

import httpx
import numpy as np

from arm_lab import sensevoice
from arm_lab.model_access import ModelAccess, ModelDenied, ModelFailure


def pcm_wav(rate, channels, seconds):
    buffer = io.BytesIO()
    values = np.tile(np.array([3276, -3276], np.int16), int(rate * seconds * channels / 2))
    with wave.open(buffer, "wb") as clip:
        clip.setparams((channels, 2, rate, 0, "NONE", "not compressed"))
        clip.writeframes(values.tobytes())
    return buffer.getvalue()


class SenseVoiceTests(unittest.TestCase):
    def test_browser_pcm_and_webm_decode_to_bounded_mono_samples(self):
        import av

        pcm = sensevoice.decode_audio(pcm_wav(48000, 2, 0.1))
        self.assertEqual(pcm.dtype, np.float32)
        self.assertEqual(pcm.shape, (1600,))
        self.assertLessEqual(float(np.max(np.abs(pcm))), 1)
        compressed = io.BytesIO()
        with av.open(compressed, "w", format="webm") as output:
            stream = output.add_stream("libopus", rate=48000)
            stream.layout = "mono"
            frame = av.AudioFrame.from_ndarray(np.zeros((1, 4800), np.int16), format="s16", layout="mono")
            frame.sample_rate = 48000
            for packet in stream.encode(frame):
                output.mux(packet)
            for packet in stream.encode(None):
                output.mux(packet)
        decoded = sensevoice.decode_audio(compressed.getvalue())
        self.assertTrue(1500 <= decoded.size <= 1700)
        with self.assertRaises(ValueError):
            sensevoice.decode_audio(pcm_wav(16000, 1, 30.1))

    def test_weight_download_is_pinned_hash_checked_and_atomic(self):
        good = b"fixed-small-fixture"
        files = {"tokens.txt": (len(good), hashlib.sha256(good).hexdigest())}
        original_client = httpx.Client
        requests = []
        def response(request):
            requests.append(str(request.url))
            return httpx.Response(200, content=good)
        with TemporaryDirectory() as tmp, patch.object(sensevoice, "FILES", files):
            folder = Path(tmp)
            with patch.object(sensevoice.httpx, "Client", side_effect=AssertionError("unapproved download")):
                with self.assertRaises(ValueError):
                    sensevoice.prepare_model(folder)
            with patch.object(sensevoice.httpx, "Client", side_effect=lambda **kw: original_client(
                    transport=httpx.MockTransport(response), **kw)):
                manifest = sensevoice.prepare_model(folder, allow_download=True)
            self.assertIn(sensevoice.MODEL_REVISION, requests[0])
            self.assertEqual(manifest["files"]["tokens.txt"]["sha256"], files["tokens.txt"][1])
            self.assertTrue(sensevoice.model_ready(folder))
            with patch.object(sensevoice.httpx, "Client", side_effect=AssertionError("valid cache downloaded")):
                sensevoice.prepare_model(folder)
            (folder / "tokens.txt").write_bytes(b"x" * len(good))
            with self.assertRaises(ValueError):
                sensevoice.prepare_model(folder)
            with patch.object(sensevoice.httpx, "Client", side_effect=lambda **kw: original_client(
                    transport=httpx.MockTransport(lambda req: httpx.Response(200, content=b"bad")), **kw)):
                with self.assertRaises(ValueError):
                    sensevoice.prepare_model(folder, allow_download=True)
            self.assertEqual((folder / "tokens.txt").read_bytes(), b"x" * len(good))
            self.assertEqual(list(folder.iterdir()), [folder / "tokens.txt"])

    def test_authorization_precedes_loading_and_failures_consume_one_attempt(self):
        access = ModelAccess()
        self.assertEqual(access.asr_engine, "sensevoice")
        recognizer = MagicMock()
        recognizer.transcribe.return_value = "向左移动十二毫米"
        with patch("arm_lab.model_access.prepare_model") as prepare, \
             patch("arm_lab.model_access.SenseVoiceRecognizer", return_value=recognizer) as loader:
            with self.assertRaises(ModelDenied):
                access.transcribe(b"recording")
            prepare.assert_not_called()
            self.assertEqual(access.transcribe(b"recording", approved_once=True), "向左移动十二毫米")
            access.remaining["asr"] = 1
            recognizer.transcribe.side_effect = ValueError("invalid recording")
            with self.assertRaisesRegex(ModelFailure, "invalid recording"):
                access.transcribe(b"bad-recording")
            self.assertEqual((access.used["asr"], access.remaining["asr"]), (2, 0))
            loader.assert_called_once()
            prepare.assert_called_once_with(access.sensevoice_dir, allow_download=True)

    def test_legacy_whisper_can_still_be_explicitly_selected(self):
        access = ModelAccess()
        access.asr_engine = "whisper"
        with patch.object(access, "_transcribe_whisper", return_value="向左") as legacy, \
             patch("arm_lab.model_access.SenseVoiceRecognizer") as new:
            self.assertEqual(access.transcribe(b"audio", approved_once=True), "向左")
            legacy.assert_called_once_with(b"audio", True)
            new.assert_not_called()
            self.assertEqual(access.asr_model, "Systran/faster-whisper-small")

    def test_service_reports_selected_model_in_state_and_transcript(self):
        from fastapi.testclient import TestClient
        from arm_lab.web import create_app

        access = ModelAccess()
        with patch.object(access, "transcribe", return_value="不要向前移动"), \
             TestClient(create_app(access, "A", allow_local_rules=True), base_url="http://127.0.0.1:8768",
                        client=("127.0.0.1", 1234)) as client:
            state = client.get("/api/state").json()
            self.assertEqual(state["asr_engine"], "sensevoice")
            self.assertEqual(state["asr_model"], sensevoice.MODEL_ID)
            headers = {"Origin": "http://127.0.0.1:8768"}
            result = client.post("/api/transcribe", files={"file": ("test.wav", b"dummy", "audio/wav")},
                                 headers=headers).json()
            self.assertEqual((result["text"], result["engine"], result["model"]),
                             ("不要向前移动", "sensevoice", sensevoice.MODEL_ID))
