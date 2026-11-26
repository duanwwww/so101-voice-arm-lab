"""Explicit in-memory model fixtures for browser tests; no provider HTTP calls."""
import asyncio
from contextlib import contextmanager
import io
import socket
from threading import Thread
import time
from unittest.mock import patch
import wave

import uvicorn

from arm_lab.model_access import ModelAccess
from arm_lab.planner import actions, local_intent, offline_reply
from arm_lab.web import create_app


FREEFORM = "让眼前的小家伙在左侧格子安家，随后把腕部轻轻抬高两度"
ONGOING = "手腕向下转十度，再手腕向上转十度，再手腕向下转十度，再手腕向上转十度"
FIXTURE_COMMANDS = {
    FREEFORM, ONGOING, "把红色木块放到 C 区", "先向左挪12毫米，再把手腕向下转3度",
    "向前抓取木块", "放到左边的工位", "把木块搬过去", "B区",
    "向前移动四毫米", "向左移动十二毫米", "向左移动12毫米。",
}


def short_wav():
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", ""))
        wav.writeframes(b"\0\0" * 1600)
    return output.getvalue()


def fixture_json(task):
    """Create a known answer for the fixture vocabulary, never a live model result."""
    text = task["command"]
    if text not in FIXTURE_COMMANDS:
        raise AssertionError(f"Unexpected fixture planner input: {text}")
    if text == FREEFORM:
        return {"steps": [{"action": "transfer", "object": "red_1", "from": "B", "to": "A"},
                          {"action": "wrist", "deg": -2}]}
    intent = offline_reply(task["dialogue"], text) if task.get("dialogue") else local_intent(text)
    if intent.kind == "ask":
        return {"action": "ask", "question": intent.message}
    slot = task["objects"][0]["slot"] if len(task["objects"]) == 1 else None
    steps = []
    for step in actions(intent):
        item = {"action": step.kind}
        if step.kind in ("pick", "transfer"):
            item.update(object="red_1", **{"from": step.source or slot})
        if step.kind in ("place", "transfer"):
            item["to"] = step.target
        if step.kind == "move":
            item.update(axis=step.axis, mm=step.amount)
        if step.kind == "wrist":
            item["deg"] = step.amount
        steps.append(item)
    return {"steps": steps}


class BrowserFixtureAccess(ModelAccess):
    def __init__(self, events):
        super().__init__(planner_model="deepseek-reasoner")
        self.events = events

    def chat_json(self, instructions, task, key, approved_once):
        assert approved_once, "Web requests must not require per-call approval"
        answer = fixture_json(task)
        self.consume("deepseek", approved_once=True)
        self.events.append({"type": "planner_fixture", "input": task, "answer": answer})
        return answer

    def voice_ready(self):
        return True

    def transcribe(self, audio, *, approved_once=False):
        raise AssertionError("The test must install its own ASR fixture")

    async def speak(self, *args, **kwargs):
        raise AssertionError("TTS is not part of the simplified console")


@contextmanager
def server_at_random_port(access, layout="B", playback=0):
    with patch.dict("os.environ", {"SJTU_DEEPSEEK_API_KEY": "fixture-only"}), socket.socket() as sock:
        # Configure fixture readiness before create_app snapshots the key state.
        # Use real web defaults: use_deepseek=False cannot select local rules.
        app = create_app(access, layout, playback_speed=playback)

        @app.get("/__browser_fixture__")
        def fixture_marker():
            return {"models": "in-memory fixtures", "real_provider_calls": 0}

        sock.bind(("127.0.0.1", 0))
        sock.listen()
        server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
        thread = Thread(target=lambda: asyncio.run(server.serve(sockets=[sock])), daemon=True)
        thread.start()
        try:
            for _ in range(150):
                if server.started:
                    break
                time.sleep(.1)
            assert server.started, "test server failed to start"
            yield f"http://127.0.0.1:{sock.getsockname()[1]}"
        finally:
            server.should_exit = True
            thread.join(12)


def assert_simple_console(page):
    for selector in ("#planner", "#approval", "#autoVoice", "#feedback", "#speak"):
        assert page.locator(selector).count() == 0, selector
    # Declared before interacting so any accidental browser confirmation fails the run.
    page.on("dialog", lambda dialog: (_ for _ in ()).throw(AssertionError(dialog.message)))


def open_scene_settings(page):
    # Scene controls may be collapsed, but initialization remains explicit.
    if not page.locator("#layout").is_visible():
        page.locator("details").filter(has=page.locator("#layout")).locator("summary").click()
