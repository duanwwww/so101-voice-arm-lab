"""Exercise mid-task voice turns in Edge against a private loopback MuJoCo server."""
import argparse
import asyncio
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import socket
import sys
from threading import Thread
import time
from unittest.mock import patch

from playwright.sync_api import expect, sync_playwright
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arm_lab.model_access import ModelAccess
from arm_lab.web import create_app
from browser_diagram import short_wav


PROHIBIT = "不要向前移动一厘米"
QUERY = "当前状态"
DRAFT = "向前移动四毫米"
PAUSE = "暂停当前任务"
RESUME = "恢复控制"
ONGOING = "手腕向下转十度，再手腕向上转十度，再手腕向下转十度，再手腕向上转十度"
PLAYBACK = 0.08  # Deliberately slow enough to converse during real trajectory steps.
REAL_MOVE = "向左移动十二毫米"
REAL_AUDIO = {
    PROHIBIT: ("negative.wav", "7c3148c14821d482208eb87d5062e918801925d761d430c1ce1996b11a685944"),
    REAL_MOVE: ("move.wav", "f7783322eaeb544f1e39a5a567b7e7662e8a54175efd55f399482a876601f911"),
}


@contextmanager
def server_at_random_port(access, playback):
    app = create_app(access, "B", playback_speed=playback)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
        thread = Thread(target=lambda: asyncio.run(server.serve(sockets=[sock])), daemon=True)
        thread.start()
        try:
            for _ in range(150):
                if server.started:
                    break
                time.sleep(0.1)
            assert server.started, "test server failed to start"
            yield f"http://127.0.0.1:{sock.getsockname()[1]}"
        finally:
            server.should_exit = True
            thread.join(12)


def exercise(page, url, samples, draft_reference, events):
    errors, posted, beats, resets = [], [], [], []
    page.on("pageerror", lambda error: errors.append(str(error)))

    def log_request(request):
        if request.method == "POST" and request.url.endswith("/api/commands"):
            posted.append(request.post_data_json)
        if request.method == "POST" and request.url.endswith("/api/reset"):
            resets.append(request.url)

    def log_response(response):
        if response.url.endswith("/api/heartbeat") and response.status == 200:
            beats.append((response.request.post_data_json["id"], time.monotonic()))

    page.on("request", log_request)
    page.on("response", log_response)
    page.goto(url)
    entry, send = page.locator("#entry"), page.locator("#send")
    expect(page.locator("#mode")).to_contain_text("已连接")
    expect(page.locator("#vision")).to_contain_text("红块 · B")
    page.locator("#planner").select_option("local")

    def state():
        return page.request.get(url + "/api/state").json()

    def submit(text):
        with page.expect_response(lambda r: r.url.endswith("/api/commands") and
                                  r.request.post_data_json["text"] == text, timeout=45000) as result:
            send.click()
        assert result.value.ok, result.value.text()
        receipt = result.value.json()
        assert receipt["text"] == text, receipt
        return receipt

    def upload(reference):
        expect(page.locator("#audiofile")).to_be_enabled()
        page.locator("#audiofile").set_input_files(
            {"name": "input.wav", "mimeType": "audio/wav", "buffer": samples[reference]})
        expect(page.locator("#approveTitle")).to_contain_text("转写")
        with page.expect_response(lambda r: r.url.endswith("/api/transcribe"), timeout=60000) as upload_result:
            page.locator("#approveYes").click()
        assert upload_result.value.ok, upload_result.value.text()
        text = upload_result.value.json()["text"]
        expect(entry).to_have_value(text)
        expect(send).to_be_enabled()
        return text

    initial = state()
    entry.fill(ONGOING)
    expect(page.locator("#cameraStatus")).not_to_contain_text("等待")
    send.click()
    expect(page.locator("#mode")).to_contain_text("执行中")
    original_id = next(p["id"] for p in posted if p["text"] == ONGOING)
    active = state()
    assert active["watchdog_active"]
    expect(page.locator("#mic")).to_be_enabled()

    interactions = []
    for reference, status in ((PROHIBIT, "ignored"), (QUERY, "observed")):
        phrase = upload(reference)
        receipt = submit(phrase)
        snapshot = state()
        assert receipt["status"] == status and not receipt["plan"] and not receipt["steps"]
        assert snapshot["busy"] and snapshot["watchdog_active"] and not snapshot["paused"]
        assert snapshot["revision"] == active["revision"]
        interactions.append({"phrase": phrase, "status": status, "message": receipt["message"],
                             "busy": snapshot["busy"], "sim_seconds": snapshot["sim_seconds"]})

    draft = upload(draft_reference)
    send.click()
    expect(page.locator("#notice")).to_contain_text("保留在输入框")
    expect(entry).to_have_value(draft)
    draft_text, review = page.locator("#draftText"), page.locator("#reviewDraft")
    expect(draft_text).to_contain_text(draft)
    expect(draft_text).to_be_visible()
    expect(review).to_be_disabled()
    assert state()["busy"] and not any(p["text"] == draft for p in posted)

    upload(PAUSE)
    expect(draft_text).to_contain_text(draft)
    pause = submit(PAUSE)
    assert pause["status"] == "paused"
    expect(page.locator("#resume")).to_be_enabled(timeout=10000)
    stopped = state()
    assert stopped["paused"] and not stopped["busy"] and not stopped["stopped"]
    assert stopped["last"]["text"] == ONGOING and stopped["last"]["status"] == "paused"
    assert len(stopped["last"]["steps"]) < 4
    assert stopped["sim_seconds"] > initial["sim_seconds"]
    expect(draft_text).to_be_visible()
    expect(review).to_be_disabled()
    upload(RESUME)
    expect(draft_text).to_contain_text(draft)
    resume = submit(RESUME)
    assert resume["status"] == "resumed" and not resume["plan"]
    resumed = state()
    assert not resumed["paused"] and not resumed["watchdog_expired"]
    for key in ("sim_seconds", "tcp_mm", "layout", "scene"):
        assert resumed[key] == stopped[key], key

    # Review only refills the editor. Until the separate submit there must be no
    # new request, simulation step, change of pose, or command revision.
    expect(entry).to_have_value("")
    expect(draft_text).to_be_visible()
    expect(draft_text).to_contain_text(draft)
    expect(review).to_be_enabled()
    post_count_before_review = len(posted)
    review.click()
    expect(entry).to_have_value(draft)
    expect(page.locator("#notice")).to_contain_text("核对当前画面和草稿后发送")
    reviewed = state()
    assert len(posted) == post_count_before_review
    assert not reviewed["busy"] and reviewed["last"]["status"] == "resumed"
    for key in ("sim_seconds", "tcp_mm", "revision"):
        assert reviewed[key] == resumed[key], key
    final = submit(draft)
    assert final["status"] == "success" and final["measurement"]["error_mm"] <= 5, final
    axis, mm = (1, 12) if draft_reference == REAL_MOVE else (0, 4)
    assert final["measurement"]["requested_mm"] == mm and abs(final["measurement"]["displacement_mm"][axis] - mm) < 1
    expect(draft_text).to_be_hidden()
    assert [p["text"] for p in posted] == [ONGOING, interactions[0]["phrase"], QUERY, PAUSE, RESUME, draft]
    assert not errors and not resets, (errors, resets)
    initial_beats = [t for owner, t in beats if owner == original_id]
    during_upload = [sum(start <= t <= end for t in initial_beats)
                     for _, start, end in events[:4]]
    assert sum(during_upload) > 0 and not state()["watchdog_expired"]
    return {"browser": "Microsoft Edge, headless", "backend": "MuJoCo on private loopback port",
            "hardware_connected": False,
            "original_task": ONGOING, "during_motion": interactions,
            "draft": {"text": draft, "retained": True, "submitted_while_busy": False,
                      "retained_in_draft_bar_through_pause_and_resume": True,
                      "review_refilled_editor": True, "new_command_posts_on_review": 0,
                      "sim_seconds_before_review": resumed["sim_seconds"],
                      "sim_seconds_after_review": reviewed["sim_seconds"],
                      "explicit_submit_count": sum(p["text"] == draft for p in posted),
                      "cleared_after_success": True},
            "original_owner_successful_heartbeats": len(initial_beats),
            "heartbeats_within_first_four_asr_calls": during_upload,
            "original_receipt": {key: stopped["last"][key] for key in ("status", "steps", "plan")},
            "pause_snapshot": {key: stopped[key] for key in ("layout", "sim_seconds", "tcp_mm", "scene")},
            "resume_snapshot": {key: resumed[key] for key in ("layout", "sim_seconds", "tcp_mm", "scene")},
            "watchdog_expired": False, "reset_requests": len(resets), "browser_errors": errors,
            "new_task": {key: final[key] for key in ("text", "status", "measurement")}}


def run(out, use_whisper=False):
    sound, access, events = short_wav(), ModelAccess(), []
    access.asr_engine = "whisper"
    draft = REAL_MOVE if use_whisper else DRAFT
    turns = (PROHIBIT, QUERY, draft, PAUSE, RESUME)
    phrases = iter(turns)
    samples = dict.fromkeys(turns, sound)
    playback = 0.05 if use_whisper else PLAYBACK
    report = {"status": "started", "input_mode": "browser WAV uploads with per-turn consent",
              "playback_speed": playback, "real_model_calls": 0, "online_model_calls": 0,
              "local_whisper_attempts": [], "transcript_fixture_turns": []}
    out.parent.mkdir(parents=True, exist_ok=True)
    if use_whisper:
        if not access.voice_ready():
            raise RuntimeError("The downloaded local Whisper Small weights are required")
        for label, (name, expected) in REAL_AUDIO.items():
            audio_path = Path(__file__).resolve().parents[1] / "outputs" / "prompt-trial-audio" / name
            samples[label] = audio_path.read_bytes()
            if hashlib.sha256(samples[label]).hexdigest() != expected:
                raise RuntimeError(f"Unexpected audio hash for {name}")
        # A pre-existing journal may hold partial attempts. Refuse to spend again.
        with out.open("x", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    local_whisper = access.transcribe

    def save():
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def transcript(audio, *, approved_once):
        phrase = next(phrases)
        assert approved_once and audio == samples[phrase]
        start = time.monotonic()
        if use_whisper and phrase in REAL_AUDIO:
            assert report["real_model_calls"] < 2
            name, digest = REAL_AUDIO[phrase]
            attempt = {"file": name, "sha256": digest, "reference": phrase, "status": "started"}
            report["local_whisper_attempts"].append(attempt)
            report["real_model_calls"] += 1
            save()
            result = local_whisper(audio, approved_once=True)
            attempt.update(status="complete", output=result, seconds=round(time.monotonic()-start, 3))
            save()
        else:
            access.consume("asr", approved_once=True)
            time.sleep(0.80)  # Let the initial owner renew the heartbeat during fixture ASR.
            result = phrase
            report["transcript_fixture_turns"].append(phrase)
        events.append((phrase, start, time.monotonic()))
        return result

    with patch.dict("os.environ", {"HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}), \
            patch.object(access, "transcribe", side_effect=transcript), \
            patch.object(access, "chat_json", side_effect=AssertionError("unexpected planner call")), \
            server_at_random_port(access, playback) as url, sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        try:
            report.update(exercise(browser.new_page(), url, samples, draft, events))
        except Exception as error:
            report.update(status="failed", error=str(error), asr_turns_consumed=access.used["asr"])
            save()
            raise
        finally:
            browser.close()
    assert access.used == {"asr": 5, "deepseek": 0, "tts": 0}
    assert report["real_model_calls"] == (2 if use_whisper else 0)
    report["status"] = "passed"
    save()
    print("PASS: live voice UI, original heartbeat, pause/resume, explicit next motion", out)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path)
    cli.add_argument("--approved-two-local-whisper", action="store_true",
                     help="Use exactly the two pinned local WAVs for up to two local Whisper transcriptions")
    args = cli.parse_args()
    default = "live_voice_browser_hybrid.json" if args.approved_two_local_whisper else "live_voice_browser.json"
    run(args.out or Path("outputs")/default, args.approved_two_local_whisper)
