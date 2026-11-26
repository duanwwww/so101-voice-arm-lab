"""Real Edge tab closure and heartbeat loss on an isolated loopback Uvicorn/MuJoCo server."""
import argparse
import asyncio
from contextlib import contextmanager
import json
from pathlib import Path
import socket
from threading import Thread
import time
from uuid import uuid4

import httpx
from playwright.async_api import async_playwright, expect
import uvicorn

from arm_lab.model_access import ModelAccess
from arm_lab.web import create_app


WATCHDOG_S = 1.2
PLAYBACK = 0.7


@contextmanager
def isolated_server():
    access = ModelAccess()
    app = create_app(access, "B", playback_speed=PLAYBACK, watchdog_seconds=WATCHDOG_S, allow_local_rules=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        url = f"http://127.0.0.1:{sock.getsockname()[1]}"
        server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
        thread = Thread(target=lambda: asyncio.run(server.serve(sockets=[sock])), daemon=True)
        thread.start()
        try:
            for _ in range(150):
                if server.started:
                    break
                time.sleep(.1)
            if not server.started:
                raise RuntimeError("Isolated test server did not start")
            yield url, access
        finally:
            server.should_exit = True
            thread.join(20)


async def state(client):
    response = await client.get("/api/state")
    response.raise_for_status()
    return response.json()


async def until(client, predicate, timeout=25):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        snapshot = await state(client)
        if predicate(snapshot):
            return snapshot
        await asyncio.sleep(.025)
    raise TimeoutError(f"No matching state; last phase={snapshot['phase']}, busy={snapshot['busy']}, "
                       f"fault={snapshot['fault']}")


async def command_body(client, text):
    s = await state(client)
    return {"id":str(uuid4()), "watchdog_token":str(uuid4()), "text":text,
            "revision":s["revision"], "issued_at_ms":s["server_ms"], "use_deepseek":False}


def heartbeat(body):
    return {"id":body["id"], "watchdog_token":body["watchdog_token"]}


def outcome(receipt):
    return {k:receipt.get(k) for k in ("status", "message", "phases", "steps")}


async def healthy_command(client, text, check_duplicates=False):
    body = await command_body(client, text)
    t0 = time.monotonic()
    post = asyncio.create_task(client.post("/api/commands", json=body))
    await until(client, lambda s:s["busy"] and s["watchdog_active"])
    duplicate = (asyncio.create_task(client.post("/api/commands", json=body))
                 if check_duplicates else None)
    valid = 0
    while not post.done():
        response = await client.post("/api/heartbeat", json=heartbeat(body))
        if response.status_code == 200:
            valid += 1
        elif response.status_code != 409:
            response.raise_for_status()
        await asyncio.sleep(.12)
    response = await post
    response.raise_for_status()
    receipt = response.json()
    if duplicate is not None:
        same = await duplicate
        same.raise_for_status()
        assert same.json() == receipt
        replay = await client.post("/api/commands", json=body)
        replay.raise_for_status()
        assert replay.json() == receipt
    after = await state(client)
    assert valid > 0 and receipt["status"] == "success" and not after["stopped"], outcome(receipt)
    assert after["revision"] == body["revision"]+1
    return body, {"valid_heartbeat_responses":valid, "wall_seconds":round(time.monotonic()-t0, 3),
                  "holding_after":after["scene"]["holding"], "receipt":outcome(receipt),
                  "pending_and_completed_replies_identical":check_duplicates,
                  "revision_increment":after["revision"]-body["revision"]}


async def browser_drop(client, url, prior, *, camera_only=False):
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(channel="msedge", headless=True)
        try:
            page = await browser.new_page()
            posted = []
            beat_times = []
            states_received = []
            jpeg_failures = []
            drop_frames = False

            def log_request(req):
                if req.url.endswith("/api/commands") and req.method == "POST":
                    posted.append(req.post_data_json)

            def log_response(res):
                if res.url.endswith("/api/heartbeat") and res.status == 200:
                    beat_times.append(time.monotonic())
                if res.url.endswith("/api/state") and res.status == 200:
                    states_received.append(time.monotonic())

            async def camera_request(route):
                if drop_frames:
                    jpeg_failures.append(time.monotonic())
                    await route.abort("failed")
                else:
                    await route.continue_()

            page.on("request", log_request)
            page.on("response", log_response)
            if camera_only:
                await page.route(lambda u:u.split("?",1)[0] == url+"/camera.jpg", camera_request)
            await page.goto(url)
            await expect(page.locator("#mode")).to_contain_text("已连接")
            await page.locator("#planner").select_option("local")
            await page.locator("#entry").fill("放到 A 区")
            before = await state(client)
            assert before["scene"]["holding"]
            await page.locator("#send").click()
            active = await until(client, lambda s:bool(posted and beat_times) and s["busy"]
                and s["watchdog_active"] and s["scene"]["holding"] and s["phase"] == "moving"
                and s["sim_seconds"] > before["sim_seconds"]+.04)
            body = posted[0]
            wrong_owner = await client.post("/api/heartbeat", json=heartbeat(prior))
            assert wrong_owner.status_code == 409
            closed = time.monotonic()
            if camera_only:
                drop_frames = True
            else:
                await page.close()
            # The independent observer polls state but never renews the owner's lease.
            stopped = await until(client, lambda s:s["stopped"] and not s["busy"]
                and s["last"] and s["last"]["id"] == body["id"])
            observed_at = time.monotonic()
            extra = {}
            if camera_only:
                assert jpeg_failures
                state_http_ok = sum(jpeg_failures[0] < t <= observed_at for t in states_received)
                blind_renewals = sum(jpeg_failures[0] < t <= observed_at for t in beat_times)
                assert state_http_ok >= 1 and blind_renewals == 0
                await expect(page.locator("#stopStatus")).to_contain_text("锁止")
                last_frame = await page.locator("#view").get_attribute("src")
                drop_frames = False
                await page.wait_for_function("""former => {
                    const img = document.getElementById('view');
                    return img.getAttribute('src') !== former && img.complete && img.naturalWidth > 0;
                }""", arg=last_frame)
                restored = await state(client)
                assert restored["stopped"] and len(posted) == 1
                assert restored["revision"] == stopped["revision"]
                extra = {
                    "selected_camera":"/camera.jpg", "aborted_jpeg_requests":len(jpeg_failures),
                    "page_state_http_200_during_camera_outage":state_http_ok,
                    "seconds_first_aborted_jpeg_to_receipt":round(observed_at-jpeg_failures[0],3),
                    "successful_heartbeats_after_first_aborted_jpeg":blind_renewals,
                    "after_image_restored":{"new_image_decoded":True, "still_stopped":restored["stopped"],
                                            "total_commands_sent_by_page":len(posted)},
                }
            else:
                extra = {"seconds_from_close_request_to_stopped_receipt":round(observed_at-closed, 3)}
        finally:
            await browser.close()
    receipt = stopped["last"]
    assert receipt["status"] == "unverified" and stopped["fault"] and stopped["watchdog_expired"], outcome(receipt)
    assert "release" not in receipt["phases"] and receipt["gripper"]["mode"] == "stopped"
    replay = await client.post("/api/commands", json=body)
    assert replay.status_code == 200 and replay.json() == receipt
    late = await client.post("/api/heartbeat", json=heartbeat(body))
    denied = await client.post("/api/commands", json=await command_body(client,"向前移动一厘米"))
    assert late.status_code == 409 and denied.status_code == 409
    again = await state(client)
    assert again["stopped"] and again["revision"] == stopped["revision"]
    return {"before_link_loss_phase":active["phase"], "holding_before_link_loss":active["scene"]["holding"],
            "browser_valid_heartbeat_responses":len(beat_times),
            "old_session_heartbeat_http_while_busy":wrong_owner.status_code,
            **extra,
            "seconds_from_last_browser_heartbeat_response_to_stopped_receipt":round(observed_at-beat_times[-1], 3),
            "sim_seconds_at_disconnect_snapshot":active["sim_seconds"], "sim_seconds_at_receipt":stopped["sim_seconds"],
            "holding_at_final_camera_snapshot":stopped["scene"]["holding"],
            "jaw_mode_at_receipt":receipt["gripper"]["mode"], "release_executed":False,
            "late_heartbeat_http":late.status_code, "new_motion_http":denied.status_code,
            "duplicate_returned_identical_receipt":True, "fault":stopped["fault"], "receipt":outcome(receipt)}


async def no_owner_heartbeats(client, old):
    before = await state(client)
    body = await command_body(client, "向前抓取木块")
    t0 = time.monotonic()
    job = asyncio.create_task(client.post("/api/commands", json=body))
    await until(client, lambda s:s["busy"] and s["watchdog_active"])
    wrong = [{**heartbeat(body), "watchdog_token":str(uuid4())}, {**heartbeat(body), "id":str(uuid4())},
             heartbeat(old)]
    responses = []
    while not job.done():
        r = await client.post("/api/heartbeat", json=wrong[len(responses)%3])
        responses.append(r.status_code)
        await asyncio.sleep(.07)
    response = await job
    response.raise_for_status()
    receipt = response.json()
    after = await state(client)
    assert len(responses) >= len(wrong) and set(responses) == {409}
    assert receipt["status"] == "unverified" and after["stopped"] and after["watchdog_expired"], outcome(receipt)
    return {"valid_heartbeats":0, "unrelated_heartbeat_http":responses, "fault":after["fault"],
            "wall_seconds_submit_to_receipt_observation":round(time.monotonic()-t0,3),
            "sim_seconds_before":before["sim_seconds"], "sim_seconds_after":after["sim_seconds"],
            "receipt":outcome(receipt)}


async def audit(url):
    async with httpx.AsyncClient(base_url=url, timeout=50, trust_env=False) as client:
        old, healthy = await healthy_command(client, "向前抓取木块", check_duplicates=True)
        assert healthy["holding_after"]
        lost = await browser_drop(client, url, old)
        (await client.post("/api/reset", json={"layout":"B"})).raise_for_status()
        no_beats = await no_owner_heartbeats(client, old)
        denied = await client.post("/api/commands", json=await command_body(client,"向前移动一厘米"))
        assert denied.status_code == 409
        (await client.post("/api/reset", json={"layout":"B"})).raise_for_status()
        prior_camera, _ = await healthy_command(client, "向前抓取木块")
        blind = await browser_drop(client, url, prior_camera, camera_only=True)
        (await client.post("/api/reset", json={"layout":"B"})).raise_for_status()
        _, resumed = await healthy_command(client, "手腕向下转三度")
        return {"healthy":healthy, "edge_tab_closed_during_carry":lost,
                "no_valid_heartbeat":no_beats, "new_command_before_reset_http":denied.status_code,
                "edge_video_only_lost_during_carry":blind,
                "after_explicit_reset":resumed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/link_loss.json"))
    args = parser.parse_args()
    with isolated_server() as (url, access):
        result = asyncio.run(audit(url))
        assert not any(access.used.values())
    report = {"transport":"real loopback TCP + real headless Microsoft Edge; local rule executor",
              "model_calls":0, "hardware_connected":False, "watchdog_seconds":WATCHDOG_S,
              "playback_speed":PLAYBACK,
              "clock":"Real seconds use time.monotonic in the observer. Stop latency is observed receipt "
                      "latency including HTTP polling, not physical motor braking. sim_seconds is MuJoCo "
                      "simulation time at published camera snapshots. No WAN or hardware latency was measured.",
              **result}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2), encoding="utf-8")
    print("Healthy, Edge close and video-only loss during carry, invalid heartbeats and reset recovery passed:", args.out)


if __name__ == "__main__":
    main()
