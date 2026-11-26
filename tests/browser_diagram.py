"""Exercise the diagram in a real Edge page against an isolated simulator and in-memory model fixtures."""
import argparse
import asyncio
import io
import json
from pathlib import Path
import socket
import sys
from threading import Thread
import time
from unittest.mock import patch
import wave

import httpx
from playwright.sync_api import expect, sync_playwright
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arm_lab.model_access import ModelAccess
from arm_lab.web import create_app
from browser_smoke import run as offline_smoke


def short_wav():
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", ""))
        wav.writeframes(b"\0\0"*1600)
    return output.getvalue()


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    phrase = "让眼前的小家伙在左侧格子安家，随后把腕部轻轻抬高两度"
    feedback = "木块已在左侧工位，手腕也已抬高。"
    events = []
    access = ModelAccess()
    client = httpx.Client
    sound = short_wav()

    def asr(audio, *, approved_once=False):
        assert approved_once and audio == sound
        access.consume("asr", True)
        events.append("Whisper fixture")
        return phrase

    def campus(request):
        assert request.url.host == "models.sjtu.edu.cn"
        payload = json.loads(request.content)
        task = json.loads(payload["messages"][1]["content"])
        if "status" in task:
            assert task["slots"] == ["A"] and task["status"] == "success"
            answer = {"speech":feedback}
            events.append("DeepSeek feedback fixture")
        else:
            assert task["command"] == phrase and task["objects"][0]["slot"] == "B"
            assert len(task["objects"][0]["xy_mm"]) == 2
            answer = {"steps":[{"action":"transfer","object":"red_1","from":"B","to":"A"},
                               {"action":"wrist","deg":-2}]}
            events.append("DeepSeek plan fixture")
        return httpx.Response(200,json={"choices":[{"finish_reason":"stop",
                              "message":{"content":json.dumps(answer,ensure_ascii=False)}}]})

    async def tts(message, *, approved_once=False):
        assert approved_once and feedback in message
        access.consume("tts", True)
        events.append("Edge-TTS fixture")
        return sound

    with patch.dict("os.environ",{"SJTU_DEEPSEEK_API_KEY":"fixture-only"}), \
            patch.object(access,"transcribe",side_effect=asr), patch.object(access,"speak",side_effect=tts), \
            patch("arm_lab.model_access.httpx.Client",side_effect=lambda **opts:client(transport=httpx.MockTransport(campus),**opts)), \
            socket.socket() as sock:
        sock.bind(("127.0.0.1",0))
        sock.listen()
        url = f"http://127.0.0.1:{sock.getsockname()[1]}"
        server = uvicorn.Server(uvicorn.Config(create_app(access,"B",playback_speed=0),log_level="error"))
        thread = Thread(target=lambda:asyncio.run(server.serve(sockets=[sock])),daemon=True)
        thread.start()
        try:
            for _ in range(100):
                if server.started:
                    break
                time.sleep(.1)
            assert server.started
            offline_smoke(url,out/"offline_console.png")
            assert not events
            with sync_playwright() as playwright:
                browser=playwright.chromium.launch(channel="msedge",headless=True)
                try:
                    page=browser.new_page(viewport={"width":1280,"height":900},device_scale_factor=1)
                    errors=[]
                    page.on("pageerror",lambda error:errors.append(str(error)))
                    page.goto(url)
                    expect(page.locator("#mode")).to_contain_text("已连接")
                    assert page.request.post(url+"/api/reset",data={"layout":"B"}).ok
                    expect(page.locator("#vision")).to_have_text("红块 · B")

                    def approve(title):
                        expect(page.locator("#approval")).to_be_visible(timeout=15000)
                        expect(page.locator("#approveTitle")).to_contain_text(title)
                        page.locator("#approveYes").click()

                    page.locator("#audiofile").set_input_files({"name":"fixture.wav","mimeType":"audio/wav","buffer":sound})
                    approve("转写")
                    expect(page.locator("#entry")).to_have_value(phrase)
                    page.locator("#autoVoice").check()
                    page.locator("#send").click()
                    approve("规划")
                    approve("解读")
                    approve("播报")
                    expect(page.locator("#feedbackText")).to_have_text(feedback)
                    expect(page.locator("#vision")).to_have_text("红块 · A")
                    expect(page.locator("#message")).to_contain_text("已完成 2/2 步")
                    expect(page.locator("#send")).to_be_enabled()
                    assert not errors,errors
                    assert events == ["Whisper fixture","DeepSeek plan fixture","DeepSeek feedback fixture","Edge-TTS fixture"]
                    receipt=page.request.get(url+"/api/state").json()["last"]
                    assert receipt["measurement"]["error_deg"]<.2
                    page.screenshot(path=str(out/"diagram_console.png"),full_page=True)
                    (out/"browser.json").write_text(json.dumps({"mode":"Edge + real MuJoCo, in-memory model fixtures",
                        "online_model_calls":0,"events":events,"receipt":receipt},ensure_ascii=False,indent=2),encoding="utf-8")
                finally:
                    browser.close()
        finally:
            server.should_exit=True
            thread.join(10)
    print("Browser PASS: recorded input fixture, free-form planning, two verified motions, result feedback and playback.")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=Path("outputs/diagram-browser"))
    run(parser.parse_args().out)
