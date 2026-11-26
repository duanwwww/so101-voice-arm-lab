"""Simplified Edge console: fixture speech/planning, actual MuJoCo, no provider calls."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

from playwright.sync_api import expect, sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from browser_fixtures import BrowserFixtureAccess, FREEFORM, assert_simple_console, server_at_random_port, short_wav
from browser_smoke import run as smoke


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    events = []
    access = BrowserFixtureAccess(events)
    sound = short_wav()

    def asr(audio, *, approved_once=False):
        assert approved_once and audio == sound
        access.consume("asr", True)
        events.append({"type":"asr_fixture", "text":FREEFORM})
        return FREEFORM

    with patch.object(access,"transcribe",side_effect=asr), server_at_random_port(access) as url:
        smoke(url,out/"scenario_console.png")
        scenario_events = list(events)
        events.clear()
        with sync_playwright() as playwright:
            browser=playwright.chromium.launch(channel="msedge",headless=True)
            try:
                page=browser.new_page(viewport={"width":1280,"height":900},device_scale_factor=1)
                errors=[]
                page.on("pageerror",lambda error:errors.append(str(error)))
                page.goto(url)
                expect(page.locator("#mode")).to_contain_text("已连接")
                assert_simple_console(page)
                assert page.request.post(url+"/api/reset",data={"layout":"B"}).ok
                expect(page.locator("#vision")).to_have_text("红块 · B")
                page.locator("#audiofile").set_input_files({"name":"fixture.wav","mimeType":"audio/wav","buffer":sound})
                expect(page.locator("#entry")).to_have_value(FREEFORM)
                page.locator("#send").click()
                expect(page.locator("#message")).to_contain_text("已完成 2/2 步",timeout=45000)
                expect(page.locator("#vision")).to_have_text("红块 · A")
                expect(page.locator("#send")).to_be_enabled()
                assert not errors,errors
                assert [event["type"] for event in events] == ["asr_fixture","planner_fixture"]
                receipt=page.request.get(url+"/api/state").json()["last"]
                assert receipt["measurement"]["error_deg"]<.2
                assert page.request.post(url+"/api/feedback").status == 404
                assert page.request.post(url+"/api/speak").status == 404
                page.screenshot(path=str(out/"simple_console.png"),full_page=True)
                (out/"browser.json").write_text(json.dumps({
                    "mode":"Microsoft Edge + actual MuJoCo; in-memory ASR and planner fixtures",
                    "online_model_calls":0, "local_model_calls":0,
                    "approval_dialogs":0, "planner_selector_present":False, "tts_controls_present":False,
                    "scenario_events":scenario_events, "events":events, "receipt":receipt,
                    "browser_errors":errors},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
            finally:
                browser.close()
    print("Browser PASS: upload without consent, fixed planner, two verified motions and text receipt.")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=Path("outputs/simple-console-browser"))
    run(parser.parse_args().out)
