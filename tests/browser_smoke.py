"""Drive the simplified console against an isolated, in-memory planner fixture."""
import argparse
import json
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from browser_fixtures import BrowserFixtureAccess, assert_simple_console, open_scene_settings, server_at_random_port
from playwright.sync_api import expect, sync_playwright


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("--url",help="Optional URL of a test-only server exposing /__browser_fixture__")
    cli.add_argument("--screenshot",type=Path,default=Path("outputs/control_panel.png"))
    args=cli.parse_args()
    if args.url:
        run(args.url,args.screenshot)
    else:
        with server_at_random_port(BrowserFixtureAccess([])) as url:
            run(url,args.screenshot)


def run(url,screenshot):
    with sync_playwright() as p:
        browser=p.chromium.launch(channel="msedge",headless=True)
        page=browser.new_page(viewport={"width":1220,"height":900},device_scale_factor=1)
        errors=[]
        page.on("pageerror",lambda err:errors.append(str(err)))
        try:
            page.goto(url,wait_until="domcontentloaded")
            page.get_by_text("MuJoCo 仿真 · 已连接",exact=True).wait_for()
            marker=page.request.get(url+"/__browser_fixture__")
            assert marker.ok and marker.json().get("models")=="in-memory fixtures", "Refuse to drive a non-fixture server"
            assert_simple_console(page)
            open_scene_settings(page)
            page.locator("#layout").select_option("cover_goal")
            expect(page.locator("#fillTask")).to_be_disabled()
            page.locator("#reset").click()
            expect(page.locator("#layoutNote")).to_contain_text("当前场景：放料位被遮挡")
            page.locator("#fillTask").click()
            expect(page.locator("#entry")).to_have_value("把红色木块放到 C 区")
            expect(page.locator("#cameraStatus")).not_to_contain_text("等待")
            page.locator("#send").click()
            expect(page.locator("#metric")).to_contain_text("腕部相机",timeout=45000)
            page.locator("#cameraSelect").select_option("/camera/wrist.jpg")
            expect(page.locator("#view")).to_have_attribute("src", re.compile("/camera/wrist"))
            page.locator("#halt").click()
            expect(page.locator("#stopStatus")).to_contain_text("停止")
            expect(page.locator("#send")).to_be_disabled()
            page.locator("#layout").select_option("cover_both")
            page.locator("#reset").click()
            expect(page.locator("#layoutNote")).to_contain_text("当前场景：两个相机均无法复核")
            page.locator("#fillTask").click()
            expect(page.locator("#cameraStatus")).not_to_contain_text("等待")
            page.locator("#send").click()
            expect(page.locator("#message")).to_contain_text("释放后未能确认",timeout=45000)
            page.locator("#layout").select_option("B")
            page.locator("#reset").click()
            expect(page.locator("#vision")).to_have_text("红块 · B")

            def command(text, reply):
                page.locator("#entry").fill(text)
                expect(page.locator("#cameraStatus")).not_to_contain_text("等待")
                page.locator("#send").click()
                expect(page.locator("#message")).to_contain_text(reply,timeout=45000)

            command("先向左挪12毫米，再把手腕向下转3度","已完成 2/2 步")
            expect(page.locator("#plan")).to_contain_text("move_delta(y,12 mm)")
            expect(page.locator("#steps li")).to_have_count(2)
            command("向前抓取木块","双侧接触已确认")
            expect(page.locator("#vision")).to_have_text("夹爪 · 已握住木块")
            command("放到左边的工位","已在 A 区")
            command("把木块搬过去","要把木块放到哪个工位")
            expect(page.locator("#replyBar")).to_be_visible()
            command("B区","已在 B 区")
            expect(page.locator("#vision")).to_have_text("红块 · B")
            expect(page.locator("#replyBar")).not_to_be_visible()
            screenshot.parent.mkdir(parents=True,exist_ok=True)
            page.screenshot(path=str(screenshot),full_page=True)
            assert not errors,errors
            print("Browser PASS: no model options or approval dialogs, blind drop unverified, numeric sequence, pick/hold/place, clarification and reply.")
        except Exception:
            screenshot.parent.mkdir(parents=True,exist_ok=True)
            page.screenshot(path=str(screenshot.with_name(screenshot.stem+"-failed.png")),full_page=True)
            screenshot.with_suffix(".failure.json").write_text(json.dumps({
                "body":page.locator("body").inner_text(), "errors":errors,
                "state":page.request.get(url+"/api/state").json()},ensure_ascii=False,indent=2),encoding="utf8")
            raise
        finally:
            browser.close()


if __name__=="__main__":
    main()
