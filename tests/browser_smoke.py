"""Drive a real Edge page. This test cancels online consent and executes only offline motions."""
import argparse
import re
from pathlib import Path
from playwright.sync_api import expect, sync_playwright


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("--url",default="http://127.0.0.1:8768")
    cli.add_argument("--screenshot",type=Path,default=Path("outputs/control_panel.png"))
    args=cli.parse_args()
    run(args.url,args.screenshot)


def run(url,screenshot):
    with sync_playwright() as p:
        browser=p.chromium.launch(channel="msedge",headless=True)
        page=browser.new_page(viewport={"width":1220,"height":900},device_scale_factor=1)
        errors=[]
        page.on("pageerror",lambda err:errors.append(str(err)))
        try:
            page.goto(url,wait_until="domcontentloaded")
            page.get_by_text("MuJoCo 仿真 · 已连接",exact=True).wait_for()
            state=page.request.get(url+"/api/state").json()
            assert state["local_approval"] and state["quota"]["deepseek"]==0
            if state["campus_key"]:
                page.locator("#entry").fill("让面前的小家伙去右边歇一会")
                page.locator("#send").click()
                expect(page.locator("#approval")).to_be_visible()
                expect(page.locator("#approveInfo")).to_contain_text("让面前的小家伙去右边歇一会")
                page.locator("#approveNo").click()
                expect(page.locator("#approval")).not_to_be_visible()
                assert page.request.get(url+"/api/state").json()["revision"]==state["revision"]
            page.locator("#planner").select_option("local")
            page.locator("#layout").select_option("cover_goal")
            expect(page.locator("#fillTask")).to_be_disabled()
            page.locator("#reset").click()
            expect(page.locator("#layoutNote")).to_contain_text("当前场景：放料位被遮挡")
            page.locator("#fillTask").click()
            expect(page.locator("#entry")).to_have_value("把红色木块放到 C 区")
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
            page.locator("#send").click()
            expect(page.locator("#message")).to_contain_text("释放后未能确认",timeout=45000)
            page.locator("#layout").select_option("B")
            page.locator("#reset").click()
            expect(page.locator("#vision")).to_have_text("红块 · B")

            def command(text, reply):
                page.locator("#entry").fill(text)
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
            print("Browser PASS: consent cancelled, blind drop reported unverified, numeric sequence, pick/hold/place, clarification and reply.")
        finally:
            browser.close()


if __name__=="__main__":
    main()
