"""Record paired, live MuJoCo camera samples and a Chinese Windows narration.

Windows: uv run --frozen --extra voice --extra data --with pillow python scripts/record_safety_demo.py
"""
import argparse
from bisect import bisect_right
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import wave

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arm_lab.flow import execute
from arm_lab.simulator import Simulator
from arm_lab.wrist_view import observe_wrist

FPS = 12
AUDIO_RATE = 24000
SIZE = (1280, 720)
FONT_DIR = Path("C:/Windows/Fonts")


@dataclass(frozen=True)
class Chapter:
    title: str
    layout: str
    command: str
    duration: int
    note: str
    narration: str
    expected: str


CHAPTERS = (
    Chapter("一条搬运口令如何完成", "B", "把红色木块放到 C 区", 43,
            "用户给出目标。控制电脑负责分步执行，松爪退开后复看三次。",
            "这是固定底座机械臂的远程操控课程实验。左边是固定相机，右边是随夹爪移动的腕部相机，两幅都是本次物理仿真的同步画面。第一条任务是把红木块放到 C 区。这里选择文字输入和离线规则，以便单独检查控制器。程序先观察木块在哪，再完成靠近、下降、合爪、试抬和搬运。退开以后，它连续看三次，才给出最后的回执。", "success"),
    Chapter("木块没有摆正", "rotated", "把 A 区的木块放到 B 区", 32,
            "木块横向偏 4 mm、绕竖轴转 8°。新画面决定本次下爪点。",
            "这次木块偏了四毫米，还转了八度。下爪前，系统重新检查是否只有一个红块，再根据平面坐标做一次小范围修正。它不使用测试器保存的木块真实位置。抓取时收到两侧接触反馈，就停止追逐闭合角度，改用仿真中的限力握持。先抬六毫米检查夹持，再继续搬运。这里的力矩数值只适用于这套仿真。", "success"),
    Chapter("固定相机看不见落点", "cover_goal", "把红色木块放到 C 区", 36,
            "上方挡片只遮固定相机。退爪后一秒内采三次，用腕部视角复核。",
            "固定相机看得见起点，却被挡片挡住了 C 区。请同时看右侧镜头。夹爪退开后，系统才用手腕相机的独立标定和物体外框，判断木块能否完整落在 C 区。它要求三次都有证据。右侧画面来自仿真机身上的第二台相机；预约使用的实物有没有这台相机，还要现场确认。", "success"),
    Chapter("两个视角都失去证据", "cover_both", "把红色木块放到 C 区", 32,
            "本场景将腕部视频遮黑。释放动作完成后，控制器仍需证据。",
            "现在保持相同的搬运口令，同时遮住腕部画面。即使已经发出了释放动作，固定相机和腕部相机都无法确认落点，任务就应当报告未确认。系统不会猜测成功，也不会让模型擅自再抓一次。重新看清桌面后，再由操作者决定下一步。", "unverified"),
    Chapter("偏位越界，取消下爪", "edge_grasp", "先向左移动一厘米，再把木块放到 C 区，最后手腕向下转三度", 23,
            "木块离开中心 19 mm。超出 8 mm 准入范围就结束后续动作。",
            "这条口令有三步。先向左移动一厘米，然后搬木块，最后转腕。木块已经偏到工位边缘。第一步可以完成，第二步的最新观测超出八毫米，所以系统在下爪之前就停止，第三步也跳过。可以从最终回执看到实际完成了几步。", "unverified"),
    Chapter("持物中收到停止信号", "A", "把 A 区的木块放到 C 区", 28,
            "试抬后由测试程序注入停止事件，检查持物、锁止和后续指令。",
            "最后看一次可重复的停机测试。试抬后，测试程序发出与页面停止按钮相同的控制信号。已确认持物时，模拟控制器取消后续搬运并保留有界握持。锁止后再发一条移动命令，机械臂也不会继续。整段演示使用软件仿真。真实编码器误差、舵机保护和实体停止装置，要单独在现场验收。", "unverified"),
)

PHASE = {"before": "开始观察", "before_step_2": "取得下一步的新画面",
         "pregrasp_observation": "下爪前复看", "pregrasp": "到达预抓位",
         "descend": "到达抓取高度", "close": "双侧接触，切换有界握持",
         "test_lift": "完成 6 mm 试抬", "lift": "提升到搬运高度",
         "carry": "到达目标上方", "release": "已释放木块", "retreat": "夹爪已退开",
         "verify_1": "第 1 次复看", "verify_2": "第 2 次复看", "verify_3": "第 3 次复看",
         "after": "生成控制回执"}


def cameras(sim, overhead):
    _, wrist = observe_wrist(sim)
    scene = np.hstack((overhead, wrist))
    ok, jpeg = cv2.imencode(".jpg", scene, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise RuntimeError("Camera encoding failed")
    return jpeg.tobytes()


def capture(sim, chapter):
    sim.reset(chapter.layout)
    start = sim.data.time
    samples = []
    stage = "before"
    original_tick = sim._tick

    def take(image=None):
        if image is None:
            _, image = sim.camera()
        jaw = sim.read_gripper()
        telemetry = f"指尖接触 {jaw.fingers}/2   ·   " + ("已离桌" if not jaw.on_table else "木块接触桌面")
        align = sim.last_alignment
        if align:
            dx, dy = align["correction_mm"]
            telemetry += f"   ·   XY 修正 ({dx:+.1f}, {dy:+.1f}) mm"
        samples.append((float(sim.data.time-start), cameras(sim, image), stage, telemetry))

    def recording_tick(count):
        original_tick(count)
        if not samples or sim.data.time-start-samples[-1][0] >= .12:
            take()

    def observed(label, scene, frame):
        nonlocal stage
        stage = label
        take(frame)
        if chapter is CHAPTERS[-1] and label == "test_lift":
            sim.stop_event.set()

    sim._tick = recording_tick
    try:
        receipt = execute(sim, chapter.command, observer=observed)
    finally:
        sim._tick = original_tick
    if receipt["status"] != chapter.expected:
        raise ValueError(f"Unexpected result for {chapter.layout}: {receipt['message']}")
    proof = receipt["message"]
    views = receipt["measurement"].get("drop_views", [])
    if views:
        names = {"overhead": "固定", "wrist": "腕部", "unconfirmed": "无证据"}
        proof += "    " + " / ".join(names[v] for v in views)
    if chapter is CHAPTERS[-1]:
        now = sim.data.time
        blocked = execute(sim, "向前移动一厘米")
        if blocked["plan"] or now != sim.data.time or sim.stop_response != "bounded_hold":
            raise ValueError("Stop did not latch with bounded hold")
        proof = "持物锁止。再次发出“向前移动一厘米”，执行动作数为 0。"
    return samples, proof


def font(size, bold=False):
    return ImageFont.truetype(str(FONT_DIR/("msyhbd.ttc" if bold else "msyh.ttc")), size)


FONTS = {}


def written(draw, xy, value, size, color, bold=False):
    key = size, bold
    if key not in FONTS:
        FONTS[key] = font(size, bold)
    draw.text(xy, value, font=FONTS[key], fill=color, anchor="lt")


def image_for(chapter, sample, proof, marker):
    seconds, frame, phase, telemetry = sample
    rgb = Image.new("RGB", SIZE, "#142522")
    draw = ImageDraw.Draw(rgb)
    written(draw, (30, 14), "SO-101  /  远程机械臂控制实验", 21, "#c9ded7")
    written(draw, (814, 14), "MuJoCo 仿真  ·  文字直入  ·  离线规则", 22, "#efc87b")
    written(draw, (30, 54), chapter.title, 39, "#f8faf7", True)
    written(draw, (30, 111), "固定相机", 23, "#f2f5ef", True)
    written(draw, (655, 111), "腕部相机", 23, "#f2f5ef", True)
    written(draw, (253, 115), marker, 18, "#adc4bb")
    written(draw, (1000, 115), f"本回合仿真时间  {seconds:4.1f} s", 18, "#adc4bb")
    pair = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
    for i, x in enumerate((30, 655)):
        camera = pair[:, i*640:(i+1)*640]
        rescaled = cv2.resize(camera, (595, 372), interpolation=cv2.INTER_AREA)
        rgb.paste(Image.fromarray(cv2.cvtColor(rescaled, cv2.COLOR_BGR2RGB)), (x, 143))
    written(draw, (30, 533), "操作者", 19, "#efc87b", True)
    written(draw, (116, 528), chapter.command, 26, "#f5f8f4")
    written(draw, (30, 580), "反馈" if proof else "观测", 19, "#efc87b", True)
    written(draw, (116, 575), proof or (PHASE.get(phase, phase) + "    " + telemetry),
            22, "#d9eee3")
    draw.line((30, 625, 1250, 625), fill="#52685e")
    written(draw, (30, 653), chapter.note, 24, "#e8efe9")
    return np.asarray(rgb)


def voice_track(directory):
    """Local desktop narration. This is the explanation track, not a project ASR input."""
    manifest = directory / "narration.json"
    manifest.write_text(json.dumps([c.narration for c in CHAPTERS], ensure_ascii=False), encoding="utf-8")
    script = directory / "narrate.ps1"
    script.write_text("""
$ErrorActionPreference = 'Stop'
$all = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $PSScriptRoot 'narration.json') | ConvertFrom-Json
$voice = New-Object -ComObject SAPI.SpVoice
$voice.Voice = $voice.GetVoices('Language=804').Item(0)
$voice.Rate = 2
for ($i=0; $i -lt $all.Count; $i++) {
    $file = New-Object -ComObject SAPI.SpFileStream
    $file.Format.Type=22
    $file.Open((Join-Path $PSScriptRoot "$i.wav"),3,$false)
    try { $voice.AudioOutputStream=$file; [void]$voice.Speak($all[$i]) } finally { $file.Close() }
}
""", encoding="utf-8-sig")
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)], check=True)
    clips = []
    for i, chapter in enumerate(CHAPTERS):
        with wave.open(str(directory / f"{i}.wav")) as f:
            if f.getnchannels() != 1 or f.getsampwidth() != 2:
                raise ValueError("Unexpected narration format")
            signal = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
            length = round(len(signal)*AUDIO_RATE/f.getframerate())
            signal = np.interp(np.linspace(0, len(signal)-1, length), np.arange(len(signal)), signal)
        clip = np.zeros(chapter.duration*AUDIO_RATE, dtype=np.int16)
        offset = round(.5*AUDIO_RATE)
        if offset+len(signal) > len(clip):
            raise ValueError(f"Narration exceeds chapter {i+1}: {len(signal)/AUDIO_RATE:.1f}s")
        clip[offset:offset+len(signal)] = np.rint(signal).astype(np.int16)
        clips.append(clip)
    return np.concatenate(clips)


def record(out):
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as tmp:
        sound = voice_track(Path(tmp))
    sim = Simulator()
    try:
        with av.open(str(out), "w", options={"movflags": "+faststart"}) as media:
            video = media.add_stream("libx264", rate=FPS)
            video.width, video.height = SIZE
            video.pix_fmt = "yuv420p"
            video.options = {"crf": "24", "preset": "veryfast"}
            audio = media.add_stream("aac", rate=AUDIO_RATE)
            audio.layout, audio.bit_rate = "mono", 48000
            pos = 0
            for chapter in CHAPTERS:
                samples, proof = capture(sim, chapter)
                stamps = [row[0] for row in samples]
                total = stamps[-1]
                last_key, canvas = None, None
                for i in range(chapter.duration*FPS):
                    t = i/FPS
                    stage = "开场定格" if t < 3 else "结果定格" if t >= chapter.duration-7 else "动作回放"
                    fraction = np.clip((t-3)/(chapter.duration-10), 0, 1)
                    instant = total * fraction
                    index = max(0, bisect_right(stamps, instant)-1)
                    show_proof = stage == "结果定格"
                    rate = total/(chapter.duration-10)
                    marker = f"动作回放  ×{rate:.2f}" if stage == "动作回放" else stage
                    key = index, marker, show_proof
                    if last_key != key:
                        canvas = image_for(chapter, samples[index], proof if show_proof else "", marker)
                        last_key = key
                    media.mux(video.encode(av.VideoFrame.from_ndarray(canvas, format="rgb24")))
                    end = pos+AUDIO_RATE//FPS
                    frame = av.AudioFrame.from_ndarray(sound[None, pos:end], format="s16", layout="mono")
                    frame.sample_rate, frame.pts = AUDIO_RATE, pos
                    media.mux(audio.encode(frame))
                    pos = end
                print(f"{chapter.title}: {total:.2f}s / {chapter.duration}s / {proof}", flush=True)
            media.mux(video.encode())
            media.mux(audio.encode())
    finally:
        sim.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"results/safety/simulation_demo.mp4")
    record(parser.parse_args().out)
