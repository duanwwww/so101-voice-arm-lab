"""Record the RGB camera during actual stepping and include the command receipt."""
import cv2

from .flow import execute
from .scene import LAYOUTS
from .simulator import Simulator


def record(layout, out):
    import av  # Included in the voice extra's media stack.
    out.parent.mkdir(parents=True,exist_ok=True)
    sim = Simulator(layout)
    try:
        with av.open(str(out),"w") as video:
            stream = video.add_stream("libx264",rate=15)
            stream.width,stream.height,stream.pix_fmt = 640,400,"yuv420p"
            stream.options = {"crf":"20","preset":"veryfast"}
            start = float(sim.data.time)
            written = 0
            previous = None
            phase = "observe"

            def encode(image):
                for packet in stream.encode(av.VideoFrame.from_ndarray(image,format="bgr24")):
                    video.mux(packet)

            def frame(label, scene, image):
                nonlocal previous,written,phase
                phase = label
                target = round((sim.data.time-start)*15) + 12
                while previous is not None and written < target-1:
                    encode(previous)
                    written += 1
                image = image.copy()
                cv2.rectangle(image,(0,363),(640,400),(31,42,39),-1)
                cv2.putText(image,f"Scene: {layout}    |    {phase}",(13,386),
                            cv2.FONT_HERSHEY_SIMPLEX,.55,(244,248,244),1,cv2.LINE_AA)
                previous = image
                if written < target:
                    encode(image)
                    written += 1

            sim.preview = lambda: frame(phase,*sim.camera())
            receipt = execute(sim,LAYOUTS[layout].command,observer=frame)
            frame(receipt["status"],*sim.camera())
            for _ in range(18):
                encode(previous)
            for packet in stream.encode():
                video.mux(packet)
        return receipt
    finally:
        sim.close()
