"""Only RGB pixels and the fixed station calibration enter the detector."""
import cv2
import numpy as np

from .scene import WIDTH, HEIGHT, HALF_ZONE, STATIONS, TABLE_Z, HALF_BLOCK


def station_rois(model, data):
    cid = model.camera("overhead").id
    eye = data.cam_xpos[cid]
    rot = data.cam_xmat[cid].reshape(3, 3)
    focal = HEIGHT / (2*np.tan(np.deg2rad(model.cam_fovy[cid])/2))

    def pixel(point):
        x, y, z = rot.T @ (np.asarray(point)-eye)
        return (round(WIDTH/2 + focal*x/-z), round(HEIGHT/2 - focal*y/-z))

    rois = {}
    for name, (x, y) in STATIONS.items():
        pixels = np.array([pixel((x+i*HALF_ZONE, y+j*HALF_ZONE, TABLE_Z+2*HALF_BLOCK))
                           for i in (-1, 1) for j in (-1, 1)])
        rois[name] = (*pixels.min(axis=0).tolist(), *pixels.max(axis=0).tolist())
    return rois


def detect(rgb, rois):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (0, 95, 50), (8, 255, 255)) | cv2.inRange(hsv, (171, 95, 50), (179, 255, 255))
    _, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    objects = []
    for (x, y, width, height, area), center in zip(stats[1:], centers[1:]):
        if area < 35:
            continue
        slot = next((name for name, (l,t,r,b) in rois.items()
                     if l <= x and x+width-1 <= r and t <= y and y+height-1 <= b), None)
        objects.append({"slot": slot, "pixel": np.rint(center).astype(int).tolist(),
                        "box": [int(x), int(y), int(width), int(height)]})
    return {"objects": objects}


def tabletop_xy_mm(model, data, pixel):
    """Intersect the observed pixel ray with the calibrated top plane of a resting block."""
    camera = model.camera("overhead").id
    eye = data.cam_xpos[camera]
    rotation = data.cam_xmat[camera].reshape(3,3)
    focal = HEIGHT / (2*np.tan(np.deg2rad(model.cam_fovy[camera])/2))
    u, v = pixel
    ray = rotation @ np.array([(u-WIDTH/2)/focal, (HEIGHT/2-v)/focal, -1.0])
    distance = (TABLE_Z+2*HALF_BLOCK-eye[2])/ray[2]
    return ((eye+distance*ray)[:2]*1000).round(1).tolist()


def annotate(rgb, rois, scene):
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for name, (l,t,r,b) in rois.items():
        cv2.rectangle(frame, (l,t), (r,b), (207,224,215), 1)
        cv2.putText(frame, name, (l,t-5), cv2.FONT_HERSHEY_SIMPLEX, .55, (246,246,246), 1, cv2.LINE_AA)
    for obj in scene["objects"]:
        x,y,w,h = obj["box"]
        cv2.rectangle(frame, (x,y), (x+w-1,y+h-1), (58,252,250), 2)
    cv2.putText(frame, "SO-101  |  SIM    +X forward = up", (12,24),
                cv2.FONT_HERSHEY_SIMPLEX, .54, (245,245,245), 1, cv2.LINE_AA)
    return frame
