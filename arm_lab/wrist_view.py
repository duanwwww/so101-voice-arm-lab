"""Oblique confirmation from a calibrated camera mounted on the retreating gripper."""
import cv2
import numpy as np

from .scene import HALF_BLOCK, HALF_ZONE, HEIGHT, LAYOUTS, STATIONS, TABLE_Z, WIDTH
from .vision import detect


# Hypotheses around the pixel-ray estimate. A square repeats after 90 degrees.
GRID_MM = np.arange(-24, 25, 3)
OFFSETS = np.array([(x, y) for x in GRID_MM for y in GRID_MM]) / 1000
YAW = np.deg2rad(np.arange(0, 90, 15))
UNIT_CORNERS = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
CORNERS = np.stack([UNIT_CORNERS @ np.array([[np.cos(a), np.sin(a), 0],
                                             [-np.sin(a), np.cos(a), 0], [0, 0, 1]])
                    for a in YAW]) * HALF_BLOCK


def box_station(box, eye, rotation, focal):
    """Fit a resting 24-mm cube silhouette. All matching grid poses must fit the same station."""
    x, y, w, h = box
    if x <= 0 or y <= 0 or x+w >= WIDTH or y+h >= HEIGHT:
        return None
    ray = rotation @ np.array([(x+(w-1)/2-WIDTH/2)/focal, (HEIGHT/2-y-(h-1)/2)/focal, -1])
    if ray[2] >= -.25 or eye[2] <= TABLE_Z + 2*HALF_BLOCK:
        return None
    guess = eye + ray*((TABLE_Z+HALF_BLOCK-eye[2])/ray[2])
    centers = np.repeat(guess[None, :], len(OFFSETS), axis=0)
    centers[:, :2] += OFFSETS
    vertices = centers[:, None, None, :] + CORNERS[None, :, :, :]
    camera = (vertices - eye) @ rotation
    if np.any(camera[..., 2] >= -1e-5):
        return None
    u = WIDTH/2 - focal*camera[..., 0]/camera[..., 2]
    v = HEIGHT/2 + focal*camera[..., 1]/camera[..., 2]
    projected = np.stack((u.min(-1), v.min(-1), u.max(-1), v.max(-1)), axis=-1)
    plausible = np.max(abs(projected - (x, y, x+w-1, y+h-1)), axis=-1) <= 4
    if not plausible.any() or np.any(abs(OFFSETS[plausible.any(axis=1)]) >= .024):
        return None
    points = vertices[plausible][..., :2]
    return next((name for name, xy in STATIONS.items()
                 if np.all(np.abs(points-xy) <= HALF_ZONE-.003)), None)


def observe_wrist(sim):
    sim.renderer.update_scene(sim.data, camera="wrist_view")
    rgb = sim.renderer.render()
    if LAYOUTS[sim.layout].wrist_blind:
        rgb = np.zeros_like(rgb)
    cid = sim.model.camera("wrist_view").id
    eye = sim.data.cam_xpos[cid]
    orientation = sim.data.cam_xmat[cid].reshape(3, 3)
    focal = HEIGHT / (2*np.tan(np.deg2rad(sim.model.cam_fovy[cid])/2))
    objects = detect(rgb, {})["objects"]
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for obj in objects:
        obj["slot"] = box_station(obj["box"], eye, orientation, focal)
        x, y, w, h = obj["box"]
        cv2.rectangle(frame, (x, y), (x+w-1, y+h-1), (58,252,250), 2)
        if obj["slot"]:
            cv2.putText(frame, obj["slot"], (x, max(16, y-3)), cv2.FONT_HERSHEY_SIMPLEX, .55, (58,252,250), 1)
    return objects, frame


def confirm_drop(sim, target, scene):
    if scene["holding"]:
        return False, "unconfirmed"
    visible = scene["objects"]
    if visible:
        if len(visible) == 1 and visible[0]["slot"] == target:
            return True, "overhead"
        return False, "unconfirmed"
    if not getattr(sim, "wrist_enabled", True):
        return False, "unconfirmed"
    objects, _ = observe_wrist(sim)
    if len(objects) == 1 and objects[0]["slot"] == target:
        return True, "wrist"
    return False, "unconfirmed"
