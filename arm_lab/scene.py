"""Course arm plus a free block, a work surface, three taught stations and one camera."""
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .kinematics import ArmModel

SOURCE = Path(__file__).resolve().parents[1] / "vendor" / "so101_course" / "SO101"
STATIONS = {"A": (0.215, 0.077), "B": (0.215, -0.077), "C": (0.245, 0.0)}
TABLE_Z = 0.035
HALF_BLOCK = 0.012
HALF_ZONE = 0.033
WIDTH, HEIGHT = 640, 400
# Midpoint between the contact faces for a 24 mm block, in the fixed jaw frame.
GRASP_CENTER = np.array([0.005, -0.0002, -0.0945])
OPEN, CLOSED = 0.57, -0.12


@dataclass(frozen=True)
class Layout:
    label: str
    xy: tuple[float,float]
    command: str
    description: str = ""
    yaw: float = 0
    gain: float = 1
    props: tuple = ()
    cover: str = ""
    reflective_band: bool = False
    wrist_blind: bool = False


TO_C = "把红色木块放到 C 区"
BLUE = (.05,.25,.85,1)
YELLOW = (.94,.75,.06,1)
RED = (.86,.035,.035,1)
LAYOUTS = {
    **{name:Layout(name,xy,TO_C if name!="C" else "把木块放到 A 区") for name,xy in STATIONS.items()},
    "empty":Layout("空桌",(.50,.28),TO_C),
    "outside":Layout("工位外",(.29,.13),TO_C),
    "clutter":Layout("异色干扰与接力搬运",STATIONS["B"],"先把木块放到 C 区，再把木块放到 A 区",
                     "红块起于 B。两枚异色视觉道具位于工位外，测试两次搬运能否各自复核。",
                     props=((.245,.115,BLUE),(.285,-.105,YELLOW))),
    "two_red":Layout("两个红色候选",STATIONS["A"],TO_C,
                     "A 的自由木块与 B 的同色视觉道具同时可见，目标不唯一时应停住并追问。",
                     props=((*STATIONS["B"],RED),)),
    "cover_source":Layout("取料位被遮挡",STATIONS["A"],"向前抓取木块",
                          "挡片遮住 A 的相机视线，验证目标不可见时的处理。",cover="A"),
    "cover_goal":Layout("放料位被遮挡",STATIONS["B"],TO_C,
                        "固定相机看不到 C。退爪后使用腕部相机独立复核。",cover="C"),
    "cover_both":Layout("两个相机均无法复核",STATIONS["B"],TO_C,
                        "固定相机对 C 的视线被挡，腕部镜头也被遮住。放置后应报告未确认。",cover="C",wrist_blind=True),
    "dim":Layout("低曝光",STATIONS["B"],"把木块从 B 搬到 A 区",
                 "相机 RGB 通道增益为正常的 45%，检测和执行均使用变暗后的图像。",gain=.45),
    "rotated":Layout("轻微偏摆",(.215,.073),"把 A 区的木块放到 B 区",
                     "木块从 A 中心横向偏 4 mm，并绕竖轴旋转 8°。",yaw=8),
    "edge_grasp":Layout("工位边缘与中途失败",(.215,.058),"先向左移动一厘米，再把木块放到 C 区，最后手腕向下转三度",
                        "木块从 A 中心横向偏 19 mm。若第二步漏抓，第三步应跳过。"),
    "reflective_band":Layout("反光条分割红面",STATIONS["C"],"把 C 区的木块放到 A 区",
                             "同一块木块上的白色条带近似局部高光，检验红色连通域是否会把一块数成两块。",
                             reflective_band=True),
}


def make_scene():
    root = ET.parse(SOURCE / "so101_new_calib.xml").getroot()
    root.find("compiler").set("meshdir", str(SOURCE / "assets"))
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast", iterations="60")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth=str(WIDTH), offheight=str(HEIGHT))
    ET.SubElement(visual, "headlight", ambient=".36 .36 .36", diffuse=".45 .45 .45", specular=".03 .03 .03")
    world = root.find("worldbody")
    ET.SubElement(world, "light", pos="0.2 -0.25 1", directional="true", dir=".1 .2 -1",
                  diffuse=".48 .48 .48", specular=".04 .04 .04")
    ET.SubElement(world, "camera", name="overhead", pos=".17 0 .76", xyaxes="0 -1 0 1 0 0", fovy="44")
    # A rigid mount on the moving gripper. Extrinsics here are simulation calibration only.
    ET.SubElement(root.find(".//body[@name='gripper']"), "camera", name="wrist_view",
                  pos="0 .040 -.012", xyaxes=".300239 .533342 -.790825 .352646 .708271 .611550", fovy="68")
    ET.SubElement(world, "geom", name="desk", type="box", size=".45 .32 .018",
                  pos=f".18 0 {TABLE_Z-.018}", rgba=".19 .24 .26 1", friction="1 .01 .001")
    for name, (x, y) in STATIONS.items():
        ET.SubElement(world, "geom", name="tile_"+name, type="box",
                      size=f"{HALF_ZONE} {HALF_ZONE} .00025", pos=f"{x} {y} {TABLE_Z+.00025}",
                      rgba=".46 .55 .55 1" if name != "C" else ".31 .58 .45 1",
                      contype="0", conaffinity="0")
    # These fixed props perturb only the camera. Inactive ones stay below the desk.
    for name, size in (("marker_0",".012 .012 .012"), ("marker_1",".012 .012 .012"),
                       ("view_cover",".042 .043 .003")):
        prop = ET.SubElement(world,"body",name=name,mocap="true",pos="0 0 -1")
        ET.SubElement(prop,"geom",name=name+"_visual",type="box",size=size,rgba=".42 .53 .57 1",
                      contype="0",conaffinity="0")
    # Collision proxies at the two contact faces. Keep all original visual meshes.
    collision = root.find(".//default[@class='collision']/geom")
    collision.set("contype", "0")
    collision.set("conaffinity", "0")
    for body, name, pos, size in [
        ("gripper", "fixed_pad", "-.0115 -.0002 -.0945", ".0035 .007 .009"),
        ("moving_jaw_so101_v1", "moving_pad", "-.0042 -.0711 .0190", ".0035 .009 .007"),
    ]:
        ET.SubElement(root.find(f".//body[@name='{body}']"), "geom", name=name, type="box",
                      pos=pos, size=size, rgba=".22 .28 .29 1", friction="1.4 .02 .002",
                      contype="2", conaffinity="1", condim="4", solref=".009 1")
    cube = ET.SubElement(world, "body", name="block", pos=f".215 .077 {TABLE_Z+HALF_BLOCK}")
    ET.SubElement(cube, "freejoint", name="block_free")
    ET.SubElement(cube, "geom", name="red_block", type="box", size=".012 .012 .012",
                  rgba=".86 .035 .035 1", mass=".016", friction="1.2 .02 .002", condim="4")
    ET.SubElement(cube,"geom",name="block_band",type="box",size=".0038 .0122 .0122",
                  rgba=".96 .96 .96 0",mass="0",contype="0",conaffinity="0")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def taught_pose(arm: ArmModel, xy, height, opening):
    """Solve position in a vertical approach posture; station coordinates are known fixtures."""
    target = np.array([*xy, height])

    def pose(angles):
        pan, shoulder, elbow = angles
        return np.array([pan, shoulder, elbow, np.pi/2-shoulder-elbow, 0, opening])

    def center(angles):
        arm.fk(pose(angles))
        data = arm.kin_data
        gid = arm.model.body("gripper").id
        return data.xpos[gid] + data.xmat[gid].reshape(3, 3) @ GRASP_CENTER

    angles = np.array([0., -.28, .65])
    for _ in range(50):
        point = center(angles)
        err = target-point
        if np.linalg.norm(err) < 0.0001:
            q = pose(angles)
            if np.all(q >= arm.limits[:, 0]) and np.all(q <= arm.limits[:, 1]):
                return q
            raise ValueError(f"工位姿态超限 {xy} z={height}: {q.round(3)}")
        jac = np.column_stack([(center(angles + d) - point)/1e-4 for d in np.eye(3)*1e-4])
        angles += np.clip(np.linalg.solve(jac, err), -.20, .20)
    raise ValueError(f"无法示教工位 {xy} z={height}")
