"""SO-101 로봇팔 + 책상 + 블록 장면(MJCF)을 만드는 도우미.

3부부터 8부까지 같은 장면을 조금씩 바꿔 가며 사용합니다.
Menagerie의 so101.xml을 include 해야 하는데, MuJoCo는 include 경로와 mesh 경로를
"메인 XML 파일이 있는 폴더" 기준으로 찾습니다. 그래서 장면 XML을 문자열로 만든 뒤
so101.xml과 같은 폴더에 파일로 써 놓고 그 경로를 돌려줍니다.
"""

import hashlib
import os
import tempfile

from common.menagerie import so101_xml

SCENE_TEMPLATE = """
<mujoco model="physicalai_scene">
  <include file="so101.xml"/>

  <option timestep="0.002" integrator="implicitfast"/>
  <visual>
    <global offwidth="640" offheight="480"/>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
  </visual>

  <asset>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.85 0.85 0.85"
             rgb2="0.75 0.75 0.75" markrgb="0.6 0.6 0.6" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="8 8" reflectance="0.1"/>
  </asset>

  <worldbody>
    <light pos="{light_pos}" dir="0 0 -1" directional="true" diffuse="{light_diffuse}"/>
    <geom name="floor" type="plane" size="1 1 0.05" material="groundplane" rgba="{floor_rgba}"/>

    <!-- 집을 물체: 빨간 블록 (한 변 3 cm) -->
    <body name="cube" pos="{cube_x} {cube_y} 0.015">
      <freejoint name="cube_free"/>
      <geom name="cube_geom" type="box" size="0.015 0.015 0.015" mass="0.02"
            rgba="0.9 0.1 0.1 1" friction="1.5 0.005 0.0001" condim="4"/>
    </body>
{extra_cubes}

    <!-- 놓을 곳: 상자 (바닥 + 낮은 벽 4개) -->
    <body name="box" pos="{box_x} {box_y} 0">
      <geom type="box" size="0.05 0.05 0.003" pos="0 0 0.003" rgba="0.2 0.4 0.9 1"/>
      <geom type="box" size="0.05 0.003 0.015" pos="0 0.05 0.015" rgba="0.2 0.4 0.9 1"/>
      <geom type="box" size="0.05 0.003 0.015" pos="0 -0.05 0.015" rgba="0.2 0.4 0.9 1"/>
      <geom type="box" size="0.003 0.05 0.015" pos="0.05 0 0.015" rgba="0.2 0.4 0.9 1"/>
      <geom type="box" size="0.003 0.05 0.015" pos="-0.05 0 0.015" rgba="0.2 0.4 0.9 1"/>
    </body>

    <!-- 카메라 4대: 위에서(top), 고정 관찰용(fixed), 블록을 따라가는 옆·정면 카메라 -->
    <camera name="top" pos="{top_cam_pos}" quat="1 0 0 0"/>
    <camera name="top_zoom" pos="0.15 0.05 0.8" quat="1 0 0 0" fovy="34"/>
    <body name="cam_target" pos="0.15 0.05 0.03"/>
    <camera name="fixed" mode="targetbody" target="cam_target" pos="0.65 -0.5 0.4"/>
    <camera name="side" mode="targetbody" target="cube" pos="0.55 -0.55 0.35"/>
    <camera name="front" mode="targetbody" target="cube" pos="0.75 0 0.3"/>
  </worldbody>

  <keyframe>
    <!-- 대기 자세: 팔을 위로 세워 손을 41 cm 높이에. 바닥의 물체를 건드리지 않고 위쪽 카메라 시야도 가리지 않는다 -->
    <key name="home" qpos="0 -1.5 0.5 0 0 0 {cube_x} {cube_y} 0.015 1 0 0 0{extra_qpos}" ctrl="0 -1.5 0.5 0 0 0"/>
  </keyframe>
</mujoco>
"""


CUBE_COLORS = {"red": "0.9 0.1 0.1 1", "green": "0.1 0.7 0.2 1", "yellow": "0.95 0.85 0.1 1"}

EXTRA_CUBE_TEMPLATE = """
    <body name="cube_{name}" pos="{x} {y} 0.015">
      <freejoint name="cube_{name}_free"/>
      <geom name="cube_{name}_geom" type="box" size="0.015 0.015 0.015" mass="0.02"
            rgba="{rgba}" friction="1.5 0.005 0.0001" condim="4"/>
    </body>"""


def build_scene(cube_pos=(0.25, 0.0), box_pos=(0.0, 0.25), extra_cubes=None,
                light_pos=(0.0, 0.0, 1.5), light_diffuse=(0.6, 0.6, 0.6), floor_rgba=(1, 1, 1, 1),
                top_cam_pos=(0.2, 0.0, 0.8)) -> str:
    """장면 XML 파일을 so101.xml 옆에 쓰고 경로를 돌려줍니다.

    extra_cubes: [("green", (x, y)), ("yellow", (x, y))]처럼 추가 블록 (8부).
    light_pos / light_diffuse / floor_rgba / top_cam_pos: 8-3의 환경 변화 실험용.

    파일 이름에 내용의 해시를 붙여, 같은 장면은 같은 파일을 재사용하고
    여러 프로세스가 동시에 써도(5부의 병렬 환경) 서로 덮어쓰지 않게 합니다.
    쓰기는 임시 파일에 한 뒤 os.replace로 바꿔치기(원자적)합니다.
    """
    robot_dir = os.path.dirname(so101_xml("so101.xml"))
    extra_xml, extra_qpos = "", ""
    for name, (x, y) in (extra_cubes or []):
        extra_xml += EXTRA_CUBE_TEMPLATE.format(name=name, x=x, y=y, rgba=CUBE_COLORS[name])
        extra_qpos += f" {x} {y} 0.015 1 0 0 0"
    fmt = lambda v: " ".join(f"{float(a):g}" for a in v)  # noqa: E731
    xml = SCENE_TEMPLATE.format(cube_x=cube_pos[0], cube_y=cube_pos[1],
                                box_x=box_pos[0], box_y=box_pos[1],
                                extra_cubes=extra_xml, extra_qpos=extra_qpos,
                                light_pos=fmt(light_pos), light_diffuse=fmt(light_diffuse),
                                floor_rgba=fmt(floor_rgba), top_cam_pos=fmt(top_cam_pos))
    tag = hashlib.md5(xml.encode("utf-8")).hexdigest()[:8]
    path = os.path.join(robot_dir, f"physicalai_scene_{tag}.xml")
    if not os.path.exists(path):
        fd, tmp = tempfile.mkstemp(prefix="scene_", suffix=".xml", dir=robot_dir)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(xml)
        os.replace(tmp, path)
    return path
