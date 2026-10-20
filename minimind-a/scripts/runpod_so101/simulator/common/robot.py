"""SO-101 시뮬레이션 로봇을 다루는 얇은 래퍼.

3부(제어)부터 8부(최종 프로젝트)까지 같은 클래스를 씁니다.
핵심은 세 가지뿐입니다.
  - 관절 목표를 주고 시간을 흘려보내기 (move_arm, set_gripper)
  - 손끝·블록의 위치 읽기 (site_pose, cube_pos)
  - 카메라로 찍어 영상 저장 (render 옵션)
"""

import os

import mujoco
import numpy as np

from common.ik import solve_ik
from common.scene import build_scene

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_JOINT = "gripper"
SITE = "gripperframe"           # 손끝 기준점 (두 손가락 사이). 이 site의 x축이 손가락이 향하는 방향
DOWN = (0.0, 0.0, -1.0)         # 위에서 내려 집을 때 손가락 방향
GRIPPER_OPEN = 1.2              # rad, 집게 벌림
GRIPPER_CLOSED = -0.1           # rad, 집게 오므림 (약간 더 조이도록 음수)


class SO101Sim:
    def __init__(self, cube_pos=(0.25, 0.0), box_pos=(0.0, 0.25), render=True,
                 camera="fixed", fps=30, width=640, height=480, **scene_kw):
        """scene_kw는 build_scene으로 그대로 전달됩니다 (extra_cubes, light_pos 등, 8부)."""
        self.model = mujoco.MjModel.from_xml_path(build_scene(cube_pos, box_pos, **scene_kw))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.arm_qadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ARM_JOINTS])
        self.arm_act = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ARM_JOINTS])
        self.grip_act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_JOINT)
        self.grip_qadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, GRIPPER_JOINT)]
        self.site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, SITE)
        self.cube_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.box_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "box")

        self.fps = fps
        self.steps_per_frame = int(round(1 / (fps * m.opt.timestep)))
        self.frames = []
        self.camera = camera
        self.renderer = mujoco.Renderer(m, height=height, width=width) if render else None
        self.reset()

    # ---------- 상태 ----------
    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)   # keyframe "home"
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0

    @property
    def q_arm(self):
        return self.data.qpos[self.arm_qadr].copy()

    def site_pose(self):
        """손끝 기준점의 위치(3)와 회전행렬(3x3)."""
        return self.data.site_xpos[self.site_id].copy(), self.data.site_xmat[self.site_id].reshape(3, 3).copy()

    def cube_pos(self, name=None):
        """블록 위치. name이 None 이면 빨간 블록, 아니면 cube_<name> (예: "green")."""
        if name is None or name == "red":
            return self.data.xpos[self.cube_id].copy()
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"cube_{name}")
        return self.data.xpos[bid].copy()

    def box_pos(self):
        return self.data.xpos[self.box_id].copy()

    def gripper_opening(self):
        return float(self.data.qpos[self.grip_qadr])

    # ---------- 시간 진행 ----------
    def step(self, n=1):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            self._step_count += 1
            if self.renderer is not None and self._step_count % self.steps_per_frame == 0:
                self.renderer.update_scene(self.data, camera=self.camera)
                self.frames.append(self.renderer.render().copy())

    def hold(self, seconds):
        self.step(int(seconds / self.model.opt.timestep))

    def move_arm(self, q_target, seconds=1.0):
        """현재 ctrl에서 목표 관절각까지 선형 보간하며 이동 (급격한 움직임 방지)."""
        q_target = np.asarray(q_target, dtype=float)
        start = self.data.ctrl[self.arm_act].copy()
        n = max(1, int(seconds / self.model.opt.timestep))
        for i in range(n):
            a = (i + 1) / n
            self.data.ctrl[self.arm_act] = (1 - a) * start + a * q_target
            self.step()

    def set_gripper(self, value, seconds=0.5):
        self.data.ctrl[self.grip_act] = value
        self.hold(seconds)

    def open_gripper(self, seconds=0.5):
        self.set_gripper(GRIPPER_OPEN, seconds)

    def close_gripper(self, seconds=0.7):
        self.set_gripper(GRIPPER_CLOSED, seconds)

    # ---------- IK ----------
    def ik(self, target_pos, target_dir=DOWN, **kw):
        """손끝을 target_pos로, 손가락 방향을 target_dir로 보내는 관절각을 계산만 한다."""
        return solve_ik(self.model, self.data, SITE, target_pos, target_dir, ARM_JOINTS, axis=0, **kw)

    def move_to(self, target_pos, target_dir=DOWN, seconds=1.0, **kw):
        """IK로 관절각을 구한 뒤 그 자세로 이동. (관절각, IK 오차)를 돌려준다."""
        q, err = self.ik(target_pos, target_dir, **kw)
        self.move_arm(q, seconds)
        return q, err

    def finger_dir(self):
        """손가락이 향하는 방향(단위 벡터)."""
        return self.site_pose()[1][:, 0]

    # ---------- 영상 ----------
    def save_video(self, path):
        if not self.frames:
            return
        import imageio

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        imageio.mimsave(path, self.frames, fps=self.fps)
        imageio.imwrite(path.replace(".mp4", "_last.png"), self.frames[-1])
        print(f"영상 저장: {path} ({len(self.frames)} frames)")

    def close(self):
        """렌더러를 명시적으로 닫는다 (프로그램 종료 시 EGL 경고 방지)."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def snapshot(self, camera=None, width=None, height=None):
        """지정 카메라로 현재 장면을 한 장 찍어 RGB 배열로 돌려준다."""
        cam = camera or self.camera
        if width or height:
            r = mujoco.Renderer(self.model, height=height or 480, width=width or 640)
        else:
            r = self.renderer or mujoco.Renderer(self.model, height=480, width=640)
        r.update_scene(self.data, camera=cam)
        return r.render().copy()
