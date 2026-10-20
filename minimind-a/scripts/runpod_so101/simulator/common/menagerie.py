"""MuJoCo Menagerie(로봇 모델 모음) 경로 도우미.

Menagerie는 Google DeepMind가 관리하는 고품질 MuJoCo 로봇 모델 저장소입니다.
전체를 받으면 2 GB가 넘기 때문에, 이 책에서 쓰는 로봇 폴더만 sparse checkout으로 받습니다
(처음 한 번, 수십 MB). 저장 위치는 저장소 루트의 third_party/mujoco_menagerie입니다.
이미 전체 사본이 있으면 환경변수 MENAGERIE_DIR로 그 경로를 지정하세요.
"""

import os
import subprocess

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(REPO_ROOT, "third_party", "mujoco_menagerie")
# 이 책에서 사용하는 로봇 폴더
ROBOT_DIRS = ["robotstudio_so101"]


def menagerie_dir() -> str:
    path = os.environ.get("MENAGERIE_DIR", DEFAULT_DIR)
    if not os.path.isdir(path):
        print(f"[menagerie] {path}가 없어 필요한 로봇 모델만 내려받습니다 ...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", MENAGERIE_URL, path],
            check=True,
        )
        subprocess.run(["git", "-C", path, "sparse-checkout", "set", *ROBOT_DIRS], check=True)
    return path


def so101_xml(scene: str = "scene.xml") -> str:
    """SO-101 로봇팔 MJCF 경로. scene="scene.xml" 은 바닥이 포함된 장면, "so101.xml" 은 로봇만."""
    return os.path.join(menagerie_dir(), "robotstudio_so101", scene)
