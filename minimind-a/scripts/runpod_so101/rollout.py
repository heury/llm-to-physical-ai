"""Ten actual closed-loop SO101 MuJoCo episodes with the learned MiniMind-A."""

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("MENAGERIE_DIR", "/tmp/minimind-a-rollout-assets/menagerie")

from collections import deque
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import av
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import AutoTokenizer

TRAIN = Path(os.environ.get("SO101_RUN_ROOT", "/shared-volume/minimind-a-so101-20261020"))
ROOT = TRAIN / "rollout10"
SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(TRAIN / "source"))
sys.path.insert(0, str(SOURCE / "simulator"))
from model.model_vla import MiniMindVLA
from common.robot import SO101Sim

FPS, EPISODES, STEPS, EXECUTE, MAX_SECONDS = 20, 10, 10, 4, 12
TASK = "Pick up the red cube and put it in the blue box."
JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
TMP = Path("/tmp/minimind-a-rollout-video")


def save(name, data):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def in_box(cube, box):
    return bool(
        abs(cube[0] - box[0]) < 0.05 and abs(cube[1] - box[1]) < 0.05 and cube[2] < 0.05
    )


def video_writer(path, width, height):
    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=FPS)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "23", "preset": "fast"}
    return container, stream


def add_frame(container, stream, image):
    for packet in stream.encode(
        av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")
    ):
        container.mux(packet)


def close_video(container, stream):
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def font(size):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def panel(view, top, episode, seconds, status, lifted):
    image = Image.new("RGB", (960, 540), (18, 23, 32))
    image.paste(Image.fromarray(view), (0, 60))
    image.paste(Image.fromarray(top), (640, 60))
    draw = ImageDraw.Draw(image)
    draw.text(
        (18, 14), f"MiniMind-A | Episode {episode:02d}/10", fill="white", font=font(26)
    )
    draw.text(
        (660, 14),
        f"{seconds:4.1f} / {MAX_SECONDS:.0f} sim seconds",
        fill="white",
        font=font(18),
    )
    draw.text((660, 322), "Goal: red cube", fill="white", font=font(22))
    draw.text((660, 355), "into blue box", fill="white", font=font(22))
    draw.text(
        (660, 408),
        "Lifted: " + ("YES" if lifted else "NO"),
        fill=(115, 190, 255),
        font=font(22),
    )
    color = (
        (83, 224, 153)
        if status == "SUCCESS"
        else ((255, 123, 123) if status == "FAIL" else (255, 219, 111))
    )
    draw.text((660, 458), status, fill=color, font=font(28))
    draw.text((660, 506), "Actual policy rollout", fill=(160, 170, 186), font=font(17))
    return image


def main():
    ROOT.mkdir(exist_ok=True)
    TMP.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    np.random.seed(1000)
    gpu = torch.cuda.get_device_properties(0)
    assert any(name in gpu.name for name in ("A40", "B300", "L40S"))
    assert gpu.total_memory > 20 * 1024**3
    checkpoint = TRAIN / "model/best_vla.pt"
    expected = json.loads((TRAIN / "SHA256.json").read_text())["model/best_vla.pt"]
    assert sha(checkpoint) == expected
    model = MiniMindVLA.load_checkpoint(checkpoint).cuda().eval()
    stats = json.loads((TRAIN / "normalization.json").read_text())
    state_mean = np.asarray(stats["state"]["mean"], np.float32)
    state_std = np.asarray(stats["state"]["std"], np.float32)
    action_mean = np.asarray(stats["action"]["mean"], np.float32)
    action_std = np.asarray(stats["action"]["std"], np.float32)
    tok = AutoTokenizer.from_pretrained(TRAIN / "source/model", local_files_only=True)
    prompt = tok.apply_chat_template(
        [dict(role="user", content=TASK)], tokenize=False, add_generation_prompt=True
    )
    tokens = [12] * 64 + tok(prompt, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([tokens], device="cuda")
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    menagerie = Path(os.environ["MENAGERIE_DIR"])
    if (menagerie / "REVISION").exists():
        asset_commit = (menagerie / "REVISION").read_text().strip()
    else:
        asset_commit = subprocess.check_output(
            ["git", "-C", str(menagerie), "rev-parse", "HEAD"], text=True
        ).strip()
    settings = dict(
        episodes=EPISODES,
        gpu=gpu.name,
        checkpoint=str(checkpoint),
        checkpoint_sha256=expected,
        position_seed=1000,
        noise_seeds=list(range(10)),
        fps=FPS,
        maximum_sim_seconds=MAX_SECONDS,
        predicted_action_steps=16,
        executed_action_steps=EXECUTE,
        euler_steps=STEPS,
        camera="top",
        policy_image_shape=[240, 320, 3],
        box_position=[0.05, 0.22],
        success_definition="cube center abs(dx),abs(dy)<0.05m and z<0.05m for at least0.5s consecutively; still in box at episode end",
        original_simulator="ady95/physicalai_tutorial",
        simulator_commit="2cb8e6b1e54ed3e47233425be51cddc0ea02a5f0",
        menagerie_commit=asset_commit,
        mujoco=mujoco.__version__,
        render_backend=os.environ["MUJOCO_GL"],
        policy_inputs="top RGB +6observed joints+instruction only; cube coordinates never supplied to policy",
        action_handling="denormalize using training statistics; clip only native actuator control limits",
        adaptation="No retraining, no oracle IK, no scripted assistance, no episode selection",
    )
    save("settings.json", settings)
    print("MODEL_READY", gpu.name, expected, flush=True)
    # Preserve the exact asset snapshot used, including its upstream license.
    with __import__("tarfile").open(ROOT / "simulator_assets.tar.gz", "w:gz") as tar:
        tar.add(menagerie / "robotstudio_so101", arcname="robotstudio_so101")
        for name in ["LICENSE", "README.md"]:
            if (menagerie / name).exists():
                tar.add(menagerie / name, arcname=name)
    rng = np.random.default_rng(1000)
    results = []
    started = time.time()
    for episode in range(1, EPISODES + 1):
        radius, theta = rng.uniform(0.17, 0.27), rng.uniform(-0.7, 0.7)
        cube = (float(radius * np.cos(theta)), float(radius * np.sin(theta)))
        torch.manual_seed(episode - 1)
        torch.cuda.manual_seed_all(episode - 1)
        robot = SO101Sim(cube_pos=cube, box_pos=(0.05, 0.22), render=False)
        assert [
            mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(6)
        ] == JOINTS
        top_renderer = mujoco.Renderer(robot.model, height=240, width=320)
        spectator = mujoco.Renderer(robot.model, height=480, width=640)
        local_video = TMP / f"episode_{episode:02d}.mp4"
        writer, stream = video_writer(local_video, 960, 540)
        queue = deque()
        consecutive = 0
        stable = False
        stable_time = None
        lifted = False
        max_height = 0.0
        states = []
        actions = []
        positions = []
        clipped = 0
        predictions = 0
        inference_seconds = 0.0
        step_count = int(round(1 / (FPS * robot.model.opt.timestep)))
        try:
            for step in range(FPS * MAX_SECONDS):
                top_renderer.update_scene(robot.data, camera="top")
                rgb = top_renderer.render().copy()
                state = robot.data.qpos[:6].astype(np.float32).copy()
                if not queue and not stable:
                    pixels = model.vlm.processor(
                        images=Image.fromarray(rgb), return_tensors="pt"
                    )["pixel_values"].cuda()
                    state_tensor = torch.from_numpy((state - state_mean) / state_std)[
                        None
                    ].cuda()
                    before = time.perf_counter()
                    with (
                        torch.inference_mode(),
                        torch.autocast("cuda", dtype=torch.bfloat16),
                    ):
                        prediction = model.sample_actions(
                            input_ids, attention_mask, pixels, state_tensor, steps=STEPS
                        )
                    raw = prediction[0].float().cpu().numpy() * action_std + action_mean
                    inference_seconds += time.perf_counter() - before
                    assert raw.shape == (16, 6) and np.isfinite(raw).all()
                    queue.extend(raw[:EXECUTE])
                    predictions += 1
                if not stable:
                    action = queue.popleft()
                    limits = robot.model.actuator_ctrlrange[:6]
                    limited = robot.model.actuator_ctrllimited[:6].astype(bool)
                    bounded = np.where(
                        limited, np.clip(action, limits[:, 0], limits[:, 1]), action
                    )
                    clipped += int(np.count_nonzero(bounded != action))
                    robot.data.ctrl[:6] = bounded
                actions.append(robot.data.ctrl[:6].copy())
                states.append(state)
                robot.step(step_count)
                position = robot.cube_pos()
                positions.append(position.tolist())
                max_height = max(max_height, float(position[2]))
                lifted = lifted or position[2] > 0.06
                consecutive = (
                    consecutive + 1 if in_box(position, robot.box_pos()) else 0
                )
                if consecutive >= 10 and not stable:
                    stable = True
                    stable_time = (step + 1) / FPS
                    queue.clear()
                spectator.update_scene(robot.data, camera="fixed")
                view = spectator.render().copy()
                top_renderer.update_scene(robot.data, camera="top")
                top = top_renderer.render().copy()
                status = (
                    "SUCCESS"
                    if stable
                    else ("FAIL" if step == FPS * MAX_SECONDS - 1 else "RUNNING")
                )
                frame = panel(view, top, episode, (step + 1) / FPS, status, lifted)
                add_frame(writer, stream, frame)
                if step % 60 == 59:
                    save(
                        "progress.json",
                        dict(
                            completed_episodes=episode - 1,
                            current_episode=episode,
                            frame=step + 1,
                            total_frames=240,
                            total_episodes=10,
                            successes=sum(r["success"] for r in results),
                            elapsed_seconds=time.time() - started,
                        ),
                    )
                    print("FRAME", episode, step + 1, flush=True)
                if step in [0, 119, 239]:
                    (ROOT / "previews").mkdir(exist_ok=True)
                    frame.save(
                        ROOT / f"previews/episode_{episode:02d}_step_{step:03d}.jpg"
                    )
            success = bool(stable and in_box(robot.cube_pos(), robot.box_pos()))
        finally:
            close_video(writer, stream)
            top_renderer.close()
            spectator.close()
            robot.close()
        path = ROOT / f"videos/episode_{episode:02d}.mp4"
        path.parent.mkdir(exist_ok=True)
        shutil.copyfile(local_video, path)
        np.savez(
            ROOT / f"episode_{episode:02d}_trajectory.npz",
            states=np.asarray(states),
            actions=np.asarray(actions),
            cube_positions=np.asarray(positions),
        )
        result = dict(
            episode=episode,
            cube_initial_xy=list(cube),
            noise_seed=episode - 1,
            success=success,
            stable_success_time=stable_time,
            lifted=bool(lifted),
            max_cube_height=max_height,
            cube_final_xyz=positions[-1],
            frames=240,
            simulation_seconds=12.0,
            video=str(path.relative_to(ROOT)),
            prediction_calls=predictions,
            inference_seconds=inference_seconds,
            clipped_control_values=clipped,
            all_actions_finite=True,
            video_sha256=sha(path),
        )
        results.append(result)
        save(f"episode_{episode:02d}.json", result)
        save(
            "progress.json",
            dict(
                completed_episodes=episode,
                total_episodes=10,
                successes=sum(r["success"] for r in results),
                elapsed_seconds=time.time() - started,
            ),
        )
        print("EPISODE", json.dumps(result), flush=True)
    # All episodes are retained, regardless of outcome.
    (TMP / "concat.txt").write_text(
        "".join(f"file '{ROOT}/videos/episode_{i:02d}.mp4'\n" for i in range(1, 11))
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(TMP / "concat.txt"),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(TMP / "all_10_episodes.mp4"),
        ],
        check=True,
    )
    shutil.copyfile(TMP / "all_10_episodes.mp4", ROOT / "all_10_episodes.mp4")
    # 4x3 mosaic: ten synchronized real rollouts, two summary tiles.
    readers = [av.open(str(ROOT / r["video"])) for r in results]
    decoders = [iter(r.decode(video=0)) for r in readers]
    mosaic_writer, mosaic_stream = video_writer(TMP / "overview.mp4", 1280, 864)
    for step in range(240):
        mosaic = Image.new("RGB", (1280, 864), (18, 23, 32))
        draw = ImageDraw.Draw(mosaic)
        for i, decoder in enumerate(decoders):
            frame = next(decoder).to_image().crop((0, 60, 640, 540)).resize((320, 240))
            x = (i % 4) * 320
            y = (i // 4) * 288
            mosaic.paste(frame, (x, y + 40))
            status = "SUCCESS" if results[i]["success"] else "FAIL"
            draw.text(
                (x + 8, y + 8),
                f"Ep {i + 1:02d} | {status}",
                fill=(83, 224, 153) if results[i]["success"] else (255, 123, 123),
                font=font(22),
            )
        n = sum(r["success"] for r in results)
        draw.text((660, 632), "MiniMind-A", fill="white", font=font(28))
        draw.text((660, 682), "10 actual policy episodes", fill="white", font=font(23))
        draw.text((660, 728), f"Success: {n}/10", fill=(255, 219, 111), font=font(26))
        draw.text(
            (660, 780),
            f"Simulation time: {(step + 1) / 20:.1f}s",
            fill="white",
            font=font(22),
        )
        add_frame(mosaic_writer, mosaic_stream, mosaic)
    close_video(mosaic_writer, mosaic_stream)
    for reader in readers:
        reader.close()
    shutil.copyfile(TMP / "overview.mp4", ROOT / "overview.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(TMP / "overview.mp4"),
            "-filter_complex",
            "fps=5,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            str(TMP / "overview.gif"),
        ],
        check=True,
    )
    shutil.copyfile(TMP / "overview.gif", ROOT / "overview.gif")
    for path in list((ROOT / "videos").glob("*.mp4")) + [
        ROOT / "overview.mp4",
        ROOT / "all_10_episodes.mp4",
    ]:
        with av.open(str(path)) as movie:
            frames = sum(1 for _ in movie.decode(video=0))
            assert frames == (2400 if path.name == "all_10_episodes.mp4" else 240), (
                path,
                frames,
            )
    report = dict(
        episodes=10,
        successes=sum(r["success"] for r in results),
        success_rate=sum(r["success"] for r in results) / 10,
        lifted_count=sum(r["lifted"] for r in results),
        all_videos_decode=True,
        results=results,
        elapsed_seconds=time.time() - started,
        checkpoint_sha256=expected,
        simulation_rollout=True,
        note="10 fixed new scenes, one noise seed each, no retraining or selection; small-sample success rate",
    )
    save("report.json", report)
    paths = [
        p
        for p in ROOT.rglob("*")
        if p.is_file()
        and p.suffix in [".mp4", ".gif", ".jpg", ".json", ".npz", ".gz", ".py", ".md"]
        and p.name not in ["SHA256.json", "progress.json"]
        and "__pycache__" not in p.parts
    ]
    save("SHA256.json", {str(p.relative_to(ROOT)): sha(p) for p in paths})
    save(
        "SUCCESS",
        dict(
            episodes=10,
            videos=10,
            all_videos_decode=True,
            successes=report["successes"],
            report_sha256=sha(ROOT / "report.json"),
        ),
    )
    print("ROLLOUT_COMPLETE", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
