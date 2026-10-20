"""Verify shared-volume artifacts, copy lightweight reports, remove only task Pod."""

import argparse
import base64
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import urllib.request


def validate_completion(success, settings, evaluation, hashes):
    if not success.get("shared_volume_verified"):
        raise ValueError("Shared-volume completion not verified")
    if (
        evaluation.get("epochs") != settings["epochs"]
        or success.get("epochs") != settings["epochs"]
    ):
        raise ValueError("Incomplete epochs")
    expected = settings["optimizer_updates_target"]
    if (
        evaluation.get("optimizer_updates") != expected
        or success.get("optimizer_updates") != expected
    ):
        raise ValueError("Incomplete optimizer updates")
    for flag in [
        "strict_load",
        "finite_weights",
        "frozen_vlm_bitwise_unchanged",
        "image_inference_finite",
    ]:
        if not evaluation.get(flag):
            raise ValueError("Missing verification: " + flag)
    checksum = hashes.get("model/final_vla.pt")
    if (
        not checksum
        or checksum != success.get("model_sha256")
        or checksum != evaluation.get("checkpoint_sha256")
    ):
        raise ValueError("Checkpoint checksum disagreement")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pod-id", required=True)
    p.add_argument("--pod-name", required=True)
    p.add_argument("--remote-root", required=True)
    p.add_argument("--destination", required=True)
    p.add_argument("--env-file", required=True)
    p.add_argument("--ssh-key", required=True)
    a = p.parse_args()
    dest = Path(a.destination)
    dest.mkdir(parents=True, exist_ok=True)
    status = dest / "watch-status.json"
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        for line in Path(a.env_file).read_text().splitlines():
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            k, v = line.removeprefix("export ").split("=", 1)
            if k.strip().upper().replace("-", "_") == "RUNPOD_API_KEY":
                key = shlex.split(v, comments=True)[0]
    assert key

    def api(method, route):
        request = urllib.request.Request(
            "https://api.runpod.io/v2/" + route,
            method=method,
            headers={"Authorization": "Bearer " + key},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
            return json.loads(data) if data else None

    def record(**values):
        temporary = status.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(dict(time=time.time(), pod_id=a.pod_id, **values), indent=2)
        )
        temporary.replace(status)

    def remote(pod, code):
        ssh = pod["ssh"]["direct"]
        args = [
            "ssh",
            "-i",
            a.ssh_key,
            "-p",
            str(ssh["port"]),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "UserKnownHostsFile=/tmp/minimind-a-known-hosts",
            "-o",
            "ConnectTimeout=20",
            "-o",
            "ServerAliveInterval=20",
            ssh["username"] + "@" + ssh["host"],
            "python3 -c " + shlex.quote(code),
        ]
        r = subprocess.run(args, capture_output=True, text=True, timeout=180)
        if r.returncode:
            raise RuntimeError((r.stderr + r.stdout)[-2000:])
        return json.loads(r.stdout)

    poll = f"""import json
from pathlib import Path
r=Path({a.remote_root!r}); result={{}}
for n in ['phase.json','progress.json','training_settings.json','evaluation_before.json','evaluation_after.json','SUCCESS','SHA256.json']:
 p=r/n
 if p.exists():result[n]=json.loads(p.read_text())
p=r/'exit_code';result['exit_code']=p.read_text().strip() if p.exists() else None
print(json.dumps(result))"""
    verify = f"""import hashlib,json,base64
from pathlib import Path
r=Path({a.remote_root!r}).resolve(); hashes=json.loads((r/'SHA256.json').read_text())
for name,expected in hashes.items():
 p=(r/name).resolve();assert p.is_relative_to(r),name
 with p.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
 assert actual==expected,name
paths=list(r.glob('*.json'))+list(r.glob('*.npz'))+[r/'SUCCESS',r/'requirements-resolved.txt',r/'model/config.json']+list((r/'examples').glob('*.jpg'))
result={{'verified_files':len(hashes),'files':{{str(p.relative_to(r)):base64.b64encode(p.read_bytes()).decode() for p in paths}}}}
print(json.dumps(result))"""
    while True:
        try:
            pod = api("GET", "pods/" + a.pod_id)
            if pod["id"] != a.pod_id or pod["name"] != a.pod_name:
                raise RuntimeError("Task Pod provenance mismatch")
            state = remote(pod, poll)
            for name in [
                "phase.json",
                "progress.json",
                "training_settings.json",
                "evaluation_before.json",
                "evaluation_after.json",
            ]:
                if name in state:
                    (dest / name).write_text(json.dumps(state[name], indent=2))
            if state["exit_code"] not in [None, "0"]:
                record(status="training_failed", exit_code=state["exit_code"])
                return
            if state["exit_code"] == "0":
                validate_completion(
                    state["SUCCESS"],
                    state["training_settings.json"],
                    state["evaluation_after.json"],
                    state["SHA256.json"],
                )
                record(status="verify_shared_volume")
                result = remote(pod, verify)
                for name, data in result["files"].items():
                    target = (dest / name).resolve()
                    assert target.is_relative_to(dest.resolve())
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(base64.b64decode(data))
                receipt = dict(
                    pod_id=a.pod_id,
                    pod_name=a.pod_name,
                    remote_root=a.remote_root,
                    verified_files=result["verified_files"],
                    model_sha256=state["SUCCESS"]["model_sha256"],
                    shared_volume_verified=True,
                    time=time.time(),
                )
                (dest / "verified_shared_volume_receipt.json").write_text(
                    json.dumps(receipt, indent=2)
                )
                # Durable remote receipt precedes API teardown, and describes the authorized action.
                remote(
                    pod,
                    f"import json;from pathlib import Path;Path({(a.remote_root + '/verified_shared_volume_receipt.json')!r}).write_text({json.dumps(receipt)!r});print(json.dumps({{'saved':True}}))",
                )
                record(
                    status="terminating_verified_task_pod",
                    verified_files=result["verified_files"],
                )
                api("DELETE", "pods/" + a.pod_id)
                pods = api("GET", "pods")["pods"]
                assert all(pod["id"] != a.pod_id for pod in pods), (
                    "Deletion not confirmed"
                )
                volumes = api("GET", "global-volumes")["globalVolumes"]
                assert any(v["id"] == "cmuxpmy6p000007l12uzr990b" for v in volumes)
                cleanup = dict(
                    **receipt,
                    pod_deleted=True,
                    shared_volume_preserved=True,
                    remaining_pod_ids=[p["id"] for p in pods],
                )
                (dest / "cleanup.json").write_text(json.dumps(cleanup, indent=2))
                record(
                    status="complete",
                    pod_deleted=True,
                    verified_files=result["verified_files"],
                )
                return
            record(
                status="watching",
                phase=state.get("phase.json"),
                progress=state.get("progress.json"),
            )
        except Exception as error:
            record(status="retrying", error=str(error))
        time.sleep(20)


if __name__ == "__main__":
    main()
