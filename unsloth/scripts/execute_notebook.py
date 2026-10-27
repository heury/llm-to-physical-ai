"""Execute an unchanged notebook in this interpreter and retain per-cell evidence."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cell-timeout", type=int, default=7200)
    args = parser.parse_args()
    source = args.notebook.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(source, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    status = {
        "notebook": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "pod_id": os.environ.get("RUNPOD_POD_ID"),
        "python": sys.version,
        "executable": sys.executable,
        "started_at": time.time(),
        "status": "running",
        "cells": [],
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
    }
    gpu = subprocess.run(["nvidia-smi"], text=True, capture_output=True)
    status["nvidia_smi"] = gpu.stdout + gpu.stderr

    def save():
        nbformat.write(notebook, output / "executed.ipynb")
        temporary = output / "status.json.tmp"
        temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(output / "status.json")

    def started(cell, cell_index, **kwargs):
        if cell.cell_type != "code":
            return
        status["active_cell"] = cell_index
        status["cells"].append({"index": cell_index, "started_at": time.time()})
        print(f"CELL_START {cell_index}: {cell.source.splitlines()[0][:120]}", flush=True)
        save()

    def executed(cell, cell_index, **kwargs):
        record = status["cells"][-1]
        record["finished_at"] = time.time()
        record["errors"] = [
            {"ename": o.ename, "evalue": o.evalue}
            for o in cell.outputs if o.output_type == "error"
        ]
        for item in cell.outputs:
            if item.output_type == "stream":
                continue
            text = item.get("text") or item.get("data", {}).get("text/plain", "")
            if text:
                print(text, flush=True)
        print(f"CELL_DONE {cell_index}", flush=True)
        save()

    # Do not use the image's system kernel when a uv environment is active.
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    class EvidenceClient(NotebookClient):
        last_save = 0.0

        def process_message(self, msg, cell, cell_index):
            result = super().process_message(msg, cell, cell_index)
            if msg["msg_type"] == "stream":
                print(msg["content"]["text"], end="", flush=True)
            if time.monotonic() - self.last_save >= 5:
                save()
                self.last_save = time.monotonic()
            return result

    client = EvidenceClient(
        notebook,
        km=manager,
        timeout=args.cell_timeout,
        allow_errors=False,
        resources={"metadata": {"path": str(output)}},
        on_cell_start=started,
        on_cell_executed=executed,
    )
    save()
    try:
        client.execute()
        # IPython shell escapes do not raise on a failing export command.
        if "Phone_Deployment" in source.name:
            exported = output / "qwen3_0.6B_model.pte"
            if not exported.is_file() or exported.stat().st_size == 0:
                raise RuntimeError("ExecuTorch export did not produce a nonempty .pte file")
        status["status"] = "passed"
    except BaseException:
        status["status"] = "failed"
        status["traceback"] = traceback.format_exc()
        print(status["traceback"], flush=True)
    finally:
        status["finished_at"] = time.time()
        save()
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    print(f"NOTEBOOK_RESULT {status['status']}", flush=True)
    return 0 if status["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
