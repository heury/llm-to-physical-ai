#!/usr/bin/env bash
set -euo pipefail
cd /workspace/minimind-agent-retry1
python3 -c 'import json,time; from pathlib import Path; Path("status.json").write_text(json.dumps({"stage":"bootstrapping","updated_at":time.time()}))'
trap 'code=$?; python3 -c '\''import json,time,pathlib,sys; p=pathlib.Path("status.json"); old=json.loads(p.read_text()) if p.exists() else {}; p.write_text(json.dumps(old if old.get("stage")=="failed" else {"stage":"failed","error":"bootstrap exited "+sys.argv[1],"updated_at":time.time()}))'\'' "$code"; exit "$code"' ERR
export HF_HOME=/workspace/hf-cache
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
uv pip install --system --break-system-packages 'sglang==0.5.5' 'datasets==4.3.0' 'wandb>=0.22,<0.24' > install.log 2>&1
uv pip install --system --break-system-packages --no-deps 'vllm==0.11.0' >> install.log 2>&1
uv pip install --system --break-system-packages cachetools cloudpickle msgspec cbor2 blake3 >> install.log 2>&1
python3 scripts/runpod_agent/prepare.py > prepare.log 2>&1
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('Skywork/Skywork-Reward-V2-Qwen3-1.7B',revision='e51ea3e08fb81326c3b812a7ff0cb9cee83e59cc',local_dir='/workspace/minimind-agent-retry1/reward_skywork')
PY
python3 -m sglang.launch_server --model-path ./sglang_policy --attention-backend triton \
  --host 127.0.0.1 --port 8998 --dtype bfloat16 --mem-fraction-static 0.18 \
  --max-running-requests 32 --context-length 32768 --sampling-backend pytorch \
  > sglang.log 2>&1 &
echo $! > sglang.pid
python3 - <<'PY'
import requests,time
for _ in range(180):
 try:
  if requests.get('http://127.0.0.1:8998/health',timeout=5).status_code==200:break
 except requests.RequestException:pass
 time.sleep(5)
else:raise RuntimeError('SGLang startup timeout')
print('SGLANG_READY')
PY
python3 scripts/runpod_agent/run.py > job.log 2>&1
