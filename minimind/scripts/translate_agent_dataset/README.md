# Agent RL English conversion

Run from the repository root. Requires Python 3, `requests`, and an OpenRouter
key in the root `.env` (`OPENROUTER-API-KEY`) or `OPENROUTER_API_KEY` in the environment.
Publishing additionally requires `huggingface_hub` and an authenticated Hugging Face login.

```bash
curl -L --fail -o /tmp/agent_rl.jsonl \
  https://huggingface.co/datasets/jingyaogong/minimind_dataset/resolve/main/agent_rl.jsonl
python3 minimind/scripts/translate_agent_dataset/translate.py \
  --source /tmp/agent_rl.jsonl --output-dir /tmp/agent-rl-en --workers 32
```

Translation uses `minimax/minimax-m3.1-flash` through OpenRouter. It excludes
every record containing `translate_text`, including tool definitions. It translates
message text, reasoning content, tool descriptions and textual ground truths,
reusing cached translations of identical strings. Machine-readable strings and
numeric ground truths are retained. The source can contain Unicode paragraph
separators; use JSON decoding rather than Python `str.splitlines()` to read it.

`translation_cache.jsonl` supports resuming by rerunning the command with the
same output directory. `usage.jsonl` records successful request usage; it excludes
failed/retried requests and interrupted in-flight requests, so it is not a billing
total. `validation.json` and `agent_rl_en.jsonl` are emitted only when all required
translations are available. `--limit-batches 2` performs a small pilot.

Before publishing, prepare a dataset card named `README.md` in the output
directory with source attribution, original license notices, filtering rules,
translation limitations and a data-files configuration. Review translation samples.

```bash
minimind/.venv/bin/python minimind/scripts/translate_agent_dataset/publish.py \
  --directory /tmp/agent-rl-en --repo caspar/agent-rl-en
```

Publishing creates a **public** dataset and verifies the uploaded file checksum
by downloading the committed artifact. It refuses to overwrite an existing
repository. Only the dataset, card, validation report and translation script are
uploaded; credentials and caches remain local. Expected counts are pinned to the
2026-10-26 source snapshot (39,988 input, 9,792 excluded, 30,196 output).
