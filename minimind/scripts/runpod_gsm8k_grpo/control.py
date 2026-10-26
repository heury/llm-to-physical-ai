"""Local-only authenticated CLI wrapper. Never copy this environment to a Pod."""
import os
from pathlib import Path
import subprocess
import sys


def credentials():
    config = {}
    for line in (Path(__file__).resolve().parents[3] / '.env').read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            key, value = line.split('=', 1)
            config[key.strip()] = value.strip().strip('\"\'')
    return config


if __name__ == '__main__':
    config = credentials()
    env = dict(os.environ, RUNPOD_API_KEY=config['RUNPOD-API-KEY'])
    raise SystemExit(subprocess.call(['runpodctl', *sys.argv[1:]], env=env))
