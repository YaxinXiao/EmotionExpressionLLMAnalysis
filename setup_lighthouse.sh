#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo 'Run setup on a login node, after exiting the interactive GPU job.' >&2
  exit 1
fi
module load python/3.12.1
if [ ! -x .venv/bin/python ]; then python3 -m venv .venv; fi
source .venv/bin/activate
python -m pip install --no-cache-dir --upgrade pip
python -m pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install --no-cache-dir --only-binary=av -r requirements.txt
python - <<'PY'
import imageio_ffmpeg
from pathlib import Path
import sys
target = Path(sys.executable).parent / 'ffmpeg'
if not target.exists():
    target.symlink_to(imageio_ffmpeg.get_ffmpeg_exe())
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
from qwen_omni_utils import process_mm_info
print('Required imports passed.')
PY
python -m pip check
python -m pip freeze > environment.txt
python download_model.py
