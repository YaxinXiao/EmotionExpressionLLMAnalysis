"""Download fixed weights on a network-enabled login node; no GPU/model loading."""
import json
from pathlib import Path
from huggingface_hub import snapshot_download

MODEL_ID = 'Qwen/Qwen2.5-Omni-7B'
REVISION = 'ae9e1690543ffd5c0221dc27f79834d0294cba00'
dest = Path(__file__).resolve().parent / 'models' / 'Qwen2.5-Omni-7B'
snapshot_download(MODEL_ID, revision=REVISION, local_dir=dest,
                  allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.tiktoken', '*.jinja'], max_workers=4)
index = json.loads((dest / 'model.safetensors.index.json').read_text())
for name in set(index['weight_map'].values()):
    if not (dest / name).is_file() or (dest / name).stat().st_size == 0:
        raise RuntimeError(f'Missing weight file: {name}')
(dest / 'download_identity.json').write_text(json.dumps({'model_id': MODEL_ID, 'revision': REVISION}))
print('Download complete:', dest)
