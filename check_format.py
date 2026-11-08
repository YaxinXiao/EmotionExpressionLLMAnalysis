"""Check constrained decoding on CPU with the saved tokenizer; no model weights loaded."""
import argparse
import importlib.metadata
import json
from pathlib import Path

from analyze import EOS_TOKEN_IDS, METRICS, ROOT, TEXT_KEYS, make_format_filter, parse_response


def check_tokenizer(tokenizer):
    import torch
    from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data

    for token, expected in zip(['<|im_end|>', '<|endoftext|>'], EOS_TOKEN_IDS):
        if tokenizer.convert_tokens_to_ids(token) != expected:
            raise RuntimeError('Tokenizer does not match the pinned Qwen model')
    data = build_token_enforcer_tokenizer_data(tokenizer)
    prompt = tokenizer.encode('格式检查，不是视频评分。', add_special_tokens=False)

    def walk(text):
        fn = make_format_filter(data)
        ids = list(prompt)
        allowed = fn(0, torch.tensor(ids))
        for token in tokenizer.encode(text, add_special_tokens=False):
            if token not in allowed:
                tail = tokenizer.decode(ids[len(prompt):])[-100:]
                raise RuntimeError(f'Valid JSON token rejected: {tokenizer.decode([token])!r} after {tail!r}')
            ids.append(token)
            allowed = fn(0, torch.tensor(ids))
        return allowed

    # This fixture is synthetic and is never written to a results directory.
    value = {'scores': {
        name: {'score': [1, 2, 3, 4, 5, None][i % 6],
               'evidence': '[2.0-5.0 s] 测试：中文、English、引号 " 与括号 }；非真实评分。',
               'confidence': ['low', 'medium', 'high'][i % 3]}
        for i, name in enumerate(METRICS)},
        **{name: '仅测试格式，未分析视频。' for name in TEXT_KEYS}}
    raw = json.dumps(value, ensure_ascii=False)
    allowed = walk(raw)
    if tokenizer.eos_token_id not in allowed or parse_response(raw) != value:
        raise RuntimeError('Complete JSON does not round-trip or cannot terminate')

    def reject(prefix, forbidden):
        allowed = walk(prefix)
        for text in forbidden:
            first = tokenizer.encode(text, add_special_tokens=False)[0]
            if first in allowed:
                raise RuntimeError(f'Invalid continuation allowed: {prefix!r} + {text!r}')
        if any(token in allowed for token in EOS_TOKEN_IDS):
            raise RuntimeError('Incomplete answer permits early end-of-sequence')

    reject('{"scores":{"valence":', ['4', 'null', '"'])
    reject('{"scores":{"valence":{"score":', ['0', '6', 'true', '"'])
    reject('{"scores":{"valence":{"score":3', ['}', '0', '.'])
    reject('{"scores":{"valence":{"score":3,"evidence":"测试"', ['}'])
    print('PASS: 14 nested metrics; Chinese evidence; scores 1-5/null; required fields; EOS.', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=ROOT / 'models' / 'Qwen2.5-Omni-7B')
    args = p.parse_args()
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(1)
    print('Versions:', {name: importlib.metadata.version(name) for name in
                        ['torch', 'transformers', 'lm-format-enforcer', 'pydantic']}, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    check_tokenizer(tokenizer)


if __name__ == '__main__':
    main()
