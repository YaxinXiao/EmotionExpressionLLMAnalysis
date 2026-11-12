"""Check constrained decoding on CPU with the saved tokenizer; no model weights loaded."""
import argparse
import importlib.metadata
import json
from pathlib import Path

from analyze import EOS_TOKEN_IDS, METRICS, ROOT, TEXT_KEYS, make_format_filter, parse_response


def walk_allowed_text(tokenizer, prompt, text, allowed_tokens_fn):
    """Reach the exact fixture through allowed tokens, permitting lossless BPE splits.

    LMFE's JSON free-text shortcut can reject a token spanning text, a closing
    quote, and a delimiter even when the same characters are legal separately.
    This checks an allowed spelling of the fixture, not one specific BPE encoding.
    Every replacement token must still pass the same production token mask.
    """
    ids = list(prompt)
    prompt_length = len(ids)
    allowed = allowed_tokens_fn(ids)
    split_count = 0

    def decode(tokens):
        return tokenizer.decode(tokens, clean_up_tokenization_spaces=False)

    def advance(token):
        nonlocal allowed
        if token not in allowed:
            tail = decode(ids[prompt_length:])[-100:]
            raise RuntimeError(f'Fixture has no allowed token path: {decode([token])!r} after {tail!r}')
        ids.append(token)
        allowed = allowed_tokens_fn(ids)

    for token in tokenizer.encode(text, add_special_tokens=False):
        if token in allowed:
            advance(token)
            continue
        piece = decode([token])
        replacement = [part for character in piece
                       for part in tokenizer.encode(character, add_special_tokens=False)]
        generated = ids[prompt_length:]
        if (not piece or '\ufffd' in piece or len(replacement) < 2
                or decode(replacement) != piece
                or decode(generated + replacement) != decode(generated + [token])):
            raise RuntimeError(f'Cannot losslessly split rejected fixture token: {piece!r}')
        for part in replacement:
            advance(part)
        split_count += 1

    if decode(ids[prompt_length:]) != text:
        raise RuntimeError('Allowed fixture tokens did not decode to the exact original text')
    return allowed, split_count


def check_tokenizer(tokenizer):
    import torch
    from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data

    for token, expected in zip(['<|im_end|>', '<|endoftext|>'], EOS_TOKEN_IDS):
        if tokenizer.convert_tokens_to_ids(token) != expected:
            raise RuntimeError('Tokenizer does not match the pinned Qwen model')
    data = build_token_enforcer_tokenizer_data(tokenizer)
    prompt = tokenizer.encode('Format check only; no video is being rated.', add_special_tokens=False)
    split_count = 0

    def walk(text):
        nonlocal split_count
        fn = make_format_filter(data)
        allowed, splits = walk_allowed_text(tokenizer, prompt, text,
            lambda ids: fn(0, torch.tensor(ids, dtype=torch.long)))
        split_count += splits
        return allowed

    # This fixture is synthetic and is never written to a results directory.
    value = {'scores': {
        name: {'score': [1, 2, 3, 4, 5, None][i % 6],
               'evidence': '[2.0-5.0 s] Synthetic fixture: gesture, quote " and brace }; not a video rating.',
               'confidence': ['low', 'medium', 'high'][i % 3]}
        for i, name in enumerate(METRICS)},
        **{name: 'Synthetic nonverbal fixture only; no video analyzed.' for name in TEXT_KEYS}}
    raw = json.dumps(value, ensure_ascii=False)
    allowed = walk(raw)
    if tokenizer.eos_token_id not in allowed or parse_response(raw) != value:
        raise RuntimeError('Complete JSON does not round-trip or cannot terminate')

    # Preserve Unicode regression coverage without changing the English task prompt.
    value['scores']['valence']['evidence'] = '[2.0-5.0 s] Unicode fixture only: 中文, café, quote " and brace }.'
    unicode_raw = json.dumps(value, ensure_ascii=False)
    if tokenizer.eos_token_id not in walk(unicode_raw) or parse_response(unicode_raw) != value:
        raise RuntimeError('Unicode fixture does not round-trip or cannot terminate')

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
    reject('{"scores":{"valence":{"score":3,"evidence":"Synthetic fixture"', ['}'])
    print(f'Fixture composite tokens split losslessly: {split_count}', flush=True)
    print('PASS: 14 nested metrics; four nonverbal text fields; English/Unicode fixtures; scores 1-5/null; EOS.', flush=True)


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
