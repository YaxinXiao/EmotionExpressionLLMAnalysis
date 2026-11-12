"""Score audiovisual debate expression with Qwen2.5-Omni on an allocated GPU."""
import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_ID = 'Qwen/Qwen2.5-Omni-7B'
REVISION = 'ae9e1690543ffd5c0221dc27f79834d0294cba00'
ANALYSIS_MODE = 'nonverbal_prompt_only'
MEASUREMENT_TARGET = 'perceived_nonverbal_expression'
METRIC_OPERATIONALIZATIONS = {
    'self_esteem': 'Displayed self-assurance/self-affirmation; not actual self-worth or trait self-esteem.',
    'guilty': 'Perceived apologetic/remorse-like display; not actual guilt, responsibility, or wrongdoing.',
}
METRICS = ['valence', 'arousal', 'ashamed', 'feelings_of_closeness_to_others', 'elated',
           'self_esteem', 'frustration', 'happy', 'calm', 'proud', 'guilty', 'angry',
           'friendly_feelings', 'unhappy']
TEXT_KEYS = ['visual_evidence', 'audio_evidence', 'interaction_pattern', 'limitations']
MAX_NEW_TOKENS = 4096
EOS_TOKEN_IDS = [151645, 151643]  # <|im_end|>, <|endoftext|>, verified for the fixed revision.
SYSTEM_TEXT = ('You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, '
               'capable of perceiving auditory and visual inputs, as well as generating text and speech.')


def response_schema():
    """Constrain structure during decoding; do not supply or repair model ratings."""
    metric = {
        'type': 'object',
        'properties': {
            'score': {'anyOf': [{'enum': [1, 2, 3, 4, 5]}, {'type': 'null'}]},
            'evidence': {'type': 'string', 'minLength': 1, 'maxLength': 180},
            'confidence': {'type': 'string', 'enum': ['low', 'medium', 'high']},
        },
        'required': ['score', 'evidence', 'confidence'],
        'additionalProperties': False,
    }
    return {
        'type': 'object',
        'properties': {
            'scores': {'type': 'object', 'properties': {key: metric for key in METRICS},
                       'required': METRICS, 'additionalProperties': False},
            **{key: {'type': 'string', 'minLength': 1, 'maxLength': 350} for key in TEXT_KEYS},
        },
        'required': ['scores', *TEXT_KEYS],
        'additionalProperties': False,
    }


def make_format_filter(tokenizer_data):
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.characterlevelparser import CharacterLevelParserConfig
    from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

    # Parser state belongs to one generation only; tokenizer data is reusable.
    # Initialize the alphabet before constructing minLength string states so a
    # Chinese first character is accepted, including in the top-level evidence.
    config = CharacterLevelParserConfig(alphabet=tokenizer_data.tokenizer_alphabet)
    parser = JsonSchemaParser(response_schema(), config=config)
    return build_transformers_prefix_allowed_tokens_fn(tokenizer_data, parser)


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def utf8_safe_text(text):
    """Escape unencodable surrogate code points without deleting or guessing text."""
    return text.encode('utf-8', errors='backslashreplace').decode('utf-8')


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    text = json.dumps(value, ensure_ascii=False, indent=2)
    temp.write_text(utf8_safe_text(text), encoding='utf-8')
    temp.replace(path)


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f'Duplicate JSON key: {key}')
        value[key] = item
    return value


def extract_response_object(raw):
    """Read the first complete object without editing scores or guessing missing fields."""
    start = raw.find('{')
    if start < 0:
        raise ValueError('No JSON object in model response')
    value, end = json.JSONDecoder(object_pairs_hook=_unique_object).raw_decode(raw, start)
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object')
    prefix = raw[:start].strip()
    suffix = raw[end:].strip()
    if suffix.startswith('```'):
        suffix = suffix[3:].strip()
    return value, {'json_start': start, 'json_end': end,
                   'prefix_chars': len(prefix), 'trailing_text_chars': len(suffix)}


def complete_score_object(raw):
    try:
        value, _ = extract_response_object(raw)
        return isinstance(value.get('scores'), dict)
    except (ValueError, TypeError):
        return False


def make_json_stopper(tokenizer, prompt_length):
    import torch
    from transformers import StoppingCriteria

    class CompletedJSON(StoppingCriteria):
        matched = False

        def __call__(self, input_ids, scores, **kwargs):
            if input_ids.shape[0] != 1:
                raise ValueError('This stopping rule requires batch size 1')
            count = input_ids.shape[1] - prompt_length
            # Inspect generated tokens only; never stop on the example JSON in the input.
            if count > 0 and count % 8 == 0:
                text = tokenizer.decode(input_ids[0, prompt_length:], skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
                self.matched = complete_score_object(text)
            return torch.full((1,), self.matched, device=input_ids.device, dtype=torch.bool)

    return CompletedJSON()


def parse_response(raw):
    value, location = extract_response_object(raw)
    remainder = raw[location['json_end']:]
    if '{' in remainder:
        try:
            json.JSONDecoder(object_pairs_hook=_unique_object).raw_decode(remainder, remainder.index('{'))
        except json.JSONDecodeError:
            pass
        else:
            raise ValueError('Multiple JSON objects in one answer; cannot select a rating silently')
    if set(value) != {'scores', *TEXT_KEYS}:
        raise ValueError('Response must include scores and exactly these text fields: ' + ', '.join(TEXT_KEYS))
    if set(value.get('scores', {})) != set(METRICS):
        raise ValueError('Response must include exactly 14 metrics')
    for name in METRICS:
        item = value['scores'][name]
        if not isinstance(item, dict) or set(item) != {'score', 'evidence', 'confidence'}:
            raise ValueError(f'Incomplete metric fields: {name}')
        score = item.get('score')
        if score is not None and (type(score) is not int or not 1 <= score <= 5):
            raise ValueError(f'Invalid score: {name}={score}')
        if not isinstance(item.get('evidence'), str) or not item['evidence'].strip():
            raise ValueError(f'Missing evidence: {name}')
        if item.get('confidence') not in {'low', 'medium', 'high'}:
            raise ValueError(f'Invalid confidence: {name}')
    for name in TEXT_KEYS:
        if not isinstance(value.get(name), str):
            raise ValueError(f'Missing text field: {name}')
    return value


def evidence_review_flags(value, core_a, core_b):
    """Flag missing review locations; this is not an automated truth/accuracy check."""
    flags = []
    text_fields = {f'{metric}.evidence': value['scores'][metric]['evidence'] for metric in METRICS}
    text_fields.update({key: value[key] for key in TEXT_KEYS})
    for name, text in text_fields.items():
        surrogates = sorted({ord(char) for char in text if 0xD800 <= ord(char) <= 0xDFFF})
        if surrogates:
            codes = ', '.join(f'U+{code:04X}' for code in surrogates)
            flags.append(f'{name}: Unicode surrogate {codes}; preserved as escape, review original text')
    for metric in METRICS:
        item = value['scores'][metric]
        if item['score'] is None:
            continue
        evidence = item['evidence']
        match = re.match(r'^\[\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*s\s*\]', evidence)
        if not match:
            flags.append(f'{metric}: evidence must begin with [start-end s]')
            continue
        start, end = map(float, match.groups())
        if not (core_a - 0.1 <= start < end <= core_b + 0.1):
            flags.append(f'{metric}: evidence timestamp outside the scored core')
    return flags


def export_tables(manifest, output):
    segment_rows, evidence_rows, records = [], [], {}
    for segment in manifest['segments']:
        path = output / 'segments' / (segment['segment_id'] + '.json')
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        if record.get('status') != 'ok':
            continue
        records[segment['segment_id']] = record
        base = {k: segment[k] for k in ['video_id', 'segment_id', 'core_start_s', 'core_end_s']}
        row = {**base, 'review_status': record.get('review_status', 'unreviewed'),
               'review_flags': ' | '.join(record.get('review_flags', []))}
        for metric in METRICS:
            item = record['analysis']['scores'][metric]
            row[metric] = item['score'] if item['score'] is not None else 'NA'
            evidence_rows.append({**base, **item, 'metric': metric, 'score': row[metric],
                                  'clip_start_s': segment['clip_start_s']})
        segment_rows.append(row)
    summary = []
    for video in manifest['videos']:
        entries = [s for s in manifest['segments'] if s['video_id'] == video['video_id']]
        good = [s for s in entries if s['segment_id'] in records]
        row = {'video_id': video['video_id'], 'source_filename': video['source_filename'],
               'duration_s': video['duration_s'], 'completed_segments': len(good),
               'expected_segments': len(entries), 'status': 'complete' if len(good) == len(entries) else 'partial',
               'review_status': 'unreviewed_model_draft',
               'flagged_segments': sum(bool(records[s['segment_id']].get('review_flags')) for s in good)}
        for metric in METRICS:
            pairs = [(s['core_end_s'] - s['core_start_s'], records[s['segment_id']]['analysis']['scores'][metric]['score']) for s in good]
            valid = [(d, score) for d, score in pairs if score is not None]
            covered = sum(d for d, _ in valid)
            row[metric] = round(sum(d * score for d, score in valid) / covered, 3) if covered else 'NA'
            row[metric + '_coverage'] = round(covered / video['duration_s'], 4)
        summary.append(row)
    def save(name, rows, fields):
        with (output / name).open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows({key: utf8_safe_text(value) if isinstance(value, str) else value
                              for key, value in row.items()} for row in rows)
    save('segment_scores.csv', segment_rows, ['video_id', 'segment_id', 'core_start_s', 'core_end_s', 'review_status', 'review_flags'] + METRICS)
    save('score_evidence.csv', evidence_rows, ['video_id', 'segment_id', 'core_start_s', 'core_end_s', 'clip_start_s', 'metric', 'score', 'evidence', 'confidence'])
    save('video_scores.csv', summary, list(summary[0]) if summary else ['video_id'])
    return len(records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--limit', type=int, help='Number of initial segments to process for a smoke test')
    p.add_argument('--output', type=Path, default=ROOT / 'results' / 'run_007_nonverbal')
    p.add_argument('--export-only', action='store_true')
    args = p.parse_args()
    if args.limit is not None and args.limit < 1:
        p.error('--limit must be positive')
    manifest_path = ROOT / 'data' / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.export_only:
        export_tables(manifest, args.output)
        return
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Run through Slurm; do not load the model on a login node.')
    import torch
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration, StoppingCriteriaList
    from qwen_omni_utils import process_mm_info
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU available in this job')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
    rubric = (ROOT / 'rubric.txt').read_text(encoding='utf-8')
    from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
    versions = {k: importlib.metadata.version(k) for k in ['torch', 'torchvision', 'transformers', 'qwen-omni-utils', 'av', 'librosa', 'accelerate', 'lm-format-enforcer', 'pydantic']}
    identity = {'model_id': MODEL_ID, 'revision': REVISION, 'analysis_mode': ANALYSIS_MODE,
                'measurement_target': MEASUREMENT_TARGET,
                'metric_operationalizations': METRIC_OPERATIONALIZATIONS,
                'semantic_control': {'method': 'prompt_only', 'original_audio_supplied': True,
                                     'on_screen_text_masked': False},
                'manifest_sha256': digest_bytes(manifest_path.read_bytes()),
                'rubric_sha256': digest_bytes(rubric.encode()), 'script_sha256': digest_bytes(Path(__file__).read_bytes()),
                'versions': versions, 'dtype': 'bfloat16', 'attention': 'sdpa', 'fps': 2,
                'max_pixels': 384 * 28 * 28,
                'generation': {'do_sample': False, 'max_new_tokens': MAX_NEW_TOKENS,
                               'eos_token_id': EOS_TOKEN_IDS, 'pad_token_id': 151643,
                               'format_enforcer': 'lm-format-enforcer',
                               'stop_on_complete_score_object': True, 'stop_check_interval_tokens': 8},
                'response_schema': response_schema(),
                'system_text': SYSTEM_TEXT,
                'python': platform.python_version()}
    config = args.output / 'run_config.json'
    if config.exists() and json.loads(config.read_text()) != identity:
        raise RuntimeError('Existing output belongs to a different configuration. Choose a fresh --output directory.')
    write_json(config, identity)
    print('RESULTS', args.output.resolve(), flush=True)
    print('MODE', ANALYSIS_MODE, '(original audio/video retained; semantic exclusion is instructed)', flush=True)
    records_dir = args.output / 'segments'
    records_dir.mkdir(exist_ok=True)
    model_path = ROOT / 'models' / 'Qwen2.5-Omni-7B'
    if json.loads((model_path / 'download_identity.json').read_text()) != {'model_id': MODEL_ID, 'revision': REVISION}:
        raise RuntimeError('Model download identity does not match')
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path, local_files_only=True)
    for token, expected in zip(['<|im_end|>', '<|endoftext|>'], EOS_TOKEN_IDS):
        if processor.tokenizer.convert_tokens_to_ids(token) != expected:
            raise RuntimeError(f'Tokenizer does not match the pinned model: {token}')
    print('FORMAT building constrained JSON decoder', flush=True)
    format_tokenizer_data = build_token_enforcer_tokenizer_data(processor.tokenizer)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map={'': 0},
        attn_implementation='sdpa', local_files_only=True).eval()
    selected = manifest['segments'][:args.limit] if args.limit else manifest['segments']
    failures = 0
    for segment in selected:
        dest = records_dir / (segment['segment_id'] + '.json')
        if dest.exists() and json.loads(dest.read_text()).get('status') == 'ok':
            print('SKIP', segment['segment_id'], flush=True)
            continue
        media = ROOT / 'data' / segment['path']
        if digest_bytes(media.read_bytes()) != segment['sha256']:
            raise RuntimeError(f'Media checksum mismatch: {media}')
        core_a = segment['core_start_s'] - segment['clip_start_s']
        core_b = segment['core_end_s'] - segment['clip_start_s']
        question = (rubric + f'\nFor this segment, rate only the core interval from {core_a:.2f} to '
                    f'{core_b:.2f} seconds within the video. Use the remaining footage before and '
                    'after this interval only to understand context; do not score it again. '
                    'Express evidence times in seconds within the current short clip. '
                    'Use only nonverbal vocal features and visible behavior; do not use word meaning '
                    'or any on-screen text. Return the required JSON in English.')
        messages = [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM_TEXT}]},
                    {'role': 'user', 'content': [{'type': 'video', 'video': str(media), 'fps': 2,
                                                 'max_pixels': 384 * 28 * 28}, {'type': 'text', 'text': question}]}]
        started = time.time()
        print('ANALYZE', segment['segment_id'], flush=True)
        raw = ''
        diagnostics = {}
        inputs = generated = output_ids = audios = images = videos = None
        try:
            text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            audios, images, videos, video_kwargs = process_mm_info(
                messages, use_audio_in_video=True, return_video_kwargs=True)
            if not audios or not videos:
                raise ValueError('Both audio and video must reach the model')
            inputs = processor(text=text, audio=audios, images=images, videos=videos,
                               return_tensors='pt', padding=True, use_audio_in_video=True,
                               videos_kwargs={'fps': float(video_kwargs['fps'][0]),
                                              'do_sample_frames': False, 'do_resize': False})
            inputs = inputs.to('cuda').to(torch.bfloat16)
            if not {'input_features', 'pixel_values_videos'}.issubset(inputs.keys()):
                raise ValueError('Processor did not produce audio and video features')
            torch.cuda.reset_peak_memory_stats()
            input_length = inputs['input_ids'].shape[1]
            stopper = make_json_stopper(processor.tokenizer, input_length)
            format_filter = make_format_filter(format_tokenizer_data)
            print('GENERATE', segment['segment_id'], f'input_tokens={input_length}', flush=True)
            with torch.inference_mode():
                generated = model.generate(**inputs, use_audio_in_video=True, do_sample=False,
                                           max_new_tokens=MAX_NEW_TOKENS,
                                           eos_token_id=EOS_TOKEN_IDS, pad_token_id=151643,
                                           prefix_allowed_tokens_fn=format_filter,
                                           stopping_criteria=StoppingCriteriaList([stopper]))
            output_ids = generated[0, input_length:]
            raw = processor.tokenizer.decode(output_ids, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)
            last_token = int(output_ids[-1]) if len(output_ids) else None
            diagnostics = {'input_tokens': input_length, 'generated_tokens': len(output_ids),
                           'last_token_id': last_token,
                           'stop_reason': ('json_complete' if stopper.matched else
                                           'eos' if last_token in EOS_TOKEN_IDS else
                                           'token_limit' if len(output_ids) >= MAX_NEW_TOKENS else 'other'),
                           'last_token_ids': output_ids[-16:].tolist(),
                           'audio_feature_shape': list(inputs['input_features'].shape),
                           'video_grid_thw': inputs['video_grid_thw'].tolist()}
            parsed = parse_response(raw)
            _, parsing = extract_response_object(raw)
            review_flags = evidence_review_flags(parsed, core_a, core_b)
            if raw[:parsing['json_start']].strip().lower() not in {'', '```json', '```'}:
                review_flags.append('Text precedes the JSON object; review raw response')
            if parsing['trailing_text_chars']:
                review_flags.append('Text remains after the JSON object; review raw response')
            write_json(dest, {'status': 'ok', 'analysis_mode': ANALYSIS_MODE,
                       'segment': segment, 'analysis': parsed, 'raw_response': raw,
                       'review_status': 'unreviewed_model_draft', 'review_flags': review_flags,
                       'parsing': parsing, 'generation': diagnostics,
                       'elapsed_s': round(time.time() - started, 2), 'gpu': torch.cuda.get_device_name(),
                       'peak_gpu_allocated_gib': round(torch.cuda.max_memory_allocated() / 2**30, 3),
                       'job_id': os.environ['SLURM_JOB_ID']})
            print('SCORED', segment['segment_id'], f'tokens={len(output_ids)}',
                  f'stop={diagnostics["stop_reason"]}', f'review_flags={len(review_flags)}', flush=True)
        except Exception as exc:
            failures += 1
            write_json(dest, {'status': 'error', 'analysis_mode': ANALYSIS_MODE,
                             'segment': segment, 'raw_response': raw,
                             'error': repr(exc), 'generation': diagnostics,
                             'elapsed_s': round(time.time() - started, 2), 'job_id': os.environ['SLURM_JOB_ID']})
            print('ERROR', segment['segment_id'], repr(exc), flush=True)
            if raw:
                print('RAW_PREVIEW', repr(raw[:400]), flush=True)
            # Stop on execution failures; invalid JSON can be reviewed without discarding later clips.
            if not raw:
                raise
        finally:
            inputs = generated = output_ids = audios = images = videos = None
            torch.cuda.empty_cache()
            export_tables(manifest, args.output)
    count = export_tables(manifest, args.output)
    print(f'Validated segments: {count}/{len(manifest["segments"])}; errors in this run: {failures}', flush=True)
    print('Validation above checks output format only. Scores remain unreviewed model drafts.', flush=True)
    if failures:
        sys.exit(2)


if __name__ == '__main__':
    main()
