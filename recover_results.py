"""Recover Unicode save failures from existing raw responses; never run inference."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import analyze


def recover_record(record):
    error = record.get('error', '')
    if (record.get('status') != 'error' or not error.startswith('UnicodeEncodeError(')
            or 'surrogates not allowed' not in error):
        raise ValueError('Only Unicode surrogate save failures can be recovered')
    raw = record['raw_response']
    parsed = analyze.parse_response(raw)
    _, parsing = analyze.extract_response_object(raw)
    segment = record['segment']
    flags = analyze.evidence_review_flags(
        parsed, segment['core_start_s'] - segment['clip_start_s'],
        segment['core_end_s'] - segment['clip_start_s'])
    if not any('Unicode surrogate' in flag for flag in flags):
        raise ValueError('Raw response does not contain the reported Unicode surrogate')
    if raw[:parsing['json_start']].strip().lower() not in {'', '```json', '```'}:
        flags.append('Text precedes the JSON object; review raw response')
    if parsing['trailing_text_chars']:
        flags.append('Text remains after the JSON object; review raw response')
    recovered = {key: value for key, value in record.items() if key != 'error'}
    recovered.update(status='ok', analysis=parsed, parsing=parsing,
                     review_status='unreviewed_model_draft', review_flags=flags,
                     recovery={'method': 'reparse_saved_raw_response', 'inference_rerun': False,
                               'source_record': f'recovery_sources/{segment["segment_id"]}.json'})
    return recovered


def recover_results(source, output, manifest_path):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents:
        raise ValueError('Recovery output must be outside the source directory')
    if output.exists():
        raise FileExistsError(f'Recovery output already exists: {output}')
    manifest_bytes = Path(manifest_path).read_bytes()
    manifest = json.loads(manifest_bytes)
    config_bytes = (source / 'run_config.json').read_bytes()
    config = json.loads(config_bytes)
    if config.get('manifest_sha256') != analyze.digest_bytes(manifest_bytes):
        raise ValueError('Manifest checksum differs from the original run')
    if (config.get('analysis_mode') != analyze.ANALYSIS_MODE
            or config.get('response_schema') != analyze.response_schema()):
        raise ValueError('Recovery requires the matching nonverbal response schema')

    originals, recovered = {}, {}
    expected = {segment['segment_id'] for segment in manifest['segments']}
    found = {path.stem for path in (source / 'segments').glob('*.json')}
    if not expected or len(expected) != len(manifest['segments']) or found != expected:
        raise ValueError('Source must contain exactly all manifest segment files; finish downloading first')
    # Validate every record before creating any output; retain the source bytes.
    for segment in manifest['segments']:
        segment_id = segment['segment_id']
        data = (source / 'segments' / f'{segment_id}.json').read_bytes()
        record = json.loads(data)
        if record.get('segment') != segment or record.get('analysis_mode') != config['analysis_mode']:
            raise ValueError(f'Segment identity differs from the original run: {segment_id}')
        originals[segment_id] = data
        if record.get('status') != 'ok':
            recovered[segment_id] = recover_record(record)
    if not recovered:
        raise ValueError('No Unicode save failures to recover')

    output.mkdir(parents=True)
    (output / 'segments').mkdir()
    (output / 'recovery_sources').mkdir()
    # run_config continues to describe inference; postprocessing has its own record.
    (output / 'run_config.json').write_bytes(config_bytes)
    for segment_id, data in originals.items():
        destination = output / 'segments' / f'{segment_id}.json'
        if segment_id in recovered:
            (output / 'recovery_sources' / destination.name).write_bytes(data)
            analyze.write_json(destination, recovered[segment_id])
        else:
            destination.write_bytes(data)
    count = analyze.export_tables(manifest, output)
    analyze.write_json(output / 'recovery_config.json', {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_directory': str(source), 'inference_rerun': False,
        'source_run_config_sha256': analyze.digest_bytes(config_bytes),
        'inference_script_sha256': config.get('script_sha256'),
        'manifest_sha256': analyze.digest_bytes(manifest_bytes),
        'postprocessing_script_sha256': {
            'analyze.py': analyze.digest_bytes(Path(analyze.__file__).read_bytes()),
            'recover_results.py': analyze.digest_bytes(Path(__file__).read_bytes()),
        },
        'source_segment_sha256': {key: analyze.digest_bytes(data) for key, data in originals.items()},
        'recovered_segments': sorted(recovered), 'validated_segments': count,
        'unicode_handling': 'Preserve surrogate code points as JSON escapes and visible CSV escapes; flag for review.',
        'review_status': 'unreviewed_model_draft',
    })
    return sorted(recovered), count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='New directory; must not exist')
    parser.add_argument('--manifest', type=Path, default=analyze.ROOT / 'data' / 'manifest.json')
    args = parser.parse_args()
    try:
        recovered, count = recover_results(args.source, args.output, args.manifest)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f'Recovery failed: {analyze.utf8_safe_text(str(exc))}\n')
    print('Recovered without inference:', ', '.join(recovered))
    print(f'Validated segments: {count}; scores remain unreviewed model drafts.')
    print('RESULTS', args.output.resolve())


if __name__ == '__main__':
    main()
