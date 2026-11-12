"""CPU-only regressions for malformed output and research table accounting."""
import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

import analyze
from check_format import walk_allowed_text


def example():
    return {'scores': {key: {'score': 3, 'evidence': '[2.0-5.0 s] 测试依据：引语包含 } 字符',
                            'confidence': 'medium'} for key in analyze.METRICS},
            **{key: '测试证据，不是真实视频评分' for key in analyze.TEXT_KEYS}}


class ResponseTests(unittest.TestCase):
    def test_nonverbal_response_round_trips_without_language_evidence(self):
        value = example()
        for item in value['scores'].values():
            item['evidence'] = '[2.0-5.0 s] Synthetic fixture: brief gesture; not a video rating.'
        for key in analyze.TEXT_KEYS:
            value[key] = 'Synthetic nonverbal fixture only; no video analyzed.'
        self.assertEqual(set(value), {'scores', 'visual_evidence', 'audio_evidence',
                                      'interaction_pattern', 'limitations'})
        self.assertEqual(analyze.parse_response(json.dumps(value)), value)

    def test_nonverbal_response_rejects_semantic_evidence_field(self):
        value = example()
        value['language_evidence'] = 'A transcript belongs to the old analysis condition.'
        with self.assertRaisesRegex(ValueError, 'exactly these text fields'):
            analyze.parse_response(json.dumps(value))

    def test_nonverbal_response_requires_both_audio_and_visual_evidence(self):
        for key in ['audio_evidence', 'visual_evidence']:
            with self.subTest(key=key):
                value = example()
                del value[key]
                with self.assertRaisesRegex(ValueError, 'exactly these text fields'):
                    analyze.parse_response(json.dumps(value))

    def test_fenced_json_followed_by_reported_repetition_preserves_scores(self):
        value = example()
        value['scores']['self_esteem']['score'] = None
        raw = '```json\n' + json.dumps(value, ensure_ascii=False) + '\n```\n#Landscape\n#Mountain\n#Ocean\n' + 'Human\n' * 1000
        self.assertEqual(analyze.parse_response(raw), value)
        _, metadata = analyze.extract_response_object(raw)
        self.assertGreater(metadata['trailing_text_chars'], 5000)

    def test_does_not_accept_truncated_parent_even_with_complete_nested_objects(self):
        raw = '```json\n' + json.dumps(example())[:-1]
        self.assertFalse(analyze.complete_score_object(raw))
        with self.assertRaises(ValueError):
            analyze.parse_response(raw)

    def test_braces_inside_strings_do_not_stop_early(self):
        raw = json.dumps(example())
        for end in range(len(raw)):
            self.assertFalse(analyze.complete_score_object(raw[:end]))
        self.assertTrue(analyze.complete_score_object(raw))

    def test_duplicate_scores_are_rejected_not_overwritten(self):
        raw = json.dumps(example()).replace('"score": 3', '"score": 2, "score": 4', 1)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            analyze.parse_response(raw)

    def test_out_of_scale_and_noninteger_scores_rejected(self):
        for score in [0, 6, True, '3', 3.5]:
            with self.subTest(score=score):
                value = example()
                value['scores']['happy']['score'] = score
                with self.assertRaises(ValueError):
                    analyze.parse_response(json.dumps(value))

    def test_missing_metric_is_not_invented(self):
        value = example()
        del value['scores']['guilty']
        with self.assertRaises(ValueError):
            analyze.parse_response(json.dumps(value))

    def test_flat_scores_from_run003_are_not_silently_repaired(self):
        value = example()
        value['scores'] = {key: 4 if key in {'valence', 'arousal', 'proud'} else None
                           for key in analyze.METRICS}
        with self.assertRaisesRegex(ValueError, 'Incomplete metric fields: valence'):
            analyze.parse_response(json.dumps(value))

    def test_conflicting_complete_objects_rejected(self):
        value = example()
        other = copy.deepcopy(value)
        other['scores']['angry']['score'] = 5
        with self.assertRaisesRegex(ValueError, 'Multiple JSON'):
            analyze.parse_response(json.dumps(value) + '\n' + json.dumps(other))

    def test_missing_timestamp_flagged_but_null_does_not_need_one(self):
        value = example()
        self.assertEqual(analyze.evidence_review_flags(value, 0, 20), [])
        value['scores']['self_esteem']['evidence'] = '语气坚定，因此自尊高'
        self.assertTrue(any('self_esteem' in flag for flag in analyze.evidence_review_flags(value, 0, 20)))
        value['scores']['self_esteem']['score'] = None
        self.assertEqual(analyze.evidence_review_flags(value, 0, 20), [])

    def test_timestamp_outside_core_flagged(self):
        value = example()
        value['scores']['angry']['evidence'] = '[0-1 s] 核心区间外的行为'
        self.assertTrue(any('angry' in flag for flag in analyze.evidence_review_flags(value, 2, 22)))

    def test_json_save_preserves_surrogates_raw_escapes_and_normal_unicode(self):
        value = {'audio_evidence': 'Incomplete character: \ud83d; low: \ude00',
                 'raw_response': r'{"audio_evidence": "Incomplete character: \ud83d"}',
                 'ordinary_text': '中文 café 😀', 'score': None}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'record.json'
            analyze.write_json(path, value)
            text = path.read_bytes().decode('utf-8', errors='strict')
            self.assertIn('中文 café 😀', text)
            self.assertEqual(json.loads(text), value)
            self.assertFalse(path.with_suffix('.json.tmp').exists())

    def test_unicode_flag_covers_null_evidence_and_top_level_text(self):
        value = example()
        value['audio_evidence'] = 'Synthetic incomplete character: \ud83d'
        value['scores']['guilty'].update(score=None, evidence='Synthetic low surrogate: \ude00')
        flags = analyze.evidence_review_flags(value, 0, 20)
        self.assertEqual(len(flags), 2)
        self.assertTrue(any('audio_evidence: Unicode surrogate U+D83D' in flag for flag in flags))
        self.assertTrue(any('guilty.evidence: Unicode surrogate U+DE00' in flag for flag in flags))
        value['audio_evidence'] = '中文 café 😀'
        value['scores']['guilty']['evidence'] = 'No observable expression in this synthetic fixture.'
        self.assertEqual(analyze.evidence_review_flags(value, 0, 20), [])

    def test_csv_export_escapes_surrogates_without_changing_scores(self):
        value = example()
        evidence = '[2.0-5.0 s] Synthetic incomplete character: \ud83d'
        value['scores']['happy']['evidence'] = evidence
        value['scores']['guilty']['score'] = None
        segment = {'video_id': 'V01', 'segment_id': 'a', 'core_start_s': 0,
                   'core_end_s': 20, 'clip_start_s': 0}
        manifest = {'videos': [{'video_id': 'V01', 'source_filename': 'fixture.mov',
                                'duration_s': 20}], 'segments': [segment]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'segments').mkdir()
            analyze.write_json(root / 'segments' / 'a.json', {
                'status': 'ok', 'analysis': value,
                'review_flags': analyze.evidence_review_flags(value, 0, 20)})
            self.assertEqual(analyze.export_tables(manifest, root), 1)
            with (root / 'score_evidence.csv').open(encoding='utf-8-sig') as file:
                rows = {row['metric']: row for row in csv.DictReader(file)}
            self.assertEqual(rows['happy']['evidence'], evidence.replace('\ud83d', r'\ud83d'))
            self.assertEqual(rows['happy']['score'], '3')
            self.assertEqual(rows['guilty']['score'], 'NA')
            with (root / 'video_scores.csv').open(encoding='utf-8-sig') as file:
                summary = next(csv.DictReader(file))
            self.assertEqual(summary['happy'], '3.0')
            self.assertEqual(summary['guilty_coverage'], '0.0')
            self.assertEqual(summary['flagged_segments'], '1')

    def test_duration_weighting_na_coverage_and_partial_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'segments').mkdir()
            segments = [{'video_id': 'V01', 'segment_id': 'a', 'core_start_s': 0, 'core_end_s': 20, 'clip_start_s': 0},
                        {'video_id': 'V01', 'segment_id': 'b', 'core_start_s': 20, 'core_end_s': 30, 'clip_start_s': 18}]
            manifest = {'videos': [{'video_id': 'V01', 'source_filename': 'fixture.mov', 'duration_s': 30}], 'segments': segments}
            for i, segment in enumerate(segments):
                value = copy.deepcopy(example())
                value['scores']['angry']['score'] = [1, 5][i]
                value['scores']['guilty']['score'] = [2, None][i]
                (root / 'segments' / (segment['segment_id'] + '.json')).write_text(json.dumps(
                    {'status': 'ok', 'analysis': value, 'review_flags': ['fixture flag'] if i else []}))
            analyze.export_tables(manifest, root)
            with (root / 'video_scores.csv').open(encoding='utf-8-sig') as f:
                row = next(csv.DictReader(f))
            self.assertAlmostEqual(float(row['angry']), 7 / 3, places=3)
            self.assertEqual(float(row['guilty']), 2)
            self.assertAlmostEqual(float(row['guilty_coverage']), 2 / 3, places=4)
            self.assertEqual(row['review_status'], 'unreviewed_model_draft')
            self.assertEqual(row['flagged_segments'], '1')
            (root / 'segments' / 'b.json').unlink()
            analyze.export_tables(manifest, root)
            with (root / 'video_scores.csv').open(encoding='utf-8-sig') as f:
                self.assertEqual(next(csv.DictReader(f))['status'], 'partial')


class FixtureTokenizer:
    """Small tokenizer reproducing the reported cross-boundary token."""
    composite_text = '.",'
    composite_id = 1000000
    eos_token_id = 1000001

    def encode(self, text, add_special_tokens=False):
        result = []
        while text:
            if text.startswith(self.composite_text):
                result.append(self.composite_id)
                text = text[len(self.composite_text):]
            else:
                result.append(ord(text[0]))
                text = text[1:]
        return result

    def decode(self, tokens, clean_up_tokenization_spaces=False):
        return ''.join(self.composite_text if token == self.composite_id else chr(token)
                       for token in tokens)


class FormatWalkTests(unittest.TestCase):
    def mask_for(self, tokenizer, target, allow_composite=False, forbidden_character=None):
        vocabulary = {ord(character) for character in target} | {tokenizer.composite_id}

        def allowed(ids):
            prefix = tokenizer.decode(ids[1:])
            if prefix == target:
                return [tokenizer.eos_token_id]
            return [token for token in vocabulary
                    if (allow_composite or token != tokenizer.composite_id)
                    and tokenizer.decode([token]) != forbidden_character
                    and target.startswith(prefix + tokenizer.decode([token]))]
        return allowed

    def test_split_composite_token_preserves_english_unicode_and_escapes(self):
        tokenizer = FixtureTokenizer()
        for evidence in ['English fixture.', 'Unicode fixture: 中文, café.', 'A quote " and brace }.']:
            with self.subTest(evidence=evidence):
                raw = json.dumps({'evidence': evidence, 'confidence': 'low'}, ensure_ascii=False)
                allowed, splits = walk_allowed_text(tokenizer, [ord('P')], raw,
                                                     self.mask_for(tokenizer, raw))
                self.assertIn(tokenizer.eos_token_id, allowed)
                self.assertEqual(splits, 1)

    def test_supported_composite_token_keeps_original_encoding(self):
        tokenizer = FixtureTokenizer()
        raw = json.dumps({'evidence': 'English fixture.', 'confidence': 'low'})
        allowed, splits = walk_allowed_text(tokenizer, [ord('P')], raw,
            self.mask_for(tokenizer, raw, allow_composite=True))
        self.assertIn(tokenizer.eos_token_id, allowed)
        self.assertEqual(splits, 0)

    def test_split_does_not_bypass_a_forbidden_character(self):
        tokenizer = FixtureTokenizer()
        raw = json.dumps({'evidence': 'English fixture.', 'confidence': 'low'})
        with self.assertRaisesRegex(RuntimeError, 'no allowed token path'):
            walk_allowed_text(tokenizer, [ord('P')], raw,
                self.mask_for(tokenizer, raw, forbidden_character='.'))

    def test_lossy_retokenization_is_rejected(self):
        class LossyTokenizer(FixtureTokenizer):
            def encode(self, text, add_special_tokens=False):
                if text == '.':
                    return [ord('!')]
                return super().encode(text, add_special_tokens=add_special_tokens)

        tokenizer = LossyTokenizer()
        raw = json.dumps({'evidence': 'English fixture.', 'confidence': 'low'})
        with self.assertRaisesRegex(RuntimeError, 'Cannot losslessly split'):
            walk_allowed_text(tokenizer, [ord('P')], raw, self.mask_for(tokenizer, raw))


if __name__ == '__main__':
    unittest.main()
