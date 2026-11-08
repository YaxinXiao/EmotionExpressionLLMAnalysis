"""CPU-only regressions for malformed output and research table accounting."""
import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

import analyze


def example():
    return {'scores': {key: {'score': 3, 'evidence': '[2.0-5.0 s] 测试依据：引语包含 } 字符',
                            'confidence': 'medium'} for key in analyze.METRICS},
            **{key: '测试证据，不是真实视频评分' for key in analyze.TEXT_KEYS}}


class ResponseTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
