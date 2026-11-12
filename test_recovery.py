"""CPU-only recovery regressions using synthetic saved model responses."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import analyze
from recover_results import recover_results


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.source = root / 'source'
        self.output = root / 'recovered'
        (self.source / 'segments').mkdir(parents=True)
        self.manifest_path = root / 'manifest.json'
        self.segments = [{'video_id': 'V01', 'segment_id': name, 'core_start_s': i * 20,
                          'core_end_s': (i + 1) * 20, 'clip_start_s': i * 20,
                          'path': f'segments/{name}.mov', 'sha256': 'synthetic'}
                         for i, name in enumerate(['a', 'b'])]
        manifest = {'videos': [{'video_id': 'V01', 'source_filename': 'synthetic.mov',
                                'duration_s': 40}], 'segments': self.segments}
        self.manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
        config = {'analysis_mode': analyze.ANALYSIS_MODE, 'response_schema': analyze.response_schema(),
                  'manifest_sha256': analyze.digest_bytes(self.manifest_path.read_bytes()),
                  'script_sha256': 'original-inference-script'}
        (self.source / 'run_config.json').write_text(json.dumps(config), encoding='utf-8')
        self.analysis = {
            'scores': {key: {'score': 3, 'evidence': '[2.0-5.0 s] Synthetic gesture only.',
                             'confidence': 'low'} for key in analyze.METRICS},
            **{key: 'Synthetic fixture only.' for key in analyze.TEXT_KEYS}}
        self.analysis['scores']['guilty']['score'] = None
        good = {'status': 'ok', 'analysis_mode': analyze.ANALYSIS_MODE,
                'segment': self.segments[0], 'analysis': self.analysis,
                'raw_response': json.dumps(self.analysis), 'review_flags': [],
                'review_status': 'unreviewed_model_draft'}
        (self.source / 'segments' / 'a.json').write_text(json.dumps(good), encoding='utf-8')
        self.analysis['audio_evidence'] = 'Synthetic unfinished character: \ud83d'
        try:
            json.dumps(self.analysis, ensure_ascii=False).encode('utf-8')
        except UnicodeEncodeError as exc:
            error = repr(exc)
        self.failed = {'status': 'error', 'analysis_mode': analyze.ANALYSIS_MODE,
                       'segment': self.segments[1], 'raw_response': json.dumps(self.analysis),
                       'error': error, 'generation': {'generated_tokens': 100},
                       'elapsed_s': 1.2, 'job_id': 'synthetic-job'}
        self.failed_path = self.source / 'segments' / 'b.json'
        self.save_failure()

    def save_failure(self):
        self.failed_path.write_text(json.dumps(self.failed), encoding='utf-8')

    def run_recovery(self):
        return recover_results(self.source, self.output, self.manifest_path)

    def test_recovers_saved_response_without_changing_source_or_successes(self):
        original_bytes = {p.relative_to(self.source): p.read_bytes()
                          for p in self.source.rglob('*') if p.is_file()}
        self.assertEqual(self.run_recovery(), (['b'], 2))
        self.assertEqual(original_bytes, {p.relative_to(self.source): p.read_bytes()
                                         for p in self.source.rglob('*') if p.is_file()})
        for name in ['run_config.json', 'segments/a.json']:
            self.assertEqual((self.output / name).read_bytes(), (self.source / name).read_bytes())
        self.assertEqual((self.output / 'recovery_sources' / 'b.json').read_bytes(),
                         self.failed_path.read_bytes())
        result = json.loads((self.output / 'segments' / 'b.json').read_text(encoding='utf-8'))
        self.assertEqual(result['raw_response'], self.failed['raw_response'])
        self.assertEqual(result['analysis'], self.analysis)
        self.assertEqual(result['generation'], self.failed['generation'])
        self.assertEqual(result['job_id'], self.failed['job_id'])
        self.assertEqual(result['review_status'], 'unreviewed_model_draft')
        self.assertTrue(any('Unicode surrogate U+D83D' in flag for flag in result['review_flags']))
        self.assertFalse(result['recovery']['inference_rerun'])
        recovery = json.loads((self.output / 'recovery_config.json').read_text())
        self.assertEqual(recovery['inference_script_sha256'], 'original-inference-script')
        self.assertEqual(recovery['source_segment_sha256']['b'], analyze.digest_bytes(self.failed_path.read_bytes()))
        with (self.output / 'video_scores.csv').open(encoding='utf-8-sig') as file:
            summary = next(csv.DictReader(file))
        self.assertEqual(summary['status'], 'complete')
        self.assertEqual(summary['completed_segments'], '2')
        self.assertEqual(summary['flagged_segments'], '1')

    def test_rejects_unrelated_failure_before_creating_output(self):
        self.failed['error'] = "ValueError('Invalid score')"
        self.save_failure()
        with self.assertRaisesRegex(ValueError, 'Only Unicode'):
            self.run_recovery()
        self.assertFalse(self.output.exists())

    def test_rejects_truncated_raw_response_before_creating_output(self):
        self.failed['raw_response'] = self.failed['raw_response'][:-1]
        self.save_failure()
        with self.assertRaises(ValueError):
            self.run_recovery()
        self.assertFalse(self.output.exists())

    def test_rejects_changed_manifest_or_incomplete_download(self):
        original = self.manifest_path.read_bytes()
        self.manifest_path.write_bytes(original + b'\n')
        with self.assertRaisesRegex(ValueError, 'Manifest checksum'):
            self.run_recovery()
        self.manifest_path.write_bytes(original)
        self.failed_path.unlink()
        with self.assertRaisesRegex(ValueError, 'all manifest segment files'):
            self.run_recovery()
        self.assertFalse(self.output.exists())

    def test_rejects_existing_destination_or_nested_output(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.run_recovery()
        self.assertEqual(list(self.output.iterdir()), [])
        with self.assertRaisesRegex(ValueError, 'outside the source'):
            recover_results(self.source, self.source / 'nested', self.manifest_path)
        self.assertFalse((self.source / 'nested').exists())


if __name__ == '__main__':
    unittest.main()
