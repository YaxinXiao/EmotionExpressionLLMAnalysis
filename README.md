# EmotionExpressionLLMAnalysis

Score observable emotional expression in audiovisual debate clips with
Qwen2.5-Omni-7B. The current `run_007_nonverbal` condition produces English
evidence and 14 ratings based on vocal delivery, facial expression, posture,
gestures, and interaction timing.

## Analysis condition

`nonverbal_prompt_only` instructs the model to exclude spoken meaning, subtitles,
on-screen text, topics, and speaker identities. Original audio and video remain
available; semantic exclusion is instructed, not physically enforced.

Scores are integers from 1 to 5, or `null` when a display cannot be assessed.
`self_esteem` means displayed self-assurance in this condition; `guilty` means a
perceived apologetic/remorse-like display. These operational definitions do not
measure actual internal states. See [rubric.txt](rubric.txt) for all anchors.

## Files

- `prepare.py`: create clips and `data/manifest.json`.
- `analyze.py`: run inference, retain raw responses, validate output, and export CSVs.
- `check_format.py`: check constrained decoding with the saved tokenizer on CPU.
- `recover_results.py`: recover Unicode save failures from saved responses without inference.
- `run.sbatch`, `setup_lighthouse.sh`: Lighthouse environment and GPU submission.
- [RUN_中文.txt](RUN_中文.txt): Chinese operating instructions and historical run notes.

## Checks and execution

Run CPU regressions from the repository root; these need only Python's standard library:

```bash
python3 -m unittest -v test_analysis test_recovery
```

On Lighthouse, use `bash setup_lighthouse.sh` for initial environment and pinned
model setup. Prepare the media before inference, and verify the account and
partition in `run.sbatch`. Submit from the project root:

```bash
module load python/3.12.1
sha256sum -c run_007_files.sha256
.venv/bin/python check_format.py
sbatch run.sbatch --limit 1 --output results/nonverbal_smoke
```

Proceed only when the checks pass. For a full run, omit `--limit 1` and choose a
fresh output directory. Inference runs on allocated GPUs, not login nodes.

## Results and recovery

- `video_scores.csv`: one row per video; duration-weighted scores and coverage.
- `segment_scores.csv`: one row per segment, with review status and flags.
- `score_evidence.csv`: one row per segment and metric, with evidence and confidence.
- `segments/*.json`: raw responses, parsed ratings, and generation metadata.
- `run_config.json`: model, prompt, code, dependency, and manifest identities.

Null scores are excluded from weighted means; coverage reports the scored
duration divided by the full video duration. All outputs remain
`unreviewed_model_draft`; format validation is not content validation.

For a completed run containing a Unicode surrogate save failure, recover locally:

```bash
python3 recover_results.py --source remote_results/RUN_NAME \
  --output remote_results/RUN_NAME_recovered
```

The destination must not exist. Recovery preserves the source run, copies
successful records unchanged, archives failed records in `recovery_sources/`,
and records provenance in `recovery_config.json`. Surrogate characters are
escaped and flagged; scores, nulls, and incomplete evidence are not rewritten.

Models, media, runtime results, and backups are excluded from Git by default.
