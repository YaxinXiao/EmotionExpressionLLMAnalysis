# Repository Guidelines

## Project Structure & Module Organization

This Python project scores observable emotion in audiovisual debate clips using Qwen2.5-Omni-7B.

- `prepare.py` creates clips and `data/manifest.json`; media lives in `data/segments/` with names such as `V01_S001.mov`.
- `analyze.py` runs inference, validates responses, and exports CSV tables. `rubric.txt` defines scoring instructions.
- `test_analysis.py` contains CPU regression tests; `check_format.py` checks constrained decoding with the saved tokenizer.
- `setup_lighthouse.sh`, `download_model.py`, and `run.sbatch` support Lighthouse setup and Slurm execution.
- `models/` and `results/` are runtime directories. `RUN_中文.txt` contains operational notes, including historical commands; check current script defaults before reuse. Preserve archived files in `backups/`.

## Build, Test, and Development Commands

Run commands from the project root. There is no separate build step.

- `python3 -m unittest -v test_analysis`: run CPU regressions without downloading model weights.
- `bash setup_lighthouse.sh`: on a Lighthouse login node, create `.venv`, install pinned dependencies, and download the fixed model revision. Requires network access and the Python 3.12.1 module.
- `.venv/bin/python check_format.py`: validate JSON constraints using the downloaded tokenizer, without loading weights.
- `python3 prepare.py --source /path/to/videos --output data_new`: prepare media in a new directory; requires `imageio-ffmpeg`. The output directory must not exist.
- `sbatch run.sbatch --limit 1 --output results/run_example`: submit a GPU smoke test. Review its evidence before omitting `--limit 1` for the full run. Verify Slurm account and partition settings first.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` functions and variables, and uppercase constants. Follow existing single-quoted strings and `pathlib.Path` usage. Keep heavyweight inference imports inside runtime functions so CPU tests remain lightweight. No formatter or linter is configured.

## Testing Guidelines

Use standard-library `unittest`, with `test_*` methods in `test_analysis.py`. Add regressions for parser, schema, timestamp, or aggregation changes using synthetic fixtures and temporary directories. No coverage threshold is configured. Run tokenizer checks when changing decoding constraints.

## Commit & Pull Request Guidelines

This checkout has no Git metadata, so commit conventions cannot be inferred. Use concise imperative subjects, such as `Fix missing-score coverage accounting`. PRs should describe behavior changes, validation performed, related issues, and effects on scoring or reproducibility.

## Research Integrity & Reproducibility

Preserve raw responses, hashes, and null scores; never invent missing evidence. Results remain `unreviewed_model_draft` until reviewed. Use a fresh output directory when code, prompts, model, or environment changes; preserve pinned versions and existing results.
