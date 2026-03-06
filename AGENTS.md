# Repository Guidelines

## Project Structure & Module Organization
`nnInteractive/` contains the main inference package.
- `nnInteractive/inference/`: session-based inference API and CVPR baseline wrapper
- `nnInteractive/interaction/`: interaction primitives (for example points)
- `nnInteractive/utils/`: crop, bbox, morphology, and OS utility helpers
- `nnInteractive/trainer/`: lightweight trainer stub used for checkpoint reconstruction

Top-level metadata and packaging live in `pyproject.toml`, `setup.py`, and `readme.md`.  
`nnInteractive/supervoxel/` is an optional subproject with its own `pyproject.toml` and CLI entry points.

## Build, Test, and Development Commands
- `pip install -e .`: editable install for core package development
- `pip install -e ".[dev]"`: install formatter/linter tooling
- `black nnInteractive/`: format Python code
- `ruff check nnInteractive/ --fix`: run lint checks and auto-fix where possible
- `codespell --skip='.git,*.pdf,*.svg'`: check spelling in source/docs
- `pre-commit run --all-files`: run all configured hooks locally

Supervoxel module (optional):
- `cd nnInteractive/supervoxel/src/sam2 && pip install -e .`
- `cd nnInteractive/supervoxel && pip install -e .`

## Coding Style & Naming Conventions
Use Python 3.10+ for the main package (`3.9+` in supervoxel).  
Follow Black formatting defaults (4-space indentation, standardized wrapping) and Ruff guidance before opening a PR.

Naming patterns in this repository:
- modules/functions/variables: `snake_case`
- classes: `PascalCase`
- constants: `UPPER_SNAKE_CASE`

## Testing Guidelines
There is currently no dedicated automated test suite in this repository.  
For changes, add a minimal reproducible validation path:
- run lint/format commands above
- run a smoke inference flow using `nnInteractiveInferenceSession` (see `readme.md` example)
- for supervoxel changes, validate CLI execution with a small sample volume

## Commit & Pull Request Guidelines
Recent history uses short, imperative commit subjects (for example `fix CPU inference`, `bbox sanity checks`, `bump to 1.1.4`). Prefer:
- one focused change per commit
- concise subject line in imperative mood
- include context in the body for behavior, compatibility, or dependency changes

For pull requests, include:
- what changed and why
- how to validate (exact commands)
- related issues/links
- notes on GPU/PyTorch assumptions when relevant (Torch `<2.9.0` for current compatibility)
