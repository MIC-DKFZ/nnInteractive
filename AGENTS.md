# Repository Guidelines

## Project Structure & Module Organization
`nnInteractive/` is the main Python package for promptable 3D medical-image segmentation.
- `nnInteractive/inference/`: local sessions, remote client support, server code, and the CVPR 2025 baseline wrapper
- `nnInteractive/interaction/`: interaction primitives such as point prompts
- `nnInteractive/utils/`: bbox, crop, morphology, checkpoint, OS, and inference helpers
- `nnInteractive/trainer/`: lightweight checkpoint reconstruction stub
- `nnInteractive/supervoxel/`: optional SAM/SAM2 supervoxel subproject with separate packaging and CLIs

Top-level docs include `readme.md`, `SERVER_CLIENT.md`, and `API_CHANGES_v2.md`. Documentation images live in `imgs/`.

## Build, Test, and Development Commands
- `pip install -e .`: install the full local inference and server stack in editable mode.
- `pip install -e ".[dev]"`: add Black, Ruff, and pre-commit tooling.
- `pip install --no-deps nnInteractive && pip install numpy httpx blosc2`: install the lightweight torch-free remote client stack.
- `nninteractive-server --model-dir /path/to/checkpoint --fold all --host 0.0.0.0 --port 1527 --device cuda:0`: run the inference server.
- `black nnInteractive/`: format Python code; the pre-commit hook uses line length 120.
- `ruff check nnInteractive/ --fix`: lint and apply safe fixes.
- `codespell --skip='.git,*.pdf,*.svg'`: check spelling in source and docs.
- `pre-commit run --all-files`: run configured hooks before submitting changes.

For supervoxel work, install both layers: `cd nnInteractive/supervoxel/src/sam2 && pip install -e .`, then `cd ../.. && pip install -e .`.

## Coding Style & Naming Conventions
Use Python 3.10+ for the main package and Python 3.9+ in `nnInteractive/supervoxel/`. Follow Black formatting with 4-space indentation and 120-character lines. Prefer typed, explicit public inference/session APIs. Use `snake_case` for modules, functions, and variables; `PascalCase` for classes; and `UPPER_SNAKE_CASE` for constants.

## Testing Guidelines
There is no dedicated automated test suite. Validate changes with the narrowest reproducible smoke path:
- local inference through `nnInteractiveInferenceSession` using the `readme.md` flow
- remote inference by starting `nninteractive-server`, checking `/healthz`, and exercising `nnInteractiveRemoteInferenceSession`
- supervoxel CLI execution on a small sample volume when touching `nnInteractive/supervoxel/`

Also run formatting, linting, and spelling checks for touched areas.

## Commit & Pull Request Guidelines
Recent history uses short, imperative subjects such as `disable compile on cpu`, `reduce threshold for blosc2`, and `bump version`. Keep commits focused and include body context for behavior changes, compatibility breaks, or dependency assumptions.

Pull requests should state what changed, why it changed, exact validation commands, related issues or integrations, and any GPU, PyTorch, checkpoint, server/client, or API-compatibility assumptions. Do not commit local model checkpoints, API keys, or generated large outputs.
