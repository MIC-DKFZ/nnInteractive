# nnInteractive server — Docker images

Two images from one multi-stage [`Dockerfile`](./Dockerfile), same server, differ only in where the weights live:

| Image | Tags | Weights |
|-------|------|---------|
| **baked** | `latest`, `<version>`, `<version>-<model>` | inside the image |
| **lite** | `lite`, `<version>-lite` | mounted/fetched at runtime |

> The official checkpoint is **CC BY-NC-SA 4.0 (non-commercial)**; the code is Apache-2.0. The baked image redistributes the weights — use it accordingly.

Running needs a GPU with the NVIDIA Container Toolkit (`--gpus all`). Building does not.

## Run — baked

```bash
docker run --gpus all -p 1527:1527 \
    -e NN_INTERACTIVE_API_KEY="$(openssl rand -hex 32)" \
    ghcr.io/mic-dkfz/nninteractive-server:latest
```

## Run — lite, mounted checkpoint

Mount a checkpoint folder (contains `fold_*/`, `dataset.json`, `plans.json`) and set the fold:

```bash
docker run --gpus all -p 1527:1527 \
    -v /path/to/nnInteractive_v1.0:/model \
    -e NNINTERACTIVE_FOLD=0 \
    -e NN_INTERACTIVE_API_KEY="$(openssl rand -hex 32)" \
    ghcr.io/mic-dkfz/nninteractive-server:lite
```

## Run — lite, fetch by id

```bash
docker run --gpus all -p 1527:1527 \
    -e NNINTERACTIVE_MODEL=nnInteractive_v1.0 \
    -e NN_INTERACTIVE_API_KEY="$(openssl rand -hex 32)" \
    ghcr.io/mic-dkfz/nninteractive-server:lite
```

Server listens on `0.0.0.0:1527`. Point the [remote client](../../../SERVER_CLIENT.md) at `http://<host>:1527`.

## Configuration

Env vars map to server flags. Extra CLI args after the image name override them.

| Env var | Default | Meaning |
|---------|---------|---------|
| `NNINTERACTIVE_MODEL` | _unset_ (lite) / baked id | official model id; fetched/resolved via manifest, fold optional |
| `NNINTERACTIVE_MODEL_DIR` | `/model` (lite) / `/opt/nninteractive` (baked) | model root (with `NNINTERACTIVE_MODEL`) or mounted checkpoint folder |
| `NNINTERACTIVE_HOST` | `0.0.0.0` | bind host |
| `NNINTERACTIVE_PORT` | `1527` | bind port |
| `NNINTERACTIVE_FOLD` | _unset_ | fold to load. Required for a mounted checkpoint folder; optional with `NNINTERACTIVE_MODEL`. `0`/`all` |
| `NN_INTERACTIVE_API_KEY` | _unset_ | bearer token; unauthenticated if unset |

```bash
docker run --gpus all -p 1527:1527 ghcr.io/mic-dkfz/nninteractive-server:latest \
    --max-sessions 4 --no-torch-compile
```

`torch.compile` is on by default: slower first start, faster predictions. `--no-torch-compile` to skip.

## Build

```bash
# baked (default model)
docker build -f nnInteractive/inference/server/Dockerfile --target baked -t nninteractive-server:latest .

# baked, specific model
docker build -f nnInteractive/inference/server/Dockerfile --target baked \
    --build-arg MODEL_NAME=nnInteractive_v1.0 -t nninteractive-server:v1.0 .

# lite
docker build -f nnInteractive/inference/server/Dockerfile --target runtime -t nninteractive-server:lite .
```

Build context is the repo root (the trailing `.`) — installs from source, so the image matches the commit.

## Publish (CI)

[`.github/workflows/docker-publish.yml`](../../../.github/workflows/docker-publish.yml) builds and pushes both images to GHCR on every `v*` tag (must match the `pyproject.toml` version). Auth uses the built-in `GITHUB_TOKEN` — no secrets to manage. After the first push, set the package to **Public** at [github.com/orgs/MIC-DKFZ/packages](https://github.com/orgs/MIC-DKFZ/packages).

```bash
# bump version in pyproject.toml, then
git tag v2.5.1
git push origin v2.5.1
```

Change `MODEL_NAME` in the workflow to ship a different baked checkpoint.
