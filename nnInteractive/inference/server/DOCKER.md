# nnInteractive server — Docker images

The [`Dockerfile`](./Dockerfile) builds the nnInteractive inference server in two
flavours from a single multi-stage build. Both serve the exact same server (see
[`SERVER_CLIENT.md`](../../../SERVER_CLIENT.md)); they differ only in whether the
model weights live inside the image.

| Image | Tags | Weights | Use it when |
|-------|------|---------|-------------|
| **baked** | `latest`, `<version>`, `<version>-<model>` | downloaded into the image at build time | quick single-model deployment — just `docker pull` and run, no extra files |
| **lite** | `lite`, `<version>-lite` | mounted at runtime | you want to serve an arbitrary or newer checkpoint without rebuilding |

> The official nnInteractive checkpoint is **CC BY-NC-SA 4.0 (non-commercial)**,
> even though this code is Apache-2.0. The baked image redistributes those
> weights — only publish/use it in line with that license.

A GPU host with the NVIDIA Container Toolkit is required to *run* either image
(`--gpus all`). Building needs no GPU.

The server loads **one** model per process at startup, so a running container
serves a single model. (Resolving models dynamically — fetching newer ones on
demand — is planned but not yet supported; for now the model is either baked in
or mounted.)

## Running the baked image (model included)

```bash
docker run --gpus all -p 1527:1527 \
    -e NN_INTERACTIVE_API_KEY="$(openssl rand -hex 32)" \
    ghcr.io/mic-dkfz/nninteractive-server:latest
```

That's it — the server starts on `0.0.0.0:1527` with the baked checkpoint. Point
the [remote client](../../../SERVER_CLIENT.md) at `http://<host>:1527`.

## Running the lite image (mount your own weights)

Download a checkpoint folder (the directory that contains `fold_*/` and
`dataset.json`/`plans.json`) and mount it at `/model`:

```bash
docker run --gpus all -p 1527:1527 \
    -v /path/to/nnInteractive_v1.0:/model \
    -e NN_INTERACTIVE_API_KEY="$(openssl rand -hex 32)" \
    ghcr.io/mic-dkfz/nninteractive-server:lite
```

## Configuration

The entrypoint maps a few env vars to server flags; anything else can be passed
as extra CLI args after the image name (they override the defaults):

| Env var | Default | Meaning |
|---------|---------|---------|
| `NNINTERACTIVE_MODEL_DIR` | `/model` (lite) / baked path | model folder |
| `NNINTERACTIVE_HOST` | `0.0.0.0` | bind host |
| `NNINTERACTIVE_PORT` | `1527` | bind port |
| `NNINTERACTIVE_FOLD` | _unset_ → auto-detect | fold to load; set `0`/`all` to force a specific `fold_*` |
| `NN_INTERACTIVE_API_KEY` | _unset_ | bearer token; unauthenticated if unset |

```bash
# Example: more sessions and disable torch.compile, via passthrough flags
docker run --gpus all -p 1527:1527 ghcr.io/mic-dkfz/nninteractive-server:latest \
    --max-sessions 4 --no-torch-compile
```

`torch.compile` is on by default, so the first startup compiles and warms up the
network (slower start, faster predictions afterwards). Pass `--no-torch-compile`
to skip it.

## Building locally

```bash
# Baked (default model)
docker build -f nnInteractive/inference/server/Dockerfile --target baked \
    -t nninteractive-server:latest .

# Baked with a specific model
docker build -f nnInteractive/inference/server/Dockerfile --target baked \
    --build-arg MODEL_NAME=nnInteractive_v1.0 -t nninteractive-server:v1.0 .

# Lite
docker build -f nnInteractive/inference/server/Dockerfile --target runtime \
    -t nninteractive-server:lite .
```

The build context must be the repo root (the last `.` above) so the package is
installed from source — the image then corresponds exactly to that commit.

## Publishing (CI)

[`.github/workflows/docker-publish.yml`](../../../.github/workflows/docker-publish.yml)
builds and pushes both images to the GitHub Container Registry (GHCR) on every
`v*` tag — the same trigger as the PyPI workflow, so one tag ships both the wheel
and the image. It enforces that the tag matches the `pyproject.toml` version.

Auth uses the built-in `GITHUB_TOKEN` (the workflow grants `packages: write`), so
there are **no registry secrets to manage**. One-time setup: after the first
push, open the package under
[github.com/orgs/MIC-DKFZ/packages](https://github.com/orgs/MIC-DKFZ/packages)
and set its visibility to **Public** (and optionally link it to this repo) so
anyone can `docker pull` it.

Releasing:

```bash
# bump version in pyproject.toml to e.g. 2.4.3, then
git tag v2.4.3
git push origin v2.4.3
```

To ship a different baked checkpoint, change `MODEL_NAME` in the workflow and
release a new tag — code version and model version are decoupled.
