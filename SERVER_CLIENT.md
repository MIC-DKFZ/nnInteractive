# nnInteractive Server / Client

The default `nnInteractiveInferenceSession` runs the model in the same Python
process as your GUI. If the machine running the GUI does not have a powerful
GPU, you can instead run the model on a remote box and drive it over the
network using **`nnInteractiveRemoteInferenceSession`** — a drop-in replacement
with the same public API as the local session.

```
[GUI / client machine]                       [GPU machine]
  nnInteractiveRemoteInferenceSession  --HTTP-->  nninteractive-server
  (talks to the server, holds the                 (holds the model + state,
   user's target_buffer)                           runs predictions)
```

This document covers how to start the server, point a client at it, and
common deployment gotchas.

## Installation

The server- and client-only dependencies are gated behind extras so users who
only need the local session don't pull them in.

**On the GPU machine** (server):

```bash
pip install -e ".[server]"
# or, once published: pip install "nnInteractive[server]"
```

**On the client machine** (the box running your GUI):

```bash
pip install -e ".[client]"
# or, once published: pip install "nnInteractive[client]"
```

Both extras can be combined with `pip install "nnInteractive[remote]"`.

## Starting the server

The server takes the checkpoint path at startup and loads the model once;
subsequent client requests reuse the loaded model.

```bash
nninteractive-server \
    --model-dir /path/to/checkpoint_folder \
    --fold all \
    --host 0.0.0.0 \
    --port 1527 \
    --device cuda:0 \
    --api-key "$(openssl rand -hex 32)"
```

| Flag | Description |
|---|---|
| `--model-dir` | Path to the trained model folder containing `inference_info.json` (or legacy `inference_session_class.json`), `plans.json`, `dataset.json`, and `fold_*/checkpoint_*.pth`. **Required.** |
| `--fold` | `0`, `1`, …, or `all`. If omitted, the server auto-detects when exactly one `fold_*` folder is present. |
| `--checkpoint` | Checkpoint filename inside the fold folder. Default: `checkpoint_final.pth`. |
| `--host` | Bind address. `127.0.0.1` (default) — local only; `0.0.0.0` — listen on all interfaces. |
| `--port` | TCP port. Default: `1527`. |
| `--device` | Torch device string, e.g. `cuda`, `cuda:0`, `cpu`. Default: `cuda`. |
| `--torch-n-threads` | CPU threads for torch. Default: `8`. |
| `--no-autozoom` | Disable adaptive zoom-out (rarely needed; on by default). |
| `--api-key` | Bearer token required on every request. See *Authentication* below. |
| `--verbose` | Verbose session-side logging. |
| `--log-level` | uvicorn log level (`info`, `warning`, `error`, …). Default: `info`. |

A successful startup looks like:

```
... INFO ... Loading checkpoint from /path/to/checkpoint_folder ...
session initialized
... INFO ... Checkpoint loaded; serving on http://0.0.0.0:1527
INFO:     Uvicorn running on http://0.0.0.0:1527 (Press CTRL+C to quit)
```

You can sanity-check the server from anywhere that can reach the port:

```bash
curl http://<server-host>:1527/healthz
# -> {"ok":true}
```

## Using the client

The `nnInteractiveRemoteInferenceSession` mirrors the public API of
`nnInteractiveInferenceSession` (`set_image`, `set_target_buffer`,
`add_bbox_interaction`, `add_point_interaction`, `add_scribble_interaction`,
`add_lasso_interaction`, `add_initial_seg_interaction`, `reset_interactions`,
`set_do_autozoom`) and exposes the same capability attributes
(`supported_interactions`, `channel_mapping`, `num_interaction_channels`,
`supports_initial_label`, `supports_zero_shot_label_refinement`,
`preferred_scribble_thickness`, `interaction_decay`, `original_image_shape`,
`do_autozoom`).

Minimal usage:

```python
from nnInteractive.inference.remote import nnInteractiveRemoteInferenceSession
import numpy as np

session = nnInteractiveRemoteInferenceSession(
    server_url="http://gpu-box.lab:1527",
    api_key="…",          # optional; see Authentication
)

session.set_image(image_4d)                       # numpy, [C, X, Y, Z]
target_buffer = np.zeros(image_4d.shape[1:], dtype=np.uint8)
session.set_target_buffer(target_buffer)

session.add_bbox_interaction([[40, 80], [50, 90], [30, 31]],
                             include_interaction=True)
# target_buffer is now updated in place with the predicted region.

session.add_point_interaction([60, 70, 30], include_interaction=True)
# … and so on. Same calls as the local session.
```

`target_buffer` is mutated in place exactly the same way as with the local
session. Under the hood, the server returns just the bbox region it touched
(blosc2-compressed), and the client writes that into your buffer — typical
binary masks compress to a tiny fraction of their raw size, so this stays
fast even on slow links.

### Timeouts

The client uses per-phase timeouts so "server unreachable" is reported
quickly while real predictions still get the time they need:

| Constructor kwarg | Default | Covers | On expiry |
|---|---|---|---|
| `connect_timeout` | 10 s | TCP / TLS handshake | `httpx.ConnectTimeout` |
| `read_timeout` | 60 s | server thinking time per call (predictions observed at 100 ms – ~10 s) | `httpx.ReadTimeout` |
| `write_timeout` | 120 s | uploading the request body (mostly `set_image`) | `httpx.WriteTimeout` |
| `pool_timeout` | 10 s | acquiring a connection from the pool | `httpx.PoolTimeout` |

All four are subclasses of `httpx.TimeoutException`, which itself is a
subclass of `httpx.HTTPError` — catch `HTTPError` for a generic "something
went wrong with the server" and `TimeoutException` for "the server didn't
respond in time."

```python
import httpx
try:
    session.add_point_interaction([60, 70, 30], include_interaction=True)
except httpx.ConnectTimeout:
    # Server is unreachable. Likely down, wrong host/port, or a firewall.
    ...
except httpx.ReadTimeout:
    # Server accepted the request but didn't finish in read_timeout seconds.
    # Either the prediction is unusually slow or the server is stuck.
    ...
except httpx.HTTPStatusError as e:
    # Server responded with 4xx/5xx. e.response.status_code / e.response.text
    ...
```

### Probing reachability — `session.ping()`

For a "Test connection" button in a GUI, the client exposes:

```python
ok: bool = session.ping(timeout=5.0)   # GET /healthz with a tight timeout
```

It returns `True` if the server answered 200 and `False` on any HTTP /
network error (timeout, refused connection, wrong auth, proxy
interception). Non-raising on purpose so UI code can just check the bool.

### One-line swap from local to remote

```python
# Local
session = nnInteractiveInferenceSession(device=torch.device("cuda"))
session.initialize_from_trained_model_folder("/path/to/checkpoint", use_fold="all")

# Remote — same API from here on
session = nnInteractiveRemoteInferenceSession("http://gpu-box:1527", api_key=KEY)
```

Note: on the remote session, `initialize_from_trained_model_folder()` is a
no-op (with a warning). The server already loaded the checkpoint at startup.
Switching checkpoints at runtime is on the roadmap.

## Authentication

Authentication is a static bearer token. The server requires it if it was
started with `--api-key`; otherwise it accepts every request without
checking.

### On the server

Pick a strong, random key (anything 32+ random bytes is fine):

```bash
export NN_INTERACTIVE_API_KEY="$(openssl rand -hex 32)"
nninteractive-server --model-dir /path/to/checkpoint --fold all --host 0.0.0.0 --port 1527
# (alternatively: pass --api-key "$KEY" on the command line)
```

The server reads `--api-key` first, then falls back to the
`NN_INTERACTIVE_API_KEY` environment variable. If neither is set, the
server logs a warning at startup and accepts unauthenticated requests.

### On the client

```python
session = nnInteractiveRemoteInferenceSession(
    server_url="http://gpu-box:1527",
    api_key="…",
)
```

If `api_key=` is omitted, the client falls back to the
`NN_INTERACTIVE_API_KEY` environment variable. If the server requires a key
and the client didn't pass one (or passed the wrong one), the very first
request — the capabilities fetch inside `__init__` — raises an HTTP 401,
so you find out at session construction time, not later in a prediction.

Rotation: change the key, restart the server, update the client. There is no
login flow.

## Recommended: SSH tunnel (avoids exposing the server)

For a single user, the simplest secure setup is to bind the server to
`127.0.0.1` on the GPU box and forward a port over SSH. The server is
unreachable from any other machine; only an authenticated SSH session can
talk to it.

**On the GPU box:**

```bash
nninteractive-server \
    --model-dir /path/to/checkpoint --fold all \
    --host 127.0.0.1 --port 1527
```

**On the client box:**

```bash
ssh -N -L 1527:127.0.0.1:1527 you@gpu-box.lab
# Leave this running in a terminal. Now http://127.0.0.1:1527 on the
# client points at the server's 127.0.0.1:1527.
```

```python
session = nnInteractiveRemoteInferenceSession("http://127.0.0.1:1527")
# No api_key needed — the server is only reachable through your SSH session.
```

For laptops / unstable links, `autossh` keeps the tunnel up:

```bash
autossh -M 0 -o "ServerAliveInterval=30" -o "ServerAliveCountMax=3" \
        -N -L 1527:127.0.0.1:1527 you@gpu-box.lab
```

If you do expose the server on `0.0.0.0`, **set an API key** and ideally
front the server with a reverse proxy that adds TLS (nginx, caddy, traefik).
The server itself does not terminate TLS.

## Proxy gotcha

If your client machine has `HTTP_PROXY` / `HTTPS_PROXY` / `ALL_PROXY` set —
common on corporate networks — `httpx` (which the client uses) will route
*every* request through the proxy by default, **including localhost ones**.
Symptoms are 403 responses with HTML error pages from the proxy instead of
JSON from the server, even with the correct API key.

Fix: add the server's host (or `127.0.0.1`/`localhost` for an SSH tunnel) to
`NO_PROXY`:

```bash
export NO_PROXY="127.0.0.1,localhost,gpu-box.lab"
export no_proxy="$NO_PROXY"   # both casings — some tools only read one
```

Then run your client program in the same shell. To make this permanent, add
the lines to your shell rc file or to the launcher script that starts the
GUI.

## Troubleshooting

- **`httpx.HTTPStatusError: 401 Unauthorized` on session construction** — the
  server was started with `--api-key` but the client didn't pass it (or
  passed the wrong one). Set `api_key=` or `NN_INTERACTIVE_API_KEY`.
- **HTML error pages instead of JSON** — almost always an HTTP proxy
  intercepting the request. See *Proxy gotcha*.
- **`ConnectionRefusedError` / `httpx.ConnectError`** — server isn't
  running, port is wrong, or a firewall is blocking it. Check
  `curl http://<host>:<port>/healthz` from the client machine, or call
  `session.ping()` from your GUI's "Test connection" path.
- **`httpx.ConnectTimeout` (after ~10 s)** — TCP/TLS handshake didn't
  complete. The host is reachable but isn't listening, or a firewall is
  silently dropping packets. Tune via `connect_timeout=` on the session
  constructor.
- **`httpx.ReadTimeout` (after ~60 s)** — server accepted the request but
  didn't finish in time. Either the prediction is unusually slow on that
  hardware/volume, or the server is wedged. Tune via `read_timeout=` if
  your workload legitimately needs more.
- **`RuntimeWarning: nnInteractiveRemoteInferenceSession ignores
  initialize_from_trained_model_folder()`** — expected. The server picked
  the checkpoint at startup; this method is a no-op on the remote session.
- **Predictions seem to hang the GUI** — every `add_*_interaction(..., 
  run_prediction=True)` call blocks until the server finishes. Run the
  remote session from a worker thread in the GUI, exactly as you would for
  a slow local prediction.

## Limitations (current version)

- One server process serves one in-flight session at a time (calls are
  serialized by a lock). No multi-tenancy yet.
- The checkpoint loaded at startup is fixed for the lifetime of the server
  process. Switch-by-name is planned.
- The server does not terminate TLS itself — put it behind a reverse proxy
  or run over an SSH tunnel for any non-trivial deployment.
- No retry/reconnect logic in the client — a network blip raises through to
  the caller; the GUI is expected to handle this.
