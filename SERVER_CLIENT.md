# nnInteractive Server / Client

The default `nnInteractiveInferenceSession` runs the model in the same Python
process as your GUI. If the machine running the GUI does not have a powerful
GPU, you can instead run the model on a remote box and drive it over the
network using **`nnInteractiveRemoteInferenceSession`** — a drop-in replacement
with the same public API as the local session.

```
[GUI client A]  ─┐
                 │ HTTP
[GUI client B]  ─┼────►  nninteractive-server  ──►  one shared model on GPU
                 │       (per-client sessions:           (loaded once at startup)
[GUI client C]  ─┘        image, target_buffer,
                          interactions per session)
```

The server loads the model once at startup and hosts up to `--max-sessions`
concurrent client sessions. Each client gets its own session (its own image,
target buffer, and interactions) via a lease token; the client library
handles the lease handshake transparently. Predictions are GPU-serialized
across sessions — two clients can preprocess images at the same time, but
only one prediction runs at a time.

This document covers how to start the server, point a client at it, the
concurrency / session model, and common deployment gotchas.

## Installation

There are two distributions, both providing the same `nnInteractive` import namespace:

- **GPU / server machine:** `pip install nnInteractive` — the full stack: torch, nnU-Net,
  fastapi and uvicorn, the local inference engine, and the `nninteractive-server` entry
  point. It depends on `nninteractive-client`, so it also includes the remote client.
- **GUI / client machine (lightweight, torch-free):** `pip install nninteractive-client` —
  a separate, much smaller distribution that pulls in only the wire stack (`numpy`, `httpx`,
  `blosc2`) and ships just `nnInteractive.inference.remote`. This is all
  `nnInteractiveRemoteInferenceSession` needs and avoids pulling torch / nnU-Net.

The two are layered, not mutually exclusive: the full package *depends on* the client and
ships disjoint files, so they coexist cleanly. A client-only machine can be upgraded to the
full stack at any time with `pip install nnInteractive` — no uninstall, no `--force-reinstall`.

> **Client code is identical either way.** Both distributions expose
> `from nnInteractive.inference.remote import nnInteractiveRemoteInferenceSession`, so a GUI
> written against the remote session runs unchanged whether it has the lightweight client or
> the full package installed. If a client-only install reaches for a full-only feature (local
> inference, the server, `nninteractive-server`), it gets a clear error telling it to
> `pip install nnInteractive`. (The `nninteractive-server` command is only provided by the
> full package.)

> **torch is optional for the client.** `nninteractive-client` does not depend on torch. The
> remote session works with numpy `target_buffer`s out of the box, and transparently supports
> `torch.Tensor` buffers too *if* torch happens to be importable in the host (e.g. 3D Slicer).

## Models: list, download, use by name

You don't have to hand the server a checkpoint folder — it can pull official models straight
from the manifest and you select them **by name**. Models are stored under
**`$NNINTERACTIVE_MODEL_DIR` (default `~/.nninteractive`)** and downloaded on first use.

```bash
# See which models exist and which are already downloaded
nninteractive-available-models

# Optionally pre-download one by id (the server also downloads on first use)
nninteractive-download-model nnInteractive_v1.0
```

Both commands (and the server) honor `NNINTERACTIVE_MODEL_DIR` to change where models are
stored, e.g. `export NNINTERACTIVE_MODEL_DIR=/data/nninteractive_models`.

Use `--model-dir` only for a **custom / local checkpoint folder** that isn't in the manifest
(and then also pass `--fold`).

## Starting the server

Start the server **by model name** with `--model`; it resolves the model at startup
(downloading it on first use if needed), loads it once, and subsequent client requests reuse the
loaded model. Omit `--model` to use the manifest's **default** model.

```bash
nninteractive-server \
    --model nnInteractive_v1.0 \
    --host 0.0.0.0 \
    --port 1527 \
    --device cuda:0 \
    --api-key "$(openssl rand -hex 32)"
```

(To serve a custom checkpoint folder instead, swap `--model nnInteractive_v1.0` for
`--model-dir /path/to/checkpoint_folder --fold all`.)

| Flag | Description |
|---|---|
| `--model` | Official model id from the manifest (e.g. `nnInteractive_v1.0`), downloaded on first use into `$NNINTERACTIVE_MODEL_DIR` (default `~/.nninteractive`). List ids with `nninteractive-available-models`. Mutually exclusive with `--model-dir`. **If neither `--model` nor `--model-dir` is given, the manifest's default model is used.** |
| `--model-dir` | Path to a **custom** trained model folder containing `inference_info.json` (or legacy `inference_session_class.json`), `plans.json`, `dataset.json`, and `fold_*/checkpoint_*.pth`. Use this only for checkpoints not in the manifest. Mutually exclusive with `--model`; requires `--fold`. |
| `--fold` | `0`, `1`, …, or `all`. Only relevant with `--model-dir`. If omitted, the server auto-detects when exactly one `fold_*` folder is present. |
| `--checkpoint` | Checkpoint filename inside the fold folder. Default: `checkpoint_final.pth`. |
| `--host` | Bind address. `127.0.0.1` (default) — local only; `0.0.0.0` — listen on all interfaces. |
| `--port` | TCP port. Default: `1527`. |
| `--device` | Torch device string, e.g. `cuda`, `cuda:0`, `cpu`. Default: `cuda`. |
| `--torch-n-threads` | CPU threads for torch. Default: `8`. |
| `--no-torch-compile` | Disable compiling the network with `torch.compile` (compile is **on by default**). With compile enabled the server runs a dummy warmup forward pass at startup to trigger the (slow) one-time compilation up front, so clients never see the first-prediction delay — startup just takes longer, every prediction is faster, and the cost is amortized across the long-lived process. Pass this flag to skip compilation (faster startup, or to work around a compile/backend issue). |
| `--no-autozoom` | Disable adaptive zoom-out (rarely needed; on by default). |
| `--max-sessions` | Maximum number of concurrent client sessions. Each holds its own image, target buffer, and interaction state; the network module (and therefore its weights) is shared by reference across all sessions — exactly one copy on the GPU regardless of session count. Predictions stay GPU-serialized across sessions. Default: `1` (single-tenant — same behavior as before). |
| `--idle-timeout-seconds` | Inactivity timeout in seconds after which a session is reaped and its slot freed. Refreshed only by real user actions (`set_image`, `add_*_interaction`, …) — *not* by heartbeats — so a connected-but-idle client is still reaped here. Default: `600` (10 min). |
| `--liveness-timeout-seconds` | Liveness timeout in seconds: a session is reaped if the server sees *no request at all* (not even a heartbeat) from the client for this long. This is how a crashed or disconnected client's slot is reclaimed quickly. The client heartbeats automatically at half this interval. Keep it well below `--idle-timeout-seconds`. Default: `60`. |
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

## Concurrency and sessions

The server hosts up to `--max-sessions` concurrent client sessions. Each client
holds its own session — its own image, target buffer, and interaction state —
while the network module itself (the `nn.Module` instance, its weights, and
its buffers) is shared by reference across every session. There is exactly
one network and one copy of the weights resident on the GPU regardless of
how many sessions are active. This gives multiple researchers on one GPU box
independent state without duplicating the model. Sharing is safe because
inference runs under `@torch.inference_mode()` and a global GPU lock
serializes predict-capable endpoints, so two sessions never touch the
network concurrently and nothing mutates it after startup.

### How a client gets a session

A session is claimed automatically when `nnInteractiveRemoteInferenceSession(...)`
is constructed (the client posts to `/claim` and stores a lease token, which
then rides on every subsequent request). The client also releases the session
automatically on `close()` (or context-manager exit). **Users and GUI authors
never see the lease token** — it's a private implementation detail.

```python
`with nnInteractiveRemoteInferenceSession(server_url, api_key=KEY) as session:`
    session.set_image(image)
    session.set_target_buffer(buf)
    session.add_point_interaction([60, 70, 30], include_interaction=True)
# context manager exit -> client posts /release -> server frees the slot.
```

### GPU serialization

Two clients each calling `add_point_interaction(..., run_prediction=True)` at the
same moment will see one of the two predictions wait briefly for the other to
finish. This is by design — there is only one GPU. Non-prediction calls
(`set_image`, `set_target_buffer`, `reset_interactions`, `add_*_interaction(...,
run_prediction=False)`) do not contend on this lock and run concurrently across
sessions.

If predictions feel slow under concurrent load, the answer is more GPUs (run one
`nninteractive-server` per GPU on different ports), not raising `--max-sessions`.

### Session expiry: two independent timeouts

The server reaps a session for either of two distinct reasons, each with its own
timeout:

- **Liveness** (`--liveness-timeout-seconds`, default 60 s) — the client process
  stopped responding entirely (crash, kill, network drop). The client library
  *automatically* heartbeats in the background (a daemon thread, every half the
  liveness timeout), so a healthy client never trips this. When a client dies,
  its heartbeats stop and the server frees the slot within ~one liveness timeout
  — instead of holding it for the full idle timeout. **You don't have to do
  anything to get this; it's on by default.**
- **Idle / inactivity** (`--idle-timeout-seconds`, default 600 s = 10 min) — the
  client is alive and heartbeating, but the *user* hasn't done anything. This
  timer is refreshed only by real interactions (`set_image`,
  `set_target_buffer`, `add_*_interaction`, `reset_interactions`, …), **not** by
  heartbeats. So a window left open with no clicks is still reclaimed at the idle
  timeout.

After a reap (for either reason), the next request from the client raises
`SessionExpiredError`. A session may also be reaped by a server restart.

```python
from nnInteractive.inference.remote import SessionExpiredError

try:
    session.add_point_interaction([60, 70, 30], include_interaction=True)
except SessionExpiredError:
    # The server-side session is gone. There is nothing to restore: the
    # image, target buffer, and the chain of interactions only exist on the
    # server, and they were dropped when the lease was reaped. The user has
    # to start the segmentation workflow over.
    session = nnInteractiveRemoteInferenceSession(server_url, api_key=KEY)
    session.set_image(image)
    session.set_target_buffer(buf)
    # GUI should surface: "Your session timed out. Please redo your prompts."
```

Note: `session.heartbeat()` proves liveness only — it does **not** postpone the
idle timeout. If you want users to keep a session across long idle stretches,
raise `--idle-timeout-seconds` on the server; there is no client-side way to
suppress the inactivity reap.

`session.lease_status()` is a read-only probe: it returns the remaining seconds
until the *idle* timeout without touching either clock — useful for a "your
session expires in N seconds" UI badge.

### Capacity (`--max-sessions`)

If all `--max-sessions` slots are in use when a new client tries to connect,
constructing the remote session raises `ServerAtCapacityError`. Typical
handling is "wait a moment and retry":

```python
import time
from nnInteractive.inference.remote import (
    nnInteractiveRemoteInferenceSession,
    ServerAtCapacityError,
)

for attempt in range(6):
    try:
        session = nnInteractiveRemoteInferenceSession(server_url, api_key=KEY)
        break
    except ServerAtCapacityError:
        time.sleep(10)
else:
    raise SystemExit("server has been at capacity for too long")
```

### For GUI developers

A few contract points worth respecting when wiring this into a GUI:

- **Construct the session in a worker thread.** HTTP + prediction both block;
  doing this on the UI thread freezes the app.
- **Catch `SessionExpiredError` around every interaction call.** A timed-out
  session cannot be restored — the image, target buffer, and accumulated
  prompts are all server-side state that has been freed. The GUI must claim
  a new session, call `set_image` and `set_target_buffer` again, and ask the
  user to redo their prompts. Show a clear "session timed out" message so
  the user understands why they're being asked to start over.
- **Call `session.close()` on app quit** (or use the `with` statement). The
  destructor also releases the lease, but explicit close is preferred so the
  server frees the slot immediately for other users.
- **You don't need to drive `heartbeat()` yourself.** The session auto-heartbeats
  from a background thread to keep the server from reaping it as a dead client.
  This does *not* extend the idle timeout, though — a window left idle past
  `--idle-timeout-seconds` is still reaped. Raise that flag on the server if your
  UX expects users to sit idle for long stretches.
- **Surface `ServerAtCapacityError` as "server is full, try again later".** It's
  a transient operator-level condition; the user can't fix it from the GUI.

## Authentication

Authentication is a static bearer token shared by everyone who can use the
server. The server requires it if it was started with `--api-key`; otherwise
it accepts every request without checking.

The bearer token gates **access to the server as a whole** — anyone who has
it can claim a session. The lease token (issued per client at `/claim`) is a
separate, per-session ownership mechanism handled transparently by the
client; it is not a second authentication layer and a GUI user never sees
it.

### On the server

Pick a strong, random key (anything 32+ random bytes is fine):

```bash
export NN_INTERACTIVE_API_KEY="$(openssl rand -hex 32)"
nninteractive-server --model nnInteractive_v1.0 --host 0.0.0.0 --port 1527
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

## Single-user secure setup: SSH tunnel

> **This pattern is for a single user only.** It binds the server to the
> GPU box's loopback interface, which means *the user running the SSH
> tunnel is the only one who can reach it*. If you want multiple
> researchers to share one server, skip this section and go to *Multi-user
> deployment* below.

If only you will be using the GPU box, the simplest secure setup is to bind
the server to `127.0.0.1` on the GPU box and forward a port over SSH. The
server is unreachable from any other machine; only your SSH session can
talk to it. Start the server with `--max-sessions 1` for this pattern —
nobody else can claim a session anyway.

**On the GPU box:**

```bash
nninteractive-server \
    --model nnInteractive_v1.0 \
    --host 127.0.0.1 --port 1527 \
    --max-sessions 1
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

## Multi-user deployment

For multiple users sharing one server, bind to `0.0.0.0` (or to a non-loopback
interface reachable on your network), pick a `--max-sessions` value that fits
your GPU, and set an API key:

```bash
nninteractive-server \
    --model nnInteractive_v1.0 \
    --host 0.0.0.0 --port 1527 \
    --max-sessions 4 \
    --api-key "$(openssl rand -hex 32)"
```

Distribute the API key to your users via whatever channel you'd use for any
other shared credential. Every authorized client claims its own session
automatically on construction; users do not coordinate.

**Add TLS.** The server itself does not terminate TLS. Put it behind a
reverse proxy (nginx, caddy, traefik) that adds HTTPS, especially if the
traffic leaves a trusted network. The proxy should pass through the
`Authorization`, `X-Lease-Token`, `X-Meta`, and `Content-Type` headers
unchanged and not buffer the response body (the server streams compressed
prediction diffs).

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
- **`SessionExpiredError`** — the server reaped your session, either because
  the user was inactive longer than `--idle-timeout-seconds`, because the
  client stopped heartbeating for longer than `--liveness-timeout-seconds`
  (usually a crash or network drop — note the background heartbeat keeps a
  healthy client well clear of this), or because the server was restarted. A
  timed-out session cannot be restored; the image, target buffer, and prompts
  have been freed on the server. Construct a new
  `nnInteractiveRemoteInferenceSession`, call `set_image` + `set_target_buffer`,
  and prompt the user to redo their interactions. To allow longer idle
  stretches, raise `--idle-timeout-seconds` on the server (heartbeats no longer
  postpone the idle timeout).
- **`ServerAtCapacityError` on construction** — every session slot is in
  use (`--max-sessions` reached). Wait and retry, ask the operator to
  bump `--max-sessions`, or scale out with more `nninteractive-server`
  processes on more GPUs.
- **Predictions feel slower with multiple users** — expected: predictions
  are serialized on the GPU across all sessions. Two clients each adding
  a point at the same time will see one wait briefly for the other. For
  higher throughput, run multiple `nninteractive-server` processes on
  multiple GPUs and route clients across them.

## Limitations (current version)

- Predictions are GPU-serialized within one server process: multiple clients
  can hold sessions and preprocess concurrently, but predictions run one at
  a time. For higher throughput across many concurrent users, run multiple
  `nninteractive-server` processes on different GPUs.
- Authentication is a single shared bearer token: anyone with the API key
  can claim a session. There is no per-user identity or quota.
- The checkpoint loaded at startup is fixed for the lifetime of the server
  process. Switch-by-name is planned.
- The server does not terminate TLS itself — front it with a reverse
  proxy for any multi-user deployment, or use the single-user SSH-tunnel
  pattern when only one user needs access.
- No retry/reconnect logic in the client — a network blip or
  `SessionExpiredError` raises through to the caller; the GUI is expected
  to handle this.
