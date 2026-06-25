# nninteractive-client

Lightweight, **torch-free** remote client for
[nnInteractive](https://github.com/MIC-DKFZ/nnInteractive).

Install this package if you only want to drive a remote `nninteractive-server`
(e.g. from a GUI, a thin client, or a host with its own torch policy such as 3D
Slicer). It pulls in only the wire dependencies (`numpy`, `httpx`, `blosc2`) —
no torch, no nnU-Net.

```bash
pip install nninteractive-client
```

```python
from nnInteractive.inference.remote import nnInteractiveRemoteInferenceSession
import numpy as np

session = nnInteractiveRemoteInferenceSession("http://gpu-box:1527", api_key="…")
session.set_image(image_4d)                       # numpy, [C, X, Y, Z]
session.set_target_buffer(np.zeros(image_4d.shape[1:], dtype=np.uint8))
session.add_point_interaction([60, 70, 30], include_interaction=True)
```

The client exposes the same public API and capability attributes as the local
`nnInteractiveInferenceSession`, so it is a drop-in replacement. See
[SERVER_CLIENT.md](https://github.com/MIC-DKFZ/nnInteractive/blob/master/SERVER_CLIENT.md)
for the full client/server guide.

## Need local inference or the server?

Those live in the full package. Installing it gives you everything (it depends
on this client):

```bash
pip install nnInteractive
```

If you try to use a full-only feature from a client-only install, you'll get a
clear error pointing you here.
