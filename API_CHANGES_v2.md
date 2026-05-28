# `nnInteractiveInferenceSession` — Public API Changes (master → v2)

This document summarizes how the public API of `nnInteractiveInferenceSession`
differs between `master` and the `project/v2_release` branch. Anything not
listed below kept the same signature and call semantics.

## Constructor — `__init__`

```diff
- __init__(device, use_torch_compile=False, verbose=False, torch_n_threads=8,
-          do_autozoom=True, use_pinned_memory=True)
+ __init__(device, use_torch_compile=False, verbose=False, torch_n_threads=8,
+          do_autozoom=True)
```

- `use_pinned_memory` is gone. The interaction tensor is no longer a
  pinned-memory `torch.Tensor` — it is now an in-memory blosc2-compressed
  array. Any caller passing `use_pinned_memory=...` must drop the kwarg.

## `set_do_autozoom`

```diff
- set_do_autozoom(do_propagation: bool, max_num_patches: Optional[int] = None)
+ set_do_autozoom(do_autozoom: bool)
```

- First positional arg renamed (`do_propagation` → `do_autozoom`). Callers
  using it positionally are fine; anyone using it as a keyword must rename.
- `max_num_patches` removed (it was already unused).

## `add_bbox_interaction`

```diff
- add_bbox_interaction(bbox_coords, include_interaction, run_prediction=True) -> np.ndarray
+ add_bbox_interaction(bbox_coords, include_interaction, run_prediction=True,
+                      override_capability_checks=False)
```

Behavior changes — these can break programs that previously passed validation:

- Raises `ValueError` if any axis of `bbox_coords` has size 0 (was previously
  silently fixed).
- Raises `ValueError` if a 3D bbox is supplied and the loaded checkpoint does
  not advertise `bbox3d` support (current public weights only support 2D
  bboxes). Pass `override_capability_checks=True` to force.
- The advertised `-> np.ndarray` return type annotation is gone (the function
  never actually returned anything; just an annotation cleanup).

## `add_point_interaction`

```diff
- add_point_interaction(coordinates, include_interaction, run_prediction=True)
+ add_point_interaction(coordinates, include_interaction, run_prediction=True,
+                       override_capability_checks=False)
```

- Runs a capability check for `"points"`; warns + raises if the loaded model
  doesn't support it (use `override_capability_checks=True` to force).

## `add_scribble_interaction` / `add_lasso_interaction`

```diff
- add_scribble_interaction(scribble_image, include_interaction, run_prediction=True)
- add_lasso_interaction(lasso_image,    include_interaction, run_prediction=True)
+ add_scribble_interaction(scribble_image, include_interaction, run_prediction=True,
+                          override_capability_checks=False, interaction_bbox=None)
+ add_lasso_interaction(lasso_image,    include_interaction, run_prediction=True,
+                       override_capability_checks=False, interaction_bbox=None)
```

- **New `interaction_bbox` kwarg (recommended — faster)**: you can now pass a
  *subcrop of the scribble / lasso array* instead of a full-volume mask.
  `interaction_bbox` is `[[x1,x2],[y1,y2],[z1,z2]]` in original-image
  coordinates (half-open). With `interaction_bbox=None` the legacy "must
  match `original_image_shape[1:]`" assertion still applies, so existing code
  keeps working unchanged.
- **Please migrate to the bbox form — the speedup is large.** Scribble and
  lasso annotations are 2D: they live in a single axial / coronal / sagittal
  plane and typically cover only a small region within that one slice. That
  means one of the three bbox dimensions is `1` and the other two are tiny
  compared to the full volume — the cropped array is *orders of magnitude*
  smaller than the full-volume mask. Passing the full volume forces the
  session to allocate, transfer, and process a 3D array that is almost
  entirely zero just to find a handful of non-zero voxels on one slice.
  Handing over the small 2D crop plus its location is dramatically faster
  and uses far less RAM/VRAM.
- Capability checks for `"scribble"` / `"lasso"` (same
  `override_capability_checks` escape hatch).

### How `interaction_bbox` works

The `scribble_image` / `lasso_image` argument you pass must already be cropped
to exactly the region named by `interaction_bbox` — it is NOT a full-volume
image with the bbox naming a region of interest inside it.

The assertion in the implementation:

```python
bbox_size = [ub - lb for lb, ub in interaction_bbox]
assert list(scribble_image.shape) == bbox_size
```

Example — session was set up with an original image whose spatial shape is
`(256, 256, 128)`. The user drew a scribble on a single axial slice
(`z = 64`) covering a small region `x∈[100,140), y∈[80,150)`:

```python
# Recommended: 2D crop, one bbox dim is 1 because the scribble lives in one slice.
# The cropped array is (40, 70, 1) — a few thousand voxels instead of 8.4M.
session.add_scribble_interaction(
    scribble_image=scribble_crop_2d,   # shape (40, 70, 1)
    include_interaction=True,
    interaction_bbox=[[100, 140], [80, 150], [64, 65]],
)

# Legacy / discouraged: full-volume array, 8.4M voxels almost all zero.
session.add_scribble_interaction(
    scribble_image=full_volume_mask,   # shape (256, 256, 128)
    include_interaction=True,
)
```

In this example the cropped form moves ~2800 voxels instead of ~8.4M — about
**three orders of magnitude less data** for a typical single-slice scribble.
The cost difference compounds across every interaction the user makes, so
this is by far the biggest perf win in v2 for any frontend that issues
scribble or lasso prompts.

**This path is the recommended one going forward: please prefer it in new
integrations and migrate existing ones when convenient.**

## `add_initial_seg_interaction`

```diff
- add_initial_seg_interaction(initial_seg, run_prediction=False)
+ add_initial_seg_interaction(initial_seg, run_prediction=False,
+                             override_capability_checks=False)
```

- Capability check for `"initial_label"`.
- Still requires `initial_seg.shape == original_image_shape[1:]` (no
  `interaction_bbox` support here).

## `initialize_from_trained_model_folder` — signature unchanged

Same `(model_training_output_dir, use_fold=None, checkpoint_name='checkpoint_final.pth')`,
but:

- Now prefers a new metadata file `inference_info.json` in the model folder;
  falls back to legacy `inference_session_class.json`. Existing official
  checkpoints still load.
- After this call, the new attributes `num_interaction_channels`,
  `supported_interactions`, `channel_mapping`, `supports_initial_label`,
  `supports_zero_shot_label_refinement` are populated. Channel layout is no
  longer hardcoded at 7 — it comes from the checkpoint capability.

## `set_image`, `set_target_buffer`, `reset_interactions`, `manual_initialization` — unchanged signatures

But under the hood:

- `self.interactions` is now a **blosc2 `NDArray`** (`numpy`-like, `float16`),
  not a `torch.Tensor`. Any collaborator code that did `session.interactions[...]`
  torch-style (`.to(...)`, `.fill_(...)`, slicing into a `torch.Tensor`) will
  break. The new tensor supports numpy-style indexing/assignment.
- The `has_positive_bbox` attribute is removed.

## Other class-level additions worth mentioning

- New class attribute
  `SUPPORTED_INTERACTION_KEYS = ("scribble", "lasso", "points", "bbox2d", "bbox3d")`
  — useful for feature-gating UI against `session.supported_interactions`.
- New `__del__` shuts down the executor; subclasses should chain to it.

## TL;DR for collaborators

1. Drop `use_pinned_memory=` from constructor calls.
2. If you call `set_do_autozoom(do_propagation=...)`, rename to
   `do_autozoom=`; remove `max_num_patches=`.
3. Don't pass collapsed or 3D bboxes (or pass
   `override_capability_checks=True`).
4. If you touch `session.interactions` directly, treat it as a blosc2
   `NDArray` (numpy-like), not a `torch.Tensor`.
5. **Strongly recommended for performance:** send scribble / lasso as a 2D
   subcrop via the new `interaction_bbox=` kwarg instead of a full-volume
   zero array. Scribble and lasso are inherently 2D (they live on a single
   slice, so one bbox dim is `1`), which makes the cropped array orders of 
   magnitude smaller than the full volume for typical
   annotations. The array you pass must already be cropped to exactly the
   bbox size. This is the single biggest perf win in v2 — please prefer it
   in new integrations and migrate existing ones.
