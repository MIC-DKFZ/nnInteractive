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

- **New `interaction_bbox` kwarg**: you can now pass a *subcrop of the
  scribble / lasso array* instead of a full-volume mask. `interaction_bbox` is
  `[[x1,x2],[y1,y2],[z1,z2]]` in original-image coordinates (half-open).
  With `interaction_bbox=None` the legacy "must match
  `original_image_shape[1:]`" assertion still applies, so existing code keeps
  working unchanged.
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
`(256, 256, 128)`. You want to add a scribble inside the subvolume
`x∈[10,50), y∈[20,80), z∈[0,32)`:

```python
# Correct: scribble array shape matches bbox size (40, 60, 32)
session.add_scribble_interaction(
    scribble_image=np.zeros((40, 60, 32), dtype=np.float16),
    include_interaction=True,
    interaction_bbox=[[10, 50], [20, 80], [0, 32]],
)

# Wrong: full-volume scribble array, will fail the assert
session.add_scribble_interaction(
    scribble_image=np.zeros((256, 256, 128), dtype=np.float16),
    include_interaction=True,
    interaction_bbox=[[10, 50], [20, 80], [0, 32]],   # size mismatch
)
```

The point of the feature is that you no longer need to allocate a full-volume
zero array just to mark a few voxels — you hand over the small crop plus its
location in original-image coordinates.

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
5. Optional: scribble / lasso can now be sent as a subcrop via the new
   `interaction_bbox=` kwarg — handy if you want to skip wrapping a tiny
   annotation in a full-volume zero array. The array you pass must already be
   cropped to exactly the bbox size.
