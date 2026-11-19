"""Tests for skipping all-zero interaction regions (nnInteractive/utils/interaction_regions.py).

Network inputs copy only the parts of a crop that ``session._interaction_regions`` marks as possibly nonzero. If a
write to ``session.interactions`` does not register its region first (``_record_interactions_region``), the network
silently sees zeros there. These tests guard against that:

1. unit tests of InteractionRegions (merging, cap, lookups);
2. end-to-end: the same scripted session (every interaction type, AutoZoom + refinement, undo, reset, initial seg,
   renormalization) runs twice, with region skipping and with whole-crop copies. Target buffer and interactions must
   be bit-identical after every step, every read validates that the skipped voxels are zero
   (VALIDATE_INTERACTION_REGIONS), and after every step the regions must cover every nonzero voxel;
3. a planted unregistered write must be caught by the validation;
4. a tripwire over the session source: every method that touches ``self.interactions`` and every call that skips
   recording (``record_undo=False``-style) must be in a reviewed list, so new write paths get a deliberate look.

Runs on CPU with a small stand-in network (no model download); also on CUDA if available. Run with
``pytest tests/test_interaction_regions.py``.
"""

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

import nnInteractive.inference.inference_session as session_module
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
from nnInteractive.interaction.point import PointInteraction_stub
from nnInteractive.utils.interaction_regions import (
    InteractionRegions,
    box_contains,
    box_volume,
)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


# ----------------------------------------------------------------------------------------------------------------------
# 1. InteractionRegions
# ----------------------------------------------------------------------------------------------------------------------


def test_far_apart_prompts_keep_separate_boxes():
    r = InteractionRegions(1)
    r.add(0, [[10, 19], [10, 19], [10, 19]])  # a point near one corner
    r.add(0, [[400, 409], [400, 409], [400, 409]])  # a point far away
    assert len(r.boxes(0)) == 2
    # a crop around the first point reads only the first box
    assert r.parts_in(0, [[0, 100], [0, 100], [0, 100]]) == [[[10, 19], [10, 19], [10, 19]]]
    # a crop between them reads nothing
    assert r.parts_in(0, [[100, 300], [100, 300], [100, 300]]) == []


def test_nearby_and_overlapping_boxes_merge():
    r = InteractionRegions(1)
    r.add(0, [[0, 100], [0, 100], [0, 100]])
    r.add(0, [[100, 200], [0, 100], [0, 100]])  # adjacent: the hull is exactly the union
    assert r.boxes(0) == [[[0, 200], [0, 100], [0, 100]]]
    r.add(0, [[10, 19], [10, 19], [300, 309]])
    r.add(0, [[12, 21], [12, 21], [312, 321]])  # two points 12 voxels apart: hull wastes < MERGE_SLACK_VOXELS
    assert len(r.boxes(0)) == 2


def test_contained_boxes_are_dropped():
    r = InteractionRegions(1)
    r.add(0, [[0, 10], [0, 10], [0, 10]])
    r.add(0, [[0, 100], [0, 100], [0, 100]])
    r.add(0, [[5, 6], [5, 6], [5, 6]])
    assert r.boxes(0) == [[[0, 100], [0, 100], [0, 100]]]


def test_cap_merges_least_wasteful_pair_and_still_covers_everything():
    r = InteractionRegions(1)
    rng = np.random.default_rng(0)
    added = []
    for _ in range(100):
        lb = rng.integers(0, 2000, 3)
        box = [[int(lb[i]), int(lb[i]) + 5] for i in range(3)]
        added.append(box)
        r.add(0, box)
    assert len(r.boxes(0)) <= InteractionRegions.MAX_BOXES_PER_CHANNEL
    for box in added:
        assert any(box_contains(b, box) for b in r.boxes(0))


def test_parts_in():
    r = InteractionRegions(2)
    r.add(0, [[0, 50], [0, 50], [0, 50]])
    region = [[10, 20], [10, 20], [10, 20]]
    assert r.parts_in(0, region) == [region]  # fully covered
    assert r.parts_in(1, region) == []  # empty channel
    assert r.parts_in(0, [[40, 60], [0, 10], [0, 10]]) == [[[40, 50], [0, 10], [0, 10]]]  # partial
    r.add(1, [[0, 10], [0, 10], [0, 10]])
    r.add(1, [[1000, 1010], [0, 10], [0, 10]])  # far enough apart that the hull would waste > MERGE_SLACK_VOXELS
    parts = r.parts_in(1, [[0, 2000], [0, 300], [0, 300]])
    assert sorted(parts) == [[[0, 10], [0, 10], [0, 10]], [[1000, 1010], [0, 10], [0, 10]]]


def test_empty_boxes_are_ignored_and_clear_copy_update():
    r = InteractionRegions(1)
    r.add(0, [[5, 5], [0, 10], [0, 10]])
    assert r.boxes(0) == []
    r.add(0, [[0, 10], [0, 10], [0, 10]])
    c = r.copy()
    r.clear()
    assert r.boxes(0) == [] and c.boxes(0) == [[[0, 10], [0, 10], [0, 10]]]
    r.add(0, [[500, 510], [0, 10], [0, 10]])
    r.update(c)
    assert sorted(r.boxes(0)) == [[[0, 10], [0, 10], [0, 10]], [[500, 510], [0, 10], [0, 10]]]


# ----------------------------------------------------------------------------------------------------------------------
# 2. End-to-end: region skipping gives bit-identical sessions
# ----------------------------------------------------------------------------------------------------------------------

PATCH = (96, 96, 96)  # the refinement planner's fixed 24-voxel margin needs patches well above 48
SHAPE = (200, 170, 140)


class StandInNetwork(nn.Module):
    """Deterministic stand-in for the trained network: foreground where the normalized image is bright, pushed
    up/down around positive/negative prompts and by prev_seg. Every interaction channel influences the output, so a
    channel that reached the network as zeros would change the segmentation."""

    def forward(self, x):
        image, inter = x[:, :1].float(), x[:, 1:].float()
        pos = inter[:, [1, 3, 5]].sum(1, keepdim=True)
        neg = inter[:, [2, 4, 6]].sum(1, keepdim=True)
        drive = F.avg_pool3d(pos - neg, 5, stride=1, padding=2) * 8
        fg = image + drive + 0.5 * inter[:, :1] - 0.2
        return torch.cat((torch.zeros_like(fg), fg), 1)


def _ellipsoid(shape, center, radii):
    grids = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    return sum(((g - c) / r) ** 2 for g, c, r in zip(grids, center, radii)) <= 1


def make_image():
    rng = np.random.default_rng(1)
    img = rng.normal(100, 20, SHAPE).clip(1, None)
    img[_ellipsoid(SHAPE, (100, 85, 70), (60, 45, 40))] = 1000  # larger than a patch: AutoZoom + refinement
    img[_ellipsoid(SHAPE, (30, 30, 30), (8, 8, 8))] = 1000  # one patch
    img[_ellipsoid(SHAPE, (170, 140, 110), (10, 10, 10))] = 1000  # far corner
    return img.astype(np.int16)[None]


def make_session(storage, skip, device, enable_undo=True):
    s = nnInteractiveInferenceSession(
        device=torch.device(device),
        use_torch_compile=False,
        torch_n_threads=4,
        interactions_storage=storage,
        enable_undo=enable_undo,
    )
    s.initialize_from_loaded_artifacts(
        {
            "capability_content": {},  # legacy defaults: 7 channels, prev_seg 0, bbox/lasso 1-2, points 3-4, scribble 5-6
            "point_interaction": PointInteraction_stub(4, True),
            "preferred_scribble_thickness": [2, 2, 2],
            "interaction_decay": 0.98,
            "pad_mode_data": "constant",
            "network": StandInNetwork().to(device),
            "plans_manager": None,
            "configuration_manager": SimpleNamespace(patch_size=list(PATCH)),
            "dataset_json": {},
            "trainer_name": "stand-in",
            "label_manager": None,
            "license": "test",
        }
    )
    s.SKIP_ZERO_INTERACTION_REGIONS = skip
    s.VALIDATE_INTERACTION_REGIONS = skip  # every skipped read checks that the skipped voxels are zero
    return s


def interactions_numpy(s):
    s._finish_preprocessing_and_initialize_interactions()  # set_image initializes them in the background
    x = s.interactions
    return x.cpu().numpy() if isinstance(x, torch.Tensor) else x[:]


def assert_regions_cover_nonzero(s):
    """The invariant: every nonzero voxel lies inside one of its channel's boxes."""
    inter = interactions_numpy(s)
    for c in range(inter.shape[0]):
        outside = inter[c] != 0
        for b in s._interaction_regions.boxes(c):
            outside[tuple(slice(lb, ub) for lb, ub in b)] = False
        assert (
            not outside.any()
        ), f"channel {c}: {int(outside.sum())} nonzero voxels outside {s._interaction_regions.boxes(c)}"


def _mask(points, width=1):
    m = np.zeros(SHAPE, dtype=np.uint8)
    for p in points:
        m[tuple(slice(max(c - width, 0), c + width + 1) for c in p)] = 1
    return m


def _initial_seg():
    return _ellipsoid(SHAPE, (100, 85, 70), (55, 40, 35)).astype(np.uint8)


def script():
    """(name, action) pairs covering every write path of the interactions."""
    line = [(170, 140 + d, 110) for d in range(-6, 7)]  # a scribble at the far-corner object
    lasso = np.zeros((40, 40, 1), dtype=np.uint8)
    lasso[5:35, 5:35] = 1

    def renorm_soon(s):
        # make _renormalize_interactions_if_needed fire within the next prompts (normally every few hundred)
        s._fp16_max_value = s.current_interaction_intensity * 1.03
        s._interaction_renorm_target = 0.5

    return [
        ("click in the large object (AutoZoom + refinement)", lambda s: s.add_point_interaction((100, 85, 70), True)),
        ("negative click", lambda s: s.add_point_interaction((140, 95, 75), False)),
        ("2D box on the small object", lambda s: s.add_bbox_interaction([[20, 41], [20, 41], [30, 31]], True)),
        ("undo the box", lambda s: s.undo()),
        ("full-size scribble, no interaction_bbox", lambda s: s.add_scribble_interaction(_mask(line), True)),
        (
            "negative lasso with interaction_bbox",
            lambda s: s.add_lasso_interaction(lasso, False, interaction_bbox=[[80, 120], [60, 100], [70, 71]]),
        ),
        ("click far away", lambda s: s.add_point_interaction((170, 140, 110), True)),
        ("renormalization", renorm_soon),
        ("click after renorm setup", lambda s: s.add_point_interaction((100, 60, 70), True)),
        ("click triggering renormalization", lambda s: s.add_point_interaction((100, 110, 70), True)),
        ("reset", lambda s: s.reset_interactions()),
        ("initial seg with prediction", lambda s: s.add_initial_seg_interaction(_initial_seg(), run_prediction=True)),
        ("click after initial seg", lambda s: s.add_point_interaction((160, 85, 70), True)),
        ("undo the click", lambda s: s.undo()),
        ("click into empty area", lambda s: s.add_point_interaction((30, 150, 120), True)),
        ("initial seg without prediction (resets)", lambda s: s.add_initial_seg_interaction(_mask([(30, 30, 30)], 6))),
        ("undo the initial seg (brings the reset interactions back)", lambda s: s.undo()),
        ("click after undo", lambda s: s.add_point_interaction((30, 30, 30), True)),
        ("negative 2D box", lambda s: s.add_bbox_interaction([[60, 140], [40, 41], [30, 110]], False)),
        ("new image", lambda s: (s.set_image(make_image()), s.set_target_buffer(np.zeros(SHAPE, np.uint8)))),
        ("click on the new image", lambda s: s.add_point_interaction((100, 85, 70), True)),
    ]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("storage", ["tensor", "blosc2"])
@pytest.mark.parametrize("enable_undo", [True, False])
def test_skipping_zero_regions_is_bit_identical(device, storage, enable_undo):
    sessions = []
    for skip in (True, False):
        s = make_session(storage, skip, device, enable_undo)
        s.set_image(make_image())
        s.set_target_buffer(np.zeros(SHAPE, dtype=np.uint8))
        sessions.append(s)
    skipping, full = sessions
    zoomed = 0
    for k, (name, action) in enumerate(script()):
        if not enable_undo and name.startswith("undo"):
            continue
        for s in sessions:
            np.random.seed(k)  # refinement planning may fall back to random sampling
            action(s)
        assert np.array_equal(skipping.target_buffer, full.target_buffer), name
        assert np.array_equal(interactions_numpy(skipping), interactions_numpy(full)), name
        assert_regions_cover_nonzero(skipping)
        zoomed += skipping.target_buffer.any()
    assert zoomed  # the script did segment something


def test_scribble_write_is_limited_to_its_nonzero_bbox():
    s = make_session("tensor", True, "cpu")
    s.set_image(make_image())
    s.set_target_buffer(np.zeros(SHAPE, dtype=np.uint8))
    s.add_scribble_interaction(_mask([(170, 140, 110)]), True, run_prediction=False)
    assert s._interaction_regions.boxes(5) == [[[169, 172], [139, 142], [109, 112]]]


def _awkward_masks():
    """Full-size masks whose nonzero bbox is easy to get wrong."""
    base = np.zeros(SHAPE, dtype=np.int16)
    base[60:70, 50:60, 40:45] = 1
    cancelling = base.copy()
    cancelling[150, 20, 20], cancelling[150, 20, 21] = 1, -1  # projection sums over x-slice 150 are exactly 0
    floating = base.astype(np.float32) * 0.7
    floating[10, 10, 10] = -2.5
    return {
        "uint8": base.astype(np.uint8),
        "bool": base.astype(bool),
        "labels": (base * 3).astype(np.uint8),
        "float with a negative voxel": floating,
        "int16 with cancelling projections": cancelling,
        "empty": np.zeros(SHAPE, dtype=np.uint8),
    }


def _nonzero_bbox(mask):
    nz = np.nonzero(mask != 0)
    return [[int(i.min()), int(i.max()) + 1] for i in nz] if nz[0].size else None


@pytest.mark.parametrize("storage", ["tensor", "blosc2"])
@pytest.mark.parametrize("kind", list(_awkward_masks()))
def test_restricted_mask_writes_equal_whole_mask_writes(storage, kind):
    """Initial seg and scribbles write only the bbox of the mask's nonzero voxels. The channel must still be exactly
    what writing the whole mask gives (initial seg: the mask as fp16; scribble: max(0, fp16(mask) * intensity)), and
    the tracked region must be exactly that bbox."""
    mask = _awkward_masks()[kind]
    bbox = _nonzero_bbox(mask)
    s = make_session(storage, True, "cpu")
    s.set_image(make_image())
    s.set_target_buffer(np.zeros(SHAPE, dtype=np.uint8))

    s.add_initial_seg_interaction(mask, run_prediction=False)
    expected = torch.from_numpy(mask).to(torch.float16).numpy()
    assert np.array_equal(interactions_numpy(s)[0], expected)
    assert s._interaction_regions.boxes(0) == ([bbox] if bbox else [])

    s.add_scribble_interaction(mask, True, run_prediction=False)
    new_values = mask.astype(np.float16)
    new_values *= s.current_interaction_intensity
    assert np.array_equal(interactions_numpy(s)[5], np.maximum(np.zeros_like(new_values), new_values))
    assert s._interaction_regions.boxes(5) == ([bbox] if bbox else [])


# ----------------------------------------------------------------------------------------------------------------------
# 3. A write that bypasses _record_interactions_region is caught
# ----------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "planted, path",
    [
        # inside the one-patch crop of the click: caught by _interactions_region_into
        ((100, 85, 70), "one-patch read"),
        # outside the one-patch crop (x 52-148) but inside the first AutoZoom crop (x 28-172): caught by
        # _interactions_region_is_zero in _build_network_input's zoomed branch
        ((160, 85, 70), "AutoZoom read"),
    ],
)
def test_unregistered_write_is_caught_by_validation(device, planted, path):
    s = make_session("tensor", True, device)
    s.set_image(make_image())
    s.set_target_buffer(np.zeros(SHAPE, dtype=np.uint8))
    # BUG on purpose: a scribble voxel written without _record_interactions_region. The click below writes only the
    # point channels, so nothing registers this voxel afterwards.
    s._finish_preprocessing_and_initialize_interactions()
    s.interactions[(5, *planted)] = 1
    with pytest.raises(AssertionError, match="did not call _record_interactions_region"):
        s.add_point_interaction((100, 85, 70), True)


# ----------------------------------------------------------------------------------------------------------------------
# 4. Tripwire: new code that touches self.interactions must be reviewed
# ----------------------------------------------------------------------------------------------------------------------

# Methods of nnInteractiveInferenceSession that mention self.interactions, and why they keep the invariant (every
# nonzero voxel inside a tracked region). Adding a method here means: I checked that every write it makes to the
# interactions calls _record_interactions_region for (a region containing) the written region beforehand, within the
# same interaction, or that it only zeroes/scales/restores values.
REVIEWED_INTERACTIONS_METHODS = {
    # write, recording first
    "_interactions_inplace_maximum": "records the region (also passes the decompressed current values)",
    "_write_interactions_channel": "records the whole channel",
    "_add_validated_bbox_interaction": "records the box",
    "add_point_interaction": "place_point calls before_write -> _record_interactions_region",
    "_paste_interactions": "records unless record_undo=False (see RECORD_SKIPPING_CALLS)",
    # write, but cannot create nonzero voxels
    "_renormalize_interactions_if_needed": "scales in place; records whole channels when undo is on",
    "reset_interactions": "zeroes everything and clears the regions",
    "_restore_undo_record": "restores pre-images; undo() unions the regions with the ones from before",
    "_reset_session": "drops the interactions and the regions",
    "_initialize_interactions": "new zero array, new empty regions",
    # read only
    "__init__": "sets it to None",
    "_compute_prev_seg_positive_bbox": "read",
    "_get_non_prev_seg_channels": "shape",
    "_blosc2_staging_available": "type check",
    "_blosc2_channel_to_device": "read",
    "_copy_interactions_region_into": "read (raw copy, used by _interactions_region_into)",
    "_read_interactions_region": "read",
    "_record_paste_pre_images_ahead": "records (adds the region ahead of the paste)",
    "_build_refinement_local_cache": "shape",
    "_new_interactions_read_buffer": "type check",
    "_record_interactions_region": "THE recording function",
    "_record_interactions_channels": "records whole channels",
    "_predict": "cratio print",
    "_build_network_input": "shape",
    "_refine_coarse_with_local_cache": "shape",
    "_compute_local_diff_map": "shape",
    "_plan_refinement_bboxes": "shape",
}

# Calls that write interactions without recording, because an earlier record in the same interaction covers them.
RECORD_SKIPPING_CALLS = {
    # 1x prediction: _record_paste_pre_images_ahead recorded scaled_bbox (paste_undo is not None), else it records
    ("_predict", "_paste_interactions"): "record_undo=paste_undo is None",
    # refinement commit: the hull of all refinement boxes (cache_bbox) is recorded in undo_steps before the commit
    ("_refine_coarse_with_local_cache", "_paste_interactions"): "record_undo=False",
}


def _session_class_ast():
    tree = ast.parse(Path(inspect.getsourcefile(session_module)).read_text())
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "nnInteractiveInferenceSession")


def test_every_method_touching_interactions_is_reviewed():
    touching = set()
    for f in _session_class_ast().body:
        if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for n in ast.walk(f):
                if isinstance(n, ast.Attribute) and n.attr == "interactions" and getattr(n.value, "id", None) == "self":
                    touching.add(f.name)
    unreviewed = touching - set(REVIEWED_INTERACTIONS_METHODS)
    assert not unreviewed, (
        f"{sorted(unreviewed)} touch self.interactions. If they write to it, they must call "
        f"_record_interactions_region for the written region first (see the comment at self.interactions in "
        f"__init__); then add them to REVIEWED_INTERACTIONS_METHODS."
    )
    stale = set(REVIEWED_INTERACTIONS_METHODS) - touching
    assert not stale, f"{sorted(stale)} no longer touch self.interactions: remove them from the list"


def test_every_call_skipping_the_record_is_reviewed():
    found = set()
    for f in _session_class_ast().body:
        if not isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for n in ast.walk(f):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                for kw in n.keywords:
                    if kw.arg == "record_undo" and not (isinstance(kw.value, ast.Constant) and kw.value.value is True):
                        found.add((f.name, n.func.attr))
    interaction_calls = {c for c in found if "interactions" in c[1]}
    assert interaction_calls == set(RECORD_SKIPPING_CALLS), (
        f"Calls writing interactions without recording changed: {sorted(interaction_calls ^ set(RECORD_SKIPPING_CALLS))}."
        f" Such a call is only correct if an earlier _record_interactions_region call in the same interaction covers"
        f" the written region; check that, then update RECORD_SKIPPING_CALLS."
    )


def test_regions_are_created_and_cleared_only_where_channels_are_zeroed():
    src = Path(inspect.getsourcefile(session_module)).read_text()
    assert src.count("self._interaction_regions.clear()") == 1  # reset_interactions, right after zeroing
    assert src.count("self._interaction_regions = InteractionRegions(") == 1  # _initialize_interactions
    assert box_volume([[0, 2], [0, 3], [0, 4]]) == 24


def test_mask_nonzero_bbox_is_exact_for_every_dtype():
    """_mask_nonzero_bbox (fast uint8 max projections) must equal the exact bbox of mask != 0 for every dtype,
    including signed masks whose float sums would cancel."""
    rng = np.random.default_rng(0)
    for _ in range(300):
        shape = tuple(int(x) for x in rng.integers(1, 12, 3))
        dtype = rng.choice(["uint8", "bool", "int16", "int32", "float32"])
        mask = np.zeros(shape, dtype=dtype)
        for _ in range(int(rng.integers(0, 4))):
            idx = tuple(int(rng.integers(0, s)) for s in shape)
            mask[idx] = True if dtype == "bool" else rng.choice([1, 2, -1, -3, 0.5]) if dtype != "uint8" else 7
        if dtype in ("int16", "int32") and shape[1] > 1 and rng.random() < 0.3:
            mask[0, 0, 0], mask[0, 1, 0] = 2, -2  # cancels in a float sum over x-slice 0
        assert nnInteractiveInferenceSession._mask_nonzero_bbox(torch.from_numpy(mask)) == _nonzero_bbox(mask)
