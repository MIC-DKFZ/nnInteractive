"""Where each interaction channel can be nonzero, so that reads of all-zero regions can be skipped.

The interaction channels are mostly zeros: a point is a small sphere, a 2D box a thin slab, prev_seg covers one
object. Yet every network input used to copy the whole crop of every channel from host RAM to the GPU, 7 fp16
channels = 14 of the 18 bytes per voxel moved for an AutoZoom step. ``InteractionRegions`` keeps, per channel, a
short list of boxes that together cover every voxel that can be nonzero. ``_interactions_region_into`` (in the
inference session) then copies only the parts of a crop that lie inside these boxes and zero-fills the rest on the
destination device.

THE INVARIANT the network input depends on:

    Every nonzero voxel of channel c lies inside at least one box of channel c.

The boxes are a cover, not the exact support: they may contain zeros and may overlap. Breaking the invariant does not
raise anything by itself; the network just silently sees zeros where there are interactions. The session upholds it
by adding the region of every write to the interaction tensor *before* the write, in
``nnInteractiveInferenceSession._record_interactions_region``, which every write must call anyway for undo (see the
comments there). Clearing is only allowed where the channels are zeroed (reset, new image).
``NNINTERACTIVE_VALIDATE_INTERACTION_REGIONS=1`` makes every skipped read verify that the skipped voxels are zero
(slow; the tests run with it).

Box layout: ``[[lb, ub], ...]`` per spatial axis, half-open, like every other bbox in this package.
"""

Box = list[list[int]]


def box_volume(box: Box) -> int:
    v = 1
    for lb, ub in box:
        v *= max(ub - lb, 0)
    return v


def box_intersection(a: Box, b: Box) -> Box | None:
    """Overlap of two boxes, or None if they do not overlap."""
    out = []
    for (alb, aub), (blb, bub) in zip(a, b):
        lb, ub = max(alb, blb), min(aub, bub)
        if ub <= lb:
            return None
        out.append([lb, ub])
    return out


def box_hull(a: Box, b: Box) -> Box:
    """Smallest box containing both."""
    return [[min(alb, blb), max(aub, bub)] for (alb, aub), (blb, bub) in zip(a, b)]


def box_contains(outer: Box, inner: Box) -> bool:
    return all(olb <= ilb and iub <= oub for (olb, oub), (ilb, iub) in zip(outer, inner))


def merge_waste(a: Box, b: Box) -> int:
    """Voxels a merged box (the hull of a and b) covers that neither a nor b covers."""
    inter = box_intersection(a, b)
    union = box_volume(a) + box_volume(b) - (box_volume(inter) if inter is not None else 0)
    return box_volume(box_hull(a, b)) - union


class InteractionRegions:
    """Per-channel list of boxes covering every voxel that can be nonzero (see the module docstring for the invariant).

    Each write adds its own box, so prompts far apart keep separate boxes: a crop around one of them copies only that
    one. A new box is merged into an existing one only when that wastes little (the hull adds at most
    ``MERGE_SLACK_FRACTION`` of the two boxes' union, or at most ``MERGE_SLACK_VOXELS`` voxels): overlapping refinement
    patches of one object, or points a few voxels apart, become one box. Beyond ``MAX_BOXES_PER_CHANNEL`` boxes, the two
    boxes whose hull wastes the least are merged, so many scattered prompts degrade gracefully towards one large box.
    Whatever the boxes, a read copies at most the requested region, so no read ever moves more than without this.
    Boxes only grow or merge, they never shrink, until the channels are zeroed (``clear``).
    """

    MAX_BOXES_PER_CHANNEL = 16
    MERGE_SLACK_FRACTION = 0.25
    MERGE_SLACK_VOXELS = 32**3

    def __init__(self, num_channels: int):
        self._boxes: list[list[Box]] = [[] for _ in range(num_channels)]

    def clear(self) -> None:
        """Forget all boxes. ONLY call this when every interaction channel has just been zeroed."""
        for boxes in self._boxes:
            boxes.clear()

    def add(self, channel: int, box: Box) -> None:
        """Record that ``box`` of ``channel`` may become nonzero. Call BEFORE writing there (see module docstring)."""
        box = [[int(lb), int(ub)] for lb, ub in box]
        if any(ub <= lb for lb, ub in box):
            return
        boxes = self._boxes[channel]
        if any(box_contains(b, box) for b in boxes):
            return
        boxes[:] = [b for b in boxes if not box_contains(box, b)]
        merged = True
        while merged:
            merged = False
            for k, b in enumerate(boxes):
                if self._cheap_to_merge(b, box):
                    box = box_hull(b, box)
                    del boxes[k]
                    merged = True
                    break
        boxes.append(box)
        while len(boxes) > self.MAX_BOXES_PER_CHANNEL:
            pairs = [(i, j) for i in range(len(boxes)) for j in range(i + 1, len(boxes))]
            i, j = min(pairs, key=lambda p: merge_waste(boxes[p[0]], boxes[p[1]]))
            hull = box_hull(boxes[i], boxes[j])
            del boxes[j], boxes[i]  # j > i: delete j first so i stays valid
            boxes[:] = [b for b in boxes if not box_contains(hull, b)]
            boxes.append(hull)

    def _cheap_to_merge(self, a: Box, b: Box) -> bool:
        waste = merge_waste(a, b)
        inter = box_intersection(a, b)
        union = box_volume(a) + box_volume(b) - (box_volume(inter) if inter is not None else 0)
        return waste <= max(self.MERGE_SLACK_VOXELS, self.MERGE_SLACK_FRACTION * union)

    def boxes(self, channel: int) -> list[Box]:
        return [[list(ax) for ax in b] for b in self._boxes[channel]]

    def parts_in(self, channel: int, region: Box) -> list[Box]:
        """The parts of ``region`` that may be nonzero in ``channel``: the overlaps of ``region`` with the channel's
        boxes, in global coordinates. ``[region]`` if one box covers all of it; ``[]`` if the whole region is zero.
        Parts may overlap (copying an overlap twice copies the same values twice, which is harmless)."""
        region = [[int(lb), int(ub)] for lb, ub in region]
        parts: list[Box] = []
        for b in self._boxes[channel]:
            inter = box_intersection(b, region)
            if inter is None:
                continue
            if inter == region:
                return [region]
            if any(box_contains(p, inter) for p in parts):
                continue
            parts = [p for p in parts if not box_contains(inter, p)]
            parts.append(inter)
        return parts

    def copy(self) -> "InteractionRegions":
        other = InteractionRegions(len(self._boxes))
        other._boxes = [self.boxes(c) for c in range(len(self._boxes))]
        return other

    def update(self, other: "InteractionRegions") -> None:
        """Add all boxes of ``other`` (union of the two covers)."""
        for c in range(len(self._boxes)):
            for b in other._boxes[c]:
                self.add(c, b)
