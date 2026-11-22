"""
 * Copyright 2024 University of Edinburgh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.

Gel layout: cut a raw gel photo into lane rows and join them side by side.

A 96-mix SSP tray is run on gel blocks with several rows of wells, e.g. one
gel with 3 rows of 32 lanes ("3x32") or two gel pieces photographed side by
side, each with 3 rows of 16 ("6x16"). Typing expects a single strip where
lane ``i`` is tray mix ``i + 1``, so the raw photo is rearranged first.

Geometry model
--------------
* A :class:`Block` is one physical gel piece, outlined by a quadrilateral
  (TL, TR, BR, BL corners, image pixels). It is rectified with a projective
  transform, which removes rotation, tilt and the vertical offset between
  pieces in one step.
* A :class:`Row` is one row of wells inside a rectified block. ``top`` /
  ``bottom`` / ``left`` / ``right`` are fractions of the rectified block, so
  they stay valid when the outline is nudged. ``top`` should sit just above
  the wells: strips are top-aligned there, so migration distances stay
  comparable across rows. ``left`` / ``right`` bound the *used* lanes, which
  excludes empty wells; ``lanes`` slots are spread evenly between them.
* Rows are joined block-major (block 0 rows top to bottom, then block 1 ...)
  with a background-filled gap between strips. The gap defaults to two lane
  pitches, so the analysis step sees each strip as its own panel (seam).

No pixel rescaling happens apart from the rectification itself, so a band's
distance from the wells is preserved in pixels.

:func:`stitch` returns the joined image and a :class:`Manifest` that records
where every lane landed, so downstream code does not need to guess lane
boundaries from band gaps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

LAYOUT_VERSION = 1
STITCHED = "stitched"
# preset name -> (blocks, rows per block, lanes per row)
PRESETS: Dict[str, Tuple[int, int, int]] = {
    "3x32": (1, 3, 32),
    "6x16": (2, 3, 16),
}
PRESTITCHED_ASPECT = 3.0  # width / height above which an image is treated as joined
DEFAULT_GAP_PITCHES = 2.0

Point = Tuple[float, float]


class LayoutError(ValueError):
    """The layout is malformed or does not fit the image."""


@dataclass
class Row:
    top: float
    bottom: float
    lanes: int
    left: float = 0.0
    right: float = 1.0

    def to_dict(self) -> Dict:
        return {"top": self.top, "bottom": self.bottom, "left": self.left,
                "right": self.right, "lanes": self.lanes}

    @classmethod
    def from_dict(cls, d: Dict) -> "Row":
        try:
            return cls(top=float(d["top"]), bottom=float(d["bottom"]),
                       lanes=int(d["lanes"]), left=float(d.get("left", 0.0)),
                       right=float(d.get("right", 1.0)))
        except (KeyError, TypeError, ValueError) as e:
            raise LayoutError(f"invalid row {d!r}: {e}") from None


@dataclass
class Block:
    quad: List[Point]  # TL, TR, BR, BL in image pixels (x, y)
    rows: List[Row] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {"quad": [[round(x, 2), round(y, 2)] for x, y in self.quad],
                "rows": [r.to_dict() for r in self.rows]}

    @classmethod
    def from_dict(cls, d: Dict) -> "Block":
        try:
            quad = [(float(p[0]), float(p[1])) for p in d["quad"]]
            rows = [Row.from_dict(r) for r in d.get("rows", [])]
        except (KeyError, TypeError, ValueError, IndexError) as e:
            raise LayoutError(f"invalid block: {e}") from None
        return cls(quad=quad, rows=rows)

    def size(self) -> Tuple[int, int]:
        """Rectified (width, height): mean of opposite edge lengths."""
        tl, tr, br, bl = (np.asarray(p, dtype=float) for p in self.quad)
        w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
        h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
        return max(1, int(round(w))), max(1, int(round(h)))

    def transform(self):
        """Projective map from rectified block coords to image coords."""
        from skimage.transform import ProjectiveTransform

        w, h = self.size()
        src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
        return ProjectiveTransform(matrix=_homography(src, np.asarray(self.quad, dtype=float)))

    def to_image(self, pts: Sequence[Point]) -> np.ndarray:
        """Map rectified-block points to image pixels."""
        return self.transform()(np.asarray(pts, dtype=float))


@dataclass
class Layout:
    blocks: List[Block]
    image_size: Tuple[int, int]  # (width, height) the quads refer to
    preset: Optional[str] = None
    gap: Optional[float] = None  # px between strips; None = 2 lane pitches

    @property
    def is_stitched(self) -> bool:
        """True for an already-joined image (identity, no blocks)."""
        return not self.blocks

    @property
    def n_lanes(self) -> int:
        return sum(r.lanes for b in self.blocks for r in b.rows)

    def to_dict(self) -> Dict:
        return {"version": LAYOUT_VERSION, "preset": self.preset,
                "image_size": list(self.image_size), "gap": self.gap,
                "order": "block-major",
                "blocks": [b.to_dict() for b in self.blocks]}

    @classmethod
    def from_dict(cls, d: Dict) -> "Layout":
        if not isinstance(d, dict):
            raise LayoutError("layout must be a JSON object")
        version = d.get("version", LAYOUT_VERSION)
        if version != LAYOUT_VERSION:
            raise LayoutError(f"unsupported layout version {version!r}")
        try:
            size = (int(d["image_size"][0]), int(d["image_size"][1]))
        except (KeyError, TypeError, ValueError, IndexError):
            raise LayoutError("layout needs image_size: [width, height]") from None
        gap = d.get("gap")
        return cls(blocks=[Block.from_dict(b) for b in d.get("blocks", [])],
                   image_size=size, preset=d.get("preset"),
                   gap=None if gap is None else float(gap))

    def save_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "Layout":
        try:
            with open(path) as f:
                return cls.from_dict(json.load(f))
        except json.JSONDecodeError as e:
            raise LayoutError(f"{path}: not valid JSON ({e})") from None

    def scaled_to(self, image_size: Tuple[int, int]) -> "Layout":
        """Rescale block outlines to another image size (for layout reuse)."""
        sx = image_size[0] / self.image_size[0]
        sy = image_size[1] / self.image_size[1]
        if sx == 1 and sy == 1:
            return self
        blocks = [Block(quad=[(x * sx, y * sy) for x, y in b.quad],
                        rows=[Row(**r.to_dict()) for r in b.rows])
                  for b in self.blocks]
        gap = None if self.gap is None else self.gap * sx
        return Layout(blocks=blocks, image_size=tuple(image_size),
                      preset=self.preset, gap=gap)

    def validate(self, expected_lanes: Optional[int] = None) -> List[str]:
        """Raise :class:`LayoutError` for unusable geometry; return warnings."""
        warnings: List[str] = []
        if self.is_stitched:
            return warnings
        W, H = self.image_size
        for bi, b in enumerate(self.blocks):
            tag = f"block {bi + 1}"
            if len(b.quad) != 4:
                raise LayoutError(f"{tag}: outline needs 4 corners, got {len(b.quad)}")
            if not _is_convex_clockwise(b.quad):
                raise LayoutError(
                    f"{tag}: outline must be a convex quad in TL, TR, BR, BL order")
            for x, y in b.quad:
                if not (-0.05 * W <= x <= 1.05 * W and -0.05 * H <= y <= 1.05 * H):
                    raise LayoutError(f"{tag}: corner ({x:.0f}, {y:.0f}) is outside the image")
            if not b.rows:
                raise LayoutError(f"{tag}: has no rows")
            for ri, r in enumerate(b.rows):
                rtag = f"{tag} row {ri + 1}"
                if r.lanes < 1:
                    raise LayoutError(f"{rtag}: lanes must be >= 1")
                if not (0.0 <= r.top < r.bottom <= 1.0):
                    raise LayoutError(f"{rtag}: need 0 <= top < bottom <= 1")
                if not (0.0 <= r.left < r.right <= 1.0):
                    raise LayoutError(f"{rtag}: need 0 <= left < right <= 1")
            spans = sorted((r.top, r.bottom, i) for i, r in enumerate(b.rows))
            for (t0, b0, i0), (t1, _, i1) in zip(spans, spans[1:]):
                if t1 < b0 - 1e-6:
                    warnings.append(f"{tag}: rows {i0 + 1} and {i1 + 1} overlap")
            if [r.top for r in b.rows] != sorted(r.top for r in b.rows):
                warnings.append(f"{tag}: rows are not listed top to bottom; "
                                "lane numbering follows the listed order")
        if expected_lanes is not None and self.n_lanes != expected_lanes:
            raise LayoutError(
                f"layout has {self.n_lanes} lanes, expected {expected_lanes}")
        return warnings


def _homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Exact 4-point homography mapping ``src`` to ``dst`` (3x3, h33 = 1).

    Solved directly rather than via ``ProjectiveTransform.estimate``, whose
    API changed across scikit-image releases.
    """
    a, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.append(u)
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y]); b.append(v)
    try:
        h = np.linalg.solve(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    except np.linalg.LinAlgError:
        raise LayoutError(f"degenerate block outline {dst.tolist()}") from None
    return np.append(h, 1.0).reshape(3, 3)


def _is_convex_clockwise(quad: Sequence[Point]) -> bool:
    """Convex, non-degenerate, and clockwise in image coords (y down)."""
    pts = np.asarray(quad, dtype=float)
    cross = []
    for i in range(4):
        a, b, c = pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]
        cross.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    return all(c > 0 for c in cross)


# --------------------------------------------------------------------------
# construction helpers

def is_prestitched(image_size: Tuple[int, int], ratio: float = PRESTITCHED_ASPECT) -> bool:
    """A strip much wider than tall is assumed to be joined already."""
    w, h = image_size
    return h > 0 and w / h >= ratio


def stitched_layout(image_size: Tuple[int, int]) -> Layout:
    """Identity layout for an image that is already a single joined strip."""
    return Layout(blocks=[], image_size=tuple(image_size), preset=STITCHED)


def preset_layout(preset: str, image_size: Tuple[int, int],
                  split_x: Optional[float] = None) -> Layout:
    """Starting layout for a preset: blocks split evenly across the image,
    rows split evenly inside each block. ``split_x`` overrides where a
    two-block image is divided."""
    if preset == STITCHED:
        return stitched_layout(image_size)
    if preset not in PRESETS:
        raise LayoutError(
            f"unknown preset {preset!r}; choose one of {', '.join(PRESETS)}")
    n_blocks, n_rows, n_lanes = PRESETS[preset]
    W, H = image_size
    if split_x is not None and n_blocks == 2:
        xs = [0.0, float(split_x), float(W)]
    else:
        xs = [W * i / n_blocks for i in range(n_blocks + 1)]
    blocks = []
    for bi in range(n_blocks):
        x0, x1 = xs[bi], xs[bi + 1]
        quad = [(x0, 0.0), (x1, 0.0), (x1, float(H)), (x0, float(H))]
        rows = [Row(top=i / n_rows, bottom=(i + 1) / n_rows, lanes=n_lanes)
                for i in range(n_rows)]
        blocks.append(Block(quad=quad, rows=rows))
    return Layout(blocks=blocks, image_size=tuple(image_size), preset=preset)


# --------------------------------------------------------------------------
# rectify + stitch

@dataclass
class LaneSlot:
    mix: int
    block: int
    row: int
    x0: float  # stitched-image x range of the lane slot
    x1: float

    @property
    def center(self) -> float:
        return (self.x0 + self.x1) / 2.0

    def to_dict(self) -> Dict:
        return {"mix": self.mix, "block": self.block, "row": self.row,
                "x0": round(self.x0, 2), "x1": round(self.x1, 2),
                "center": round(self.center, 2)}


@dataclass
class Strip:
    block: int
    row: int
    x0: int  # stitched-image x range
    x1: int
    height: int  # rows of real gel pixels (the rest is background padding)
    first_mix: int
    last_mix: int

    def to_dict(self) -> Dict:
        return {"block": self.block, "row": self.row, "x0": self.x0, "x1": self.x1,
                "height": self.height, "first_mix": self.first_mix,
                "last_mix": self.last_mix}


@dataclass
class Manifest:
    """Where every lane ended up in the stitched image."""
    image_size: Tuple[int, int]  # stitched (width, height)
    strips: List[Strip]
    lanes: List[LaneSlot]
    stitched_input: bool = False  # True when the input was already joined

    @property
    def lane_edges(self) -> List[float]:
        """Interior x boundaries between consecutive lanes (N-1 values),
        compatible with :func:`ssp.bands.assign_lanes_by_edges`. Between
        strips the boundary is the middle of the gap."""
        return [(a.x1 + b.x0) / 2.0 for a, b in zip(self.lanes, self.lanes[1:])]

    def to_dict(self) -> Dict:
        return {"version": LAYOUT_VERSION, "image_size": list(self.image_size),
                "stitched_input": self.stitched_input,
                "strips": [s.to_dict() for s in self.strips],
                "lanes": [l.to_dict() for l in self.lanes]}


def rectify_block(image: np.ndarray, block: Block) -> np.ndarray:
    """Warp one gel piece to an upright rectangle (dtype preserved)."""
    from skimage.transform import warp

    w, h = block.size()
    out = warp(image, block.transform(), output_shape=(h, w), order=1,
               mode="edge", preserve_range=True)
    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        out = np.clip(np.rint(out), info.min, info.max)
    return out.astype(image.dtype)


def _background(arr: np.ndarray) -> np.ndarray:
    """Per-channel median, used to fill padding and gaps."""
    if arr.ndim == 3:
        return np.median(arr.reshape(-1, arr.shape[-1]), axis=0).astype(arr.dtype)
    return np.asarray(np.median(arr), dtype=arr.dtype)


def _row_box(row: Row, size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    w, h = size
    x0, x1 = int(round(row.left * w)), int(round(row.right * w))
    y0, y1 = int(round(row.top * h)), int(round(row.bottom * h))
    return x0, max(x1, x0 + 1), y0, max(y1, y0 + 1)


def stitch(image: np.ndarray, layout: Layout,
           expected_lanes: Optional[int] = None) -> Tuple[np.ndarray, Manifest]:
    """Cut the rows out of ``image`` and join them into one strip."""
    image = np.asarray(image)
    H, W = image.shape[:2]
    if layout.is_stitched:
        return image, Manifest(image_size=(W, H), strips=[], lanes=[],
                               stitched_input=True)
    if tuple(layout.image_size) != (W, H):
        layout = layout.scaled_to((W, H))
    layout.validate(expected_lanes)

    crops: List[Tuple[int, int, np.ndarray]] = []  # (block, row, crop)
    pitches: List[float] = []
    for bi, block in enumerate(layout.blocks):
        rect = rectify_block(image, block)
        size = (rect.shape[1], rect.shape[0])
        for ri, row in enumerate(block.rows):
            x0, x1, y0, y1 = _row_box(row, size)
            crops.append((bi, ri, rect[y0:y1, x0:x1]))
            pitches.append((x1 - x0) / row.lanes)

    gap = layout.gap if layout.gap is not None else DEFAULT_GAP_PITCHES * float(np.median(pitches))
    gap = max(0, int(round(gap)))
    out_h = max(c.shape[0] for _, _, c in crops)
    out_w = sum(c.shape[1] for _, _, c in crops) + gap * (len(crops) - 1)
    bg = _background(np.concatenate([c.reshape(-1, *c.shape[2:]) for _, _, c in crops]))
    out = np.empty((out_h, out_w) + image.shape[2:], dtype=image.dtype)
    out[...] = bg

    strips: List[Strip] = []
    lanes: List[LaneSlot] = []
    x = 0
    mix = 1
    for (bi, ri, crop), pitch in zip(crops, pitches):
        h, w = crop.shape[:2]
        out[:h, x:x + w] = crop
        n = layout.blocks[bi].rows[ri].lanes
        step = w / n
        for k in range(n):
            lanes.append(LaneSlot(mix=mix + k, block=bi, row=ri,
                                  x0=x + k * step, x1=x + (k + 1) * step))
        strips.append(Strip(block=bi, row=ri, x0=x, x1=x + w, height=h,
                            first_mix=mix, last_mix=mix + n - 1))
        mix += n
        x += w + gap
    return out, Manifest(image_size=(out_w, out_h), strips=strips, lanes=lanes)


# --------------------------------------------------------------------------
# visual check

_COLOURS = [(0, 200, 255), (255, 170, 0), (60, 220, 60), (255, 70, 200),
            (255, 255, 60), (160, 120, 255)]


def _to_display_rgb(image: np.ndarray) -> np.ndarray:
    """8-bit RGB copy for drawing, contrast-stretched to 0.5/99.5 percentiles."""
    arr = np.asarray(image).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=-1) if arr.shape[-1] >= 3 else arr[..., 0]
    lo, hi = np.percentile(arr, [0.5, 99.5])
    arr = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
    g = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=max(8, size))
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def render_overlay(image: np.ndarray, layout: Layout) -> np.ndarray:
    """Original image with block outlines, row cuts and mix numbers drawn on."""
    from PIL import Image, ImageDraw

    image = np.asarray(image)
    H, W = image.shape[:2]
    canvas = Image.fromarray(_to_display_rgb(image))
    if layout.is_stitched:
        return np.asarray(canvas)
    if tuple(layout.image_size) != (W, H):
        layout = layout.scaled_to((W, H))
    draw = ImageDraw.Draw(canvas)
    lw = max(2, int(round(max(W, H) / 600)))
    mix = 1
    for bi, block in enumerate(layout.blocks):
        bw, bh = block.size()
        quad = [tuple(p) for p in block.quad]
        draw.line(quad + [quad[0]], fill=(255, 40, 40), width=lw * 2)
        tform = block.transform()
        for ri, row in enumerate(block.rows):
            colour = _COLOURS[(bi * len(block.rows) + ri) % len(_COLOURS)]
            x0, x1, y0, y1 = row.left * bw, row.right * bw, row.top * bh, row.bottom * bh
            poly = [tuple(p) for p in tform(np.array(
                [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float))]
            draw.line(poly + [poly[0]], fill=colour, width=lw)
            step = (x1 - x0) / row.lanes
            font = _font(int(step * 0.45))
            for k in range(row.lanes + 1):
                xe = x0 + k * step
                seg = tform(np.array([[xe, y0], [xe, y0 + (y1 - y0) * 0.12]], dtype=float))
                draw.line([tuple(seg[0]), tuple(seg[1])], fill=colour, width=max(1, lw // 2))
            for k in range(row.lanes):
                cx, cy = tform(np.array([[x0 + (k + 0.5) * step, y0 + (y1 - y0) * 0.06]]))[0]
                draw.text((cx, cy), str(mix + k), fill=colour, font=font, anchor="mm",
                          stroke_width=max(1, lw // 2), stroke_fill=(0, 0, 0))
            mix += row.lanes
    return np.asarray(canvas)


def render_stitched(stitched: np.ndarray, manifest: Manifest) -> np.ndarray:
    """Stitched image with lane slots and mix numbers, for checking numbering."""
    from PIL import Image, ImageDraw

    canvas = Image.fromarray(_to_display_rgb(stitched))
    if not manifest.lanes:
        return np.asarray(canvas)
    draw = ImageDraw.Draw(canvas)
    H = canvas.size[1]
    pitch = float(np.median([l.x1 - l.x0 for l in manifest.lanes]))
    font = _font(int(pitch * 0.45))
    for si, s in enumerate(manifest.strips):
        colour = _COLOURS[si % len(_COLOURS)]
        draw.rectangle([s.x0, 0, s.x1 - 1, s.height - 1], outline=colour, width=2)
    for lane in manifest.lanes:
        s = next(s for s in manifest.strips if s.first_mix <= lane.mix <= s.last_mix)
        colour = _COLOURS[manifest.strips.index(s) % len(_COLOURS)]
        draw.line([(lane.x0, 0), (lane.x0, H * 0.05)], fill=colour, width=1)
        draw.text((lane.center, H * 0.03), str(lane.mix), fill=colour, font=font, anchor="mm",
                  stroke_width=1, stroke_fill=(0, 0, 0))
    return np.asarray(canvas)

