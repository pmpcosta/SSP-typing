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
"""

import json

import numpy as np
import pytest

from ssp.bands import Band, mask_to_bands, assign_lanes_by_edges, cluster_lanes_by_gap


def _mask_with_blobs():
    m = np.zeros((40, 60), dtype=np.uint8)
    m[5:10, 5:10] = 1     # blob A, 25 px, centroid ~(7,7)
    m[20:24, 30:36] = 1   # blob B, 24 px
    m[30:32, 50:53] = 1   # blob C, 6 px
    return m


def test_mask_to_bands_counts_and_props():
    bands = mask_to_bands(_mask_with_blobs())
    assert len(bands) == 3
    areas = sorted(b.area for b in bands)
    assert areas == [6, 24, 25]
    for b in bands:
        assert isinstance(b, Band)
        assert b.bbox[0] < b.bbox[2] and b.bbox[1] < b.bbox[3]


def test_mask_to_bands_sorted_left_to_right():
    bands = mask_to_bands(_mask_with_blobs())
    xs = [b.centroid_x for b in bands]
    assert xs == sorted(xs)


def test_eight_connectivity_merges_diagonal():
    m = np.zeros((10, 10), dtype=np.uint8)
    m[2, 2] = 1
    m[3, 3] = 1  # diagonal neighbour
    bands = mask_to_bands(m)
    assert len(bands) == 1
    assert bands[0].area == 2


def test_min_area_filters_small_blobs():
    bands = mask_to_bands(_mask_with_blobs(), min_area=10)
    assert len(bands) == 2
    assert all(b.area >= 10 for b in bands)


def test_band_to_dict_is_json_serializable():
    bands = mask_to_bands(_mask_with_blobs())
    d = bands[0].to_dict()
    assert set(d) >= {"id", "area", "centroid_x", "centroid_y", "bbox", "lane"}
    json.dumps(d)  # must not raise


def test_empty_mask_returns_no_bands():
    assert mask_to_bands(np.zeros((20, 20), dtype=np.uint8)) == []


def test_assign_lanes_by_edges():
    bands = mask_to_bands(_mask_with_blobs())
    # lane boundaries split x-axis into [0,20),[20,45),[45,60)
    edges = [20, 45]
    assign_lanes_by_edges(bands, edges)
    lanes = {round(b.centroid_x): b.lane for b in bands}
    assert lanes[7] == 0    # blob A (x~7)
    assert lanes[32] == 1   # blob B (x~32)
    assert lanes[51] == 2   # blob C (x~51)


def test_cluster_lanes_by_gap():
    xs = [5, 6, 7, 40, 41, 80]
    labels = cluster_lanes_by_gap(xs, gap_threshold=10)
    assert labels == [0, 0, 0, 1, 1, 2]


def test_cluster_lanes_by_gap_empty():
    assert cluster_lanes_by_gap([], gap_threshold=10) == []


def test_assign_lanes_by_edges_boundary_goes_to_upper_lane():
    # a band whose centroid sits exactly on an edge belongs to the lane to
    # the right (searchsorted side="right")
    m = np.zeros((20, 40), dtype=np.uint8)
    m[5:10, 18:23] = 1  # centroid_x == 20.0
    bands = mask_to_bands(m)
    assert bands[0].centroid_x == pytest.approx(20.0)
    assign_lanes_by_edges(bands, [20])
    assert bands[0].lane == 1


def test_band_to_dict_rejects_reserved_key_collision():
    b = Band(id=0, area=5, centroid_y=1.0, centroid_x=2.0, bbox=(0, 0, 2, 3),
             extra={"id": 99})
    with pytest.raises(ValueError):
        b.to_dict()


# ---------- split_merged_bands ----------

def _row_gel(lane_bands, n=8, pitch=40, width=24, h=120, bg=0.2):
    """Bands-bright signal + mask. ``lane_bands[i]`` lists (y0, y1, level)
    for lane i; lanes not listed get one 12-px band at y 40-52."""
    sig = np.full((h, n * pitch), bg, dtype=np.float32)
    mask = np.zeros(sig.shape, dtype=np.uint8)
    for i in range(n):
        x0 = i * pitch + (pitch - width) // 2
        for y0, y1, level in lane_bands.get(i, [(40, 52, 0.9)]):
            sig[y0:y1, x0:x0 + width] = level
            mask[y0:y1, x0:x0 + width] = 1
    return mask, sig


def _blobs(mask):
    return sorted((round(b.centroid_y), round(b.centroid_x)) for b in mask_to_bands(mask))


def test_split_cuts_a_product_merged_under_its_control():
    # Gel_01 lane 70: a fainter control with a strong product right below,
    # one 24-px blob where the row's bands are 12 px
    from ssp.bands import split_merged_bands
    mask, sig = _row_gel({3: [(40, 52, 0.55), (52, 64, 0.95)]})
    mask[40:64, 3 * 40 + 8:3 * 40 + 32] = 1
    assert len(mask_to_bands(mask)) == 8
    out = split_merged_bands(mask, sig)
    lane3 = [c for c in _blobs(out) if c[1] == 140]
    assert len(lane3) == 2
    assert lane3[0][0] == pytest.approx(46, abs=2) and lane3[1][0] == pytest.approx(58, abs=2)


def test_split_keeps_an_overloaded_band_whole():
    from ssp.bands import split_merged_bands
    mask, sig = _row_gel({3: [(40, 64, 0.95)]})     # tall but uniform
    assert _blobs(split_merged_bands(mask, sig)) == _blobs(mask)


def test_split_keeps_faint_smears_and_rows_of_tall_bands():
    from ssp.bands import split_merged_bands
    # faint two-level smear: tall and stepped, but far weaker than the bands
    mask, sig = _row_gel({3: [(40, 52, 0.25), (52, 64, 0.32)]})
    mask[40:64, 128:152] = 1
    assert _blobs(split_merged_bands(mask, sig)) == _blobs(mask)
    # a dimer row: every lane's band is tall, so none is an outlier
    mask, sig = _row_gel({i: [(40, 52, 0.55), (52, 64, 0.95)] for i in range(8)})
    assert _blobs(split_merged_bands(mask, sig)) == _blobs(mask)


def test_segment_image_splits_unless_told_not_to():
    from ssp.cli import segment_image
    from ssp.inference import GelGenieONNX

    class Thresh:   # foreground where the (bands-bright) input > 0.3
        def get_inputs(self):
            return [type('io', (), {'name': 'in'})()]

        def get_outputs(self):
            return [type('io', (), {'name': 'out'})()]

        def run(self, names, feed):
            x = feed['in']
            fg = (x[:, 0] > 0.3).astype(np.float32)
            return [np.stack([1 - fg, fg], axis=1)]

    mask, sig = _row_gel({3: [(40, 52, 0.55), (52, 64, 0.95)]})
    raw = (sig * 255).astype(np.uint8)
    model = GelGenieONNX(session=Thresh())
    _, merged = segment_image(raw, model, split_merged=False)
    _, split = segment_image(raw, model)
    assert len(split) == len(merged) + 1 == 9
