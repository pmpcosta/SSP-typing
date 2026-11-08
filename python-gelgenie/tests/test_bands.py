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
