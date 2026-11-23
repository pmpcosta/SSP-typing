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

The web layout editor (static/layout.js) re-implements the layout geometry
so lane numbers follow every drag instantly. If it drifted from
ssp/layout.py, the editor would show lane numbers that the actual cut does
not use. Runs the JS under Node (skipped when Node is not installed).
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from ssp.layout import Block

NODE = shutil.which("node")
LAYOUT_JS = Path(__file__).resolve().parents[1] / "ssp" / "webapp" / "static" / "layout.js"

QUADS = [
    [[121.4, 348.7], [1636.7, 327.7], [1674.5, 2016.2], [142.2, 2016.2]],  # 3x32 photo
    [[883.0, 30.1], [1583.6, 26.1], [1584.8, 1372.0], [884.2, 1372.0]],    # 6x16 piece
    [[10, 20], [500, 0], [520, 400], [0, 380]],                            # strong skew
]
POINTS = [[0, 0], [100.5, 37.2], [400, 300], [12.25, 999.0]]

RUNNER = r"""
const fs = require('fs'), vm = require('vm');
const ctx = {}; vm.createContext(ctx);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8') + '\nthis.LE = LayoutEditor;', ctx);
const G = ctx.LE._geom, input = JSON.parse(process.argv[2]);
const out = input.quads.map((q) => {
  const g = G.homography(q);
  return { size: G.blockSize(q), convex: G.convex(q),
           mapped: input.points.map(([x, y]) => G.apply(g.H, x, y)),
           back: input.points.map(([x, y]) => { const m = G.apply(g.H, x, y); return G.apply(g.Hi, m[0], m[1]); }) };
});
console.log(JSON.stringify(out));
"""


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_js_geometry_matches_python():
    res = subprocess.run([NODE, "-e", RUNNER, str(LAYOUT_JS),
                          json.dumps({"quads": QUADS, "points": POINTS})],
                         capture_output=True, text=True, timeout=60, check=True)
    for q, js in zip(QUADS, json.loads(res.stdout)):
        b = Block(quad=[tuple(p) for p in q])
        assert tuple(js["size"]) == b.size()
        assert js["convex"] is True
        np.testing.assert_allclose(js["mapped"], b.transform()(np.array(POINTS, float)),
                                   atol=1e-6)
        np.testing.assert_allclose(js["back"], POINTS, atol=1e-6)
