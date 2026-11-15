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

Human-readable typing report shared by the CLI and the web app.
"""

from __future__ import annotations

from typing import List

from ssp.analysis import GelAnalysis
from ssp.typing_engine import TypingResult


def build_report(image_name: str, tray_name: str,
                 ga: GelAnalysis, result: TypingResult) -> str:
    """Render the full text report (summary + per-lane calls + warnings)."""
    lines: List[str] = [
        f"SSP typing: {image_name} vs tray {tray_name}",
        f"calibrated slope: {ga.slope_b:.1f} px/ln(bp)   panels: {ga.panels}",
        "",
        result.summary_text(),
        "",
        "per-lane calls (positive, invalid, and anomalous lanes):",
    ]
    for lane in ga.lanes:
        if not lane.valid:
            lines.append(f"  mix {lane.mix:3d}: INVALID (no control band)")
            continue
        if not (lane.positive or any(c.kind in ("unexplained", "hmw")
                                     for c in lane.calls)):
            continue
        bits = []
        for c in lane.calls:
            if c.kind == "control":
                continue
            bp = f"{c.bp_est:.0f}bp" if c.bp_est is not None else "?bp"
            extra = f"~{c.matched_bp}" if c.matched_bp else ""
            bits.append(f"{c.band.centroid_y:.0f}px={bp}{extra} [{c.kind}]")
        lines.append(f"  mix {lane.mix:3d}: {'; '.join(bits)}")
    for w in ga.warnings:
        lines.append(f"WARNING: {w}")
    return "\n".join(lines) + "\n"
