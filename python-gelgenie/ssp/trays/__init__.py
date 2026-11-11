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

SSP tray definitions: schema, Excel loader, and JSON round-trip.

The Excel format matches the vendor-style sheets used in the lab (e.g.
``FLClasseI.xlsx``): a single worksheet with three stacked sections
(``HLA-A*``, ``HLA-B*``, ``HLA-C*``), each introduced by a section header in
column A and a column-header row, then one row per mix::

    Mix (lane) | PCR product size (bp) | Control band size (bp) |
    Serotypes | Specificities | Comments

Loader tolerances (observed in real sheets):
  * missing control-bp cells (e.g. lane 69) -> inferred from the section's
    other lanes;
  * missing product-size cells (e.g. lane 64) -> empty list;
  * multi-size product cells like ``498, 508`` -> all integers extracted.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

_SECTION_LOCUS = {"HLA-A": "A", "HLA-B": "B", "HLA-C": "C"}
_DEFAULT_CONTROL_BP = {"A": 256, "B": 256, "C": 800}


@dataclass(frozen=True)
class TrayLane:
    """One mix / lane / well reaction of an SSP tray."""

    mix: int
    locus: str  # 'A', 'B' or 'C'
    product_bp: List[int]
    control_bp: int
    serotypes: str
    specificities: str
    comment: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "TrayLane":
        return cls(
            mix=int(d["mix"]),
            locus=str(d["locus"]),
            product_bp=[int(x) for x in d["product_bp"]],
            control_bp=int(d["control_bp"]),
            serotypes=str(d["serotypes"]),
            specificities=str(d["specificities"]),
            comment=str(d.get("comment", "")),
        )


@dataclass
class Tray:
    """A full tray definition: ordered lanes plus a name."""

    name: str
    lanes: List[TrayLane] = field(default_factory=list)

    def lane(self, mix: int) -> TrayLane:
        for l in self.lanes:
            if l.mix == mix:
                return l
        raise KeyError(f"mix {mix} not in tray {self.name!r}")

    def locus_lanes(self, locus: str) -> List[TrayLane]:
        return [l for l in self.lanes if l.locus == locus]

    def to_dict(self) -> Dict:
        return {"name": self.name, "lanes": [l.to_dict() for l in self.lanes]}

    @classmethod
    def from_dict(cls, d: Dict) -> "Tray":
        return cls(name=str(d["name"]),
                   lanes=[TrayLane.from_dict(x) for x in d["lanes"]])

    def save_json(self, path: Union[str, Path]) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> "Tray":
        return cls.from_dict(json.loads(Path(path).read_text()))


def _parse_product_sizes(value) -> List[int]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [int(value)]
    return [int(s) for s in re.findall(r"\d+", str(value))]


def load_tray_xlsx(path: Union[str, Path], name: Optional[str] = None,
                   sheet: Optional[str] = None) -> Tray:
    """Load a tray definition from a vendor-style Excel sheet.

    Parameters
    ----------
    path:
        ``.xlsx`` file with the section/mix layout described above.
    name:
        Tray name; defaults to the file stem.
    sheet:
        Worksheet name; defaults to the first sheet.
    """
    import openpyxl  # local import: only needed for xlsx loading

    path = Path(path)
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb[sheet] if sheet else wb[wb.sheetnames[0]]

    rows: List[Dict] = []
    locus: Optional[str] = None
    for row in ws.iter_rows(min_col=1, max_col=6):
        a, b, c, d, e, f = (cell.value for cell in row)
        if isinstance(a, str):
            head = a.strip()
            for prefix, loc in _SECTION_LOCUS.items():
                if head.upper().startswith(prefix):
                    locus = loc
            continue
        if isinstance(a, (int, float)) and locus is not None and int(a) >= 1:
            rows.append({
                "mix": int(a), "locus": locus,
                "product_bp": _parse_product_sizes(b),
                "control_bp": int(c) if c is not None else None,
                "serotypes": "" if d is None else str(d).strip(),
                "specificities": "" if e is None else str(e).strip(),
                "comment": "" if f is None else str(f).strip(),
            })

    if not rows:
        raise ValueError(f"no tray rows found in {path}")

    # infer missing control sizes from the rest of the section, else defaults
    by_locus: Dict[str, List[int]] = {}
    for r in rows:
        if r["control_bp"] is not None:
            by_locus.setdefault(r["locus"], []).append(r["control_bp"])
    for r in rows:
        if r["control_bp"] is None:
            peers = by_locus.get(r["locus"])
            if peers:
                r["control_bp"] = max(set(peers), key=peers.count)  # mode
            else:
                r["control_bp"] = _DEFAULT_CONTROL_BP[r["locus"]]

    lanes = [TrayLane(**r) for r in sorted(rows, key=lambda r: r["mix"])]
    dupes = len(lanes) - len({l.mix for l in lanes})
    if dupes:
        raise ValueError(f"duplicate mix numbers in {path}")
    return Tray(name=name or path.stem, lanes=lanes)
