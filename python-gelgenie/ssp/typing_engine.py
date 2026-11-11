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

SSP interpretation: lane positivity pattern -> allele-pair calls.

Olerup-style scoring. For each locus, every allele token appearing in the
tray's Specificities cells is a candidate. A genotype is an unordered pair of
candidates (homozygous pairs allowed); it *predicts* a lane positive when the
lane's specificity set contains either allele. The best genotypes are those
minimizing contradictions with the observed pattern over valid lanes
(lanes without a control band are skipped and reported as invalid).

All minimum-score pairs are reported: equal-scoring pairs are genuine kit
ambiguities (e.g. alleles that no lane on the tray distinguishes) and must
survive into the report rather than being resolved arbitrarily.

Limitations (v1):
  * lanes are scored within their tray section's locus only; cross-locus
    specificities (e.g. an A*26:04 mention inside a C-section lane) are
    ignored for scoring;
  * intensity is not used - a size-matched band is positive, full stop
    (weak-band comments on the sheet are advisory only);
  * candidates come from the tray itself; alleles absent from the sheet
    cannot be called.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from ssp.analysis import GelAnalysis
from ssp.trays import Tray
from ssp.trays.specificities import SpecificitySet, family_of, parse_specificities


class _FastSpec:
    """O(1)-membership mirror of :class:`SpecificitySet` for scoring loops."""

    __slots__ = ("all_except", "alleles", "families", "fams_of_alleles",
                 "exc", "exc_fams")

    def __init__(self, spec: SpecificitySet) -> None:
        self.all_except = spec.all_except is not None
        if self.all_except:
            self.exc: FrozenSet[str] = frozenset(spec.all_except or ())
            self.exc_fams: FrozenSet[str] = frozenset(
                family_of(x) for x in self.exc if ":" in x)
            self.alleles = frozenset()
            self.families = frozenset()
            self.fams_of_alleles = frozenset()
        else:
            self.alleles = frozenset(spec.alleles or ())
            self.families = frozenset(t for t in self.alleles if ":" not in t)
            self.fams_of_alleles = frozenset(
                family_of(t) for t in self.alleles if ":" in t)
            self.exc = frozenset()
            self.exc_fams = frozenset()

    def tokens(self) -> FrozenSet[str]:
        """Explicit allele tokens (empty for all-except specs, which cannot
        enumerate the allele universe they cover)."""
        return self.alleles

    def contains(self, token: str) -> bool:
        if self.all_except:
            if token in self.exc or family_of(token) in self.exc:
                return False
            if ":" not in token and token in self.exc_fams:
                return False  # family with excluded members: conservative no
            return True
        if token in self.alleles:
            return True
        if ":" in token:
            return family_of(token) in self.families
        return token in self.fams_of_alleles


@dataclass
class LocusResult:
    locus: str
    positives: List[int]
    negatives: List[int]
    invalid: List[int]
    candidates: List[str]
    best_pairs: List[Tuple[str, str]]
    score: Optional[int]
    allele_options: Optional[List[List[str]]]  # Cartesian form when applicable
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "locus": self.locus,
            "positives": self.positives,
            "negatives": self.negatives,
            "invalid": self.invalid,
            "n_candidates": len(self.candidates),
            "best_pairs": [list(p) for p in self.best_pairs],
            "score": self.score,
            "allele_options": self.allele_options,
            "warnings": self.warnings,
        }

    def genotype_text(self) -> str:
        if not self.best_pairs:
            return "no call"
        if self.allele_options:
            return " + ".join("/".join(opts) for opts in self.allele_options)
        shown = "; ".join(f"{a} + {b}" for a, b in self.best_pairs[:6])
        if len(self.best_pairs) > 6:
            shown += f"; ... (+{len(self.best_pairs) - 6} more)"
        return shown


@dataclass
class TypingResult:
    loci: Dict[str, LocusResult]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "warnings": self.warnings,
            "loci": {k: v.to_dict() for k, v in self.loci.items()},
        }

    def summary_text(self) -> str:
        lines = []
        for locus, r in self.loci.items():
            inv = f"  invalid lanes: {r.invalid}" if r.invalid else ""
            lines.append(f"HLA-{locus}: {r.genotype_text()}   "
                         f"(score {r.score}; positives {r.positives}{inv})")
        for w in self.warnings:
            lines.append(f"WARNING: {w}")
        return "\n".join(lines)


def type_sample(ga: GelAnalysis, tray: Tray) -> TypingResult:
    """Interpret an analysed gel against its tray definition."""
    lane_by_mix = {l.mix: l for l in ga.lanes}
    loci: List[str] = []
    for tl in tray.lanes:
        if tl.locus not in loci:
            loci.append(tl.locus)

    results: Dict[str, LocusResult] = {}
    all_warnings: List[str] = []
    for locus in loci:
        mixes = sorted(tl.mix for tl in tray.lanes if tl.locus == locus)
        specs: Dict[int, _FastSpec] = {}
        valid: List[int] = []
        invalid: List[int] = []
        for m in mixes:
            tl = tray.lane(m)
            specs[m] = _FastSpec(parse_specificities(tl.specificities))
            lane = lane_by_mix.get(m)
            if lane is None or not lane.valid:
                invalid.append(m)
            else:
                valid.append(m)
        positives = [m for m in valid if lane_by_mix[m].positive]
        negatives = [m for m in valid if not lane_by_mix[m].positive]

        candidates = sorted({
            t for m in mixes for t in specs[m].tokens()
            if t.split("*", 1)[0] == locus
        })
        warnings: List[str] = []
        if lane_by_mix.keys() and any(m not in lane_by_mix for m in mixes):
            warnings.append(f"locus {locus}: some tray lanes missing from gel analysis")
        if not positives:
            warnings.append(f"locus {locus}: no positive lanes; typing unreliable")
        if not candidates:
            warnings.append(f"locus {locus}: no candidate alleles parsed from tray")

        p_set: FrozenSet[int] = frozenset(positives)
        cover: Dict[str, FrozenSet[int]] = {
            t: frozenset(m for m in valid if specs[m].contains(t))
            for t in candidates
        }

        best: List[Tuple[str, str]] = []
        best_score: Optional[int] = None
        for t1, t2 in combinations_with_replacement(candidates, 2):
            s = len((cover[t1] | cover[t2]) ^ p_set)
            if best_score is None or s < best_score:
                best_score, best = s, [(t1, t2)]
            elif s == best_score:
                best.append((t1, t2))

        allele_options: Optional[List[List[str]]] = None
        if best:
            o1 = sorted({p[0] for p in best})
            o2 = sorted({p[1] for p in best})
            cartesian = {(min(a, b), max(a, b)) for a in o1 for b in o2}
            if cartesian == set(best):
                allele_options = [o1, o2]

        results[locus] = LocusResult(
            locus=locus, positives=positives, negatives=negatives,
            invalid=invalid, candidates=candidates, best_pairs=best,
            score=best_score, allele_options=allele_options, warnings=warnings,
        )
        all_warnings.extend(warnings)

    return TypingResult(loci=results, warnings=all_warnings)
