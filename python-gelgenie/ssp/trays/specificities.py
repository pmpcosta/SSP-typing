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

Parsing of tray-sheet ``Specificities`` cells into normalized allele tokens.

Token grammar: ``<LOCUS>*<family>[:<allele>]`` where LOCUS is 1-3 uppercase
letters plus an optional digit (A, B, C, DRB1, DQB1...), family is exactly
2 digits (all HLA families are; a 3-digit first field is invalid input and
may misparse), allele is 2-3 digits. Input is case-normalized to uppercase.
A token without ``:<allele>`` is a *family token* (e.g. ``B*13``
from Bw4/Bw6 public-epitope lists), meaning "any allele of that family".

Handles the real-world sheet quirks catalogued from FLClasseI.xlsx:
  * slash lists and ranges: ``A*01:01/02``, ``A*02:01-19``
  * chained ranges: ``B*15:13-16-17`` -> 13..17 (consecutive spans)
  * reversed ranges (typos): ``32-25`` -> swapped
  * null/expression suffixes are stripped (``09-11N`` -> 09,10,11): SSP
    detects DNA, so expression status does not change the band pattern
  * stray apostrophes: ``C*'08:01-03``
  * missing colon typos: ``B*5201-02`` -> B*52:01, B*52:02
  * ``All, except <list>`` cells -> :class:`SpecificitySet` with ``all_except``

Contract note: ``SpecificitySet.contains`` does NOT filter by locus; callers
(the typing engine) must only query candidates of the lane's own locus.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Set

_ALL_EXCEPT = re.compile(r"^\s*all\b.*?\bexcept\b", re.IGNORECASE | re.DOTALL)
_BARE_ALL = re.compile(r"^\s*all\s*$", re.IGNORECASE)
# find every locus*family[:allele-expr] pattern, ignoring surrounding prose
# (e.g. mix 92: 'C*15:05, other C*15 (weak band)')
_GROUP = re.compile(
    r"([A-Z]{1,3}[0-9]?)\*'?\s*(\d{2})(?::([0-9A-Za-z/\-]+)|([0-9][0-9A-Za-z/\-]*))?"
)
_NUMBER = re.compile(r"^(\d{1,3})[A-Za-z]?$")  # trailing expression suffix stripped


def family_of(token: str) -> str:
    """``A*31:01`` -> ``A*31``; family tokens map to themselves."""
    return token.split(":", 1)[0]


def _expand_expr(locus: str, family: str, expr: str) -> Set[str]:
    """Expand the allele-field expression of one group into tokens."""
    tokens: Set[str] = set()
    if not expr:
        return {f"{locus}*{family}"}  # family token
    for part in expr.split("/"):
        part = part.strip().rstrip(".,;")
        if not part:
            continue
        nums: List[int] = []
        for seg in part.split("-"):
            m = _NUMBER.match(seg.strip())
            if m:
                nums.append(int(m.group(1)))
        if not nums:
            continue  # tolerate junk parts
        if len(nums) == 1:
            tokens.add(f"{locus}*{family}:{nums[0]:02d}")
        else:
            for lo, hi in zip(nums, nums[1:]):  # chained spans a-b-c -> a..b, b..c
                if lo > hi:  # reversed-range typo
                    lo, hi = hi, lo
                for n in range(lo, hi + 1):
                    tokens.add(f"{locus}*{family}:{n:02d}")
    return tokens


def _find_groups(text: str) -> Set[str]:
    """Extract tokens from every allele-group pattern found in ``text``."""
    tokens: Set[str] = set()
    for m in _GROUP.finditer(text):
        locus, family = m.group(1), m.group(2)
        expr = (m.group(3) or m.group(4) or "").strip()
        tokens |= _expand_expr(locus, family, expr)
    return tokens


def parse_specificities(text: str) -> "SpecificitySet":
    """Parse one ``Specificities`` cell.

    Returns a :class:`SpecificitySet` with either ``alleles`` (explicit token
    set) or ``all_except`` (``All, except ...`` cells: everything minus the
    expanded exception tokens).    Prose around allele groups (e.g. ``(weak
    band)``, ``other C*15``) is ignored except that bare ``locus*family``
    mentions still yield family tokens. ``All`` alone covers everything;
    ``All, except <list>`` with an unparseable exception list raises
    ``ValueError`` rather than silently covering everything.
    """
    text = (text or "").strip().upper()
    if not text:
        return SpecificitySet()

    if _BARE_ALL.match(text):
        return SpecificitySet(all_except=frozenset())

    m = _ALL_EXCEPT.match(text)
    if m:
        exc = _find_groups(text[m.end():])
        if not exc:
            raise ValueError(
                f"'All, except ...' cell has no parseable allele list: {text!r}")
        return SpecificitySet(all_except=frozenset(exc))

    return SpecificitySet(alleles=frozenset(_find_groups(text)))


@dataclass(frozen=True)
class SpecificitySet:
    """Allele coverage of one tray lane, in one of two forms."""

    alleles: Optional[frozenset] = None      # explicit tokens
    all_except: Optional[frozenset] = None   # 'All, except ...' form

    def __bool__(self) -> bool:
        return bool(self.alleles) or self.all_except is not None

    def tokens(self) -> frozenset:
        """Explicit tokens (empty for all_except sets)."""
        return self.alleles or frozenset()

    def contains(self, token: str) -> bool:
        """Would a sample carrying ``token`` make this lane positive?

        Family/allele cross-matching: a family token in the lane covers any
        allele of the family, and an allele-token lane covers its family
        candidate. For ``all_except`` lanes, excluded tokens (and families
        with any excluded member — conservative) are negative, everything
        else positive.
        """
        if self.all_except is not None:
            if token in self.all_except:
                return False
            if family_of(token) in self.all_except:
                return False
            if ":" not in token and any(
                x.startswith(token + ":") for x in self.all_except
            ):
                return False
            return True
        if not self.alleles:
            return False
        if token in self.alleles:
            return True
        if family_of(token) in self.alleles:
            return True
        if ":" not in token:  # family candidate vs allele-level lane
            return any(a.startswith(token + ":") for a in self.alleles)
        return False
