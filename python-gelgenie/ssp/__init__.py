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

"""Standalone SSP HLA typing toolkit.

Derives from GelGenie (https://github.com/mattaq31/GelGenie). Provides gel-doc
image segmentation and downstream band analysis without requiring QuPath.

Submodules (added incrementally):
    ssp.preprocessing  - image normalization / padding helpers
    ssp.inference      - ONNX Runtime segmentation model wrapper
    ssp.bands          - mask -> connected-component bands -> lanes
    ssp.cli            - `ssp-typing` command line entry point
"""

__version__ = "0.1.0"
