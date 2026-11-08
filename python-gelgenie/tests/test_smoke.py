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

import subprocess
import sys


def test_ssp_package_imports():
    import ssp
    assert isinstance(ssp.__version__, str) and ssp.__version__


def test_cli_group_imports():
    from ssp.cli import cli
    assert callable(cli)


def test_console_script_runs():
    """The `ssp-typing` console script declared in setup.py must be callable."""
    result = subprocess.run(
        [sys.executable, "-m", "ssp.cli", "--version"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "ssp" in result.stdout.lower() or result.stdout.strip()
