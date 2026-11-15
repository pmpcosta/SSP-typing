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

from click.testing import CliRunner

import ssp.cli as cli_mod
from ssp.cli import cli


def test_serve_missing_model_clean_error():
    runner = CliRunner()
    result = runner.invoke(cli, [
        'serve', '--model', '/nonexistent/model.onnx', '--no-browser'])
    assert result.exit_code != 0
    assert 'model not found' in result.output
    assert 'Traceback' not in result.output


def test_serve_resolves_options(tmp_path, monkeypatch):
    model = tmp_path / 'm.onnx'
    model.write_bytes(b'x')
    captured = {}
    monkeypatch.setattr(cli_mod, '_run_server',
                        lambda config, host, port: captured.update(
                            config=config, host=host, port=port))
    runner = CliRunner()
    result = runner.invoke(cli, [
        'serve', '-m', str(model), '--port', '9999', '--host', '127.0.0.1',
        '--data-dir', str(tmp_path / 'data'), '--no-browser'])
    assert result.exit_code == 0, result.output
    assert captured['config'].model_path == str(model)
    assert captured['config'].data_dir == str(tmp_path / 'data')
    assert captured['host'] == '127.0.0.1'
    assert captured['port'] == 9999


def test_serve_env_model_and_default_data_dir(tmp_path, monkeypatch):
    model = tmp_path / 'env.onnx'
    model.write_bytes(b'x')
    monkeypatch.setenv('SSP_MODEL', str(model))
    captured = {}
    monkeypatch.setattr(cli_mod, '_run_server',
                        lambda config, host, port: captured.update(
                            config=config, host=host, port=port))
    runner = CliRunner()
    result = runner.invoke(cli, ['serve', '--no-browser'])
    assert result.exit_code == 0, result.output
    assert captured['config'].model_path == str(model)
    assert captured['config'].data_dir.endswith('.local/share/ssptyping')
