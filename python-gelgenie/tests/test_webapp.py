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

import imageio.v2 as imageio
import numpy as np
import pytest
from fastapi.testclient import TestClient

import ssp.webapp.sessions as sessions_mod
from ssp.inference import GelGenieONNX
from ssp.trays import Tray, TrayLane
from ssp.webapp import AppConfig, create_app


class _FakeIO:
    def __init__(self, name):
        self.name = name


class FakeSession:
    """foreground (class 1) where input > 0.5 (same fake as test_cli_type)"""

    def get_inputs(self):
        return [_FakeIO('input')]

    def get_outputs(self):
        return [_FakeIO('output')]

    def run(self, out_names, feed):
        x = feed['input']
        b, c, h, w = x.shape
        out = np.zeros((b, 2, h, w), dtype=np.float32)
        fg = (x[:, 0] > 0.5).astype(np.float32)
        out[:, 1] = fg * 10.0 - 5.0
        out[:, 0] = (1.0 - fg) * 10.0 - 5.0
        return [out]


def _fake_model_factory(model_path=None, **kwargs):
    return GelGenieONNX(session=FakeSession())


def _synth_image_bytes():
    """5 lanes (x=25..225, pitch 50); control y~50 everywhere; product y~30 in
    lanes 1-3 only. Controls must outnumber products (as on real gels) so the
    densest-window control picker stays unambiguous."""
    img = np.zeros((100, 250), dtype=np.uint8)
    for i, x in enumerate((25, 75, 125, 175, 225)):
        img[47:54, x - 10:x + 10] = 255
        if i < 3:
            img[27:34, x - 10:x + 10] = 255
    from io import BytesIO
    buf = BytesIO()
    imageio.imwrite(buf, img, format='png')
    return buf.getvalue()


def _synth_tray_json():
    tray = Tray(name='synth', lanes=[
        TrayLane(mix=1, locus='A', product_bp=[330], control_bp=256,
                 serotypes='1', specificities='A*01:01'),
        TrayLane(mix=2, locus='A', product_bp=[330], control_bp=256,
                 serotypes='2', specificities='A*02:01'),
        TrayLane(mix=3, locus='A', product_bp=[330], control_bp=256,
                 serotypes='1, 2', specificities='A*01:01, A*02:01'),
        TrayLane(mix=4, locus='A', product_bp=[330], control_bp=256,
                 serotypes='3', specificities='A*03:01'),
        TrayLane(mix=5, locus='A', product_bp=[330], control_bp=256,
                 serotypes='11', specificities='A*11:01'),
    ])
    return json.dumps(tray.to_dict()).encode()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(sessions_mod, 'GelGenieONNX', _fake_model_factory)
    model = tmp_path / 'model.onnx'
    model.write_bytes(b'x')
    cfg = AppConfig(model_path=str(model), data_dir=str(tmp_path / 'data'))
    app = create_app(cfg)
    with TestClient(app) as c:
        yield c


def _create(client, tray_filename='tray.json', tray_bytes=None):
    r = client.post('/api/sessions', files={
        'image': ('gel.png', _synth_image_bytes(), 'image/png'),
        'tray': (tray_filename, tray_bytes or _synth_tray_json(), 'application/json'),
    })
    assert r.status_code == 200, r.text
    return r.json()


# ---------- session lifecycle ----------

def test_create_session_runs_pipeline(client):
    s = _create(client)
    assert s['image_name'] == 'gel.png'
    assert s['tray_name'] == 'synth'
    assert s['image'] == {'width': 250, 'height': 100}
    assert s['params']['min_area'] == 5
    assert len(s['bands']) == 8  # 5 control + 3 product
    t = s['typing']['loci']['A']
    assert t['score'] == 0
    assert t['positives'] == [1, 2, 3]
    assert t['best_pairs'] == [['A*01:01', 'A*02:01']]
    assert s['edits'] == {'lane_overrides': [], 'deleted_bands': [], 'added_bands': [],
                          'band_kinds': []}
    assert s['defaults'] == {'match_tol': 0.25, 'lane_gap': 20.0}
    # tray summary for the UI results table
    assert [l['mix'] for l in s['tray']['lanes']] == [1, 2, 3, 4, 5]
    assert s['tray']['lanes'][0]['specificities'] == 'A*01:01'


def test_get_session_and_list(client):
    s = _create(client)
    sid = s['id']
    r = client.get(f'/api/sessions/{sid}')
    assert r.status_code == 200 and r.json()['id'] == sid
    r = client.get('/api/sessions')
    assert [e['id'] for e in r.json()] == [sid]
    assert r.json()[0]['image_name'] == 'gel.png'


def test_get_image_and_mask(client):
    s = _create(client)
    sid = s['id']
    r = client.get(f'/api/sessions/{sid}/image')
    assert r.status_code == 200
    assert r.content == _synth_image_bytes()  # original bytes preserved
    r = client.get(f'/api/sessions/{sid}/mask')
    assert r.status_code == 200 and r.headers['content-type'] == 'image/png'


def test_missing_session_404(client):
    assert client.get('/api/sessions/nope').status_code == 404


def test_bad_tray_clean_400(client):
    bad = Tray(name='bad', lanes=[
        TrayLane(mix=1, locus='A', product_bp=[330], control_bp=256,
                 serotypes='', specificities='A*01:01'),
        TrayLane(mix=3, locus='A', product_bp=[330], control_bp=256,
                 serotypes='', specificities='A*03:01'),
    ])
    bad_bytes = json.dumps(bad.to_dict()).encode()
    r = client.post('/api/sessions', files={
        'image': ('gel.png', _synth_image_bytes(), 'image/png'),
        'tray': ('bad.json', bad_bytes, 'application/json'),
    })
    assert r.status_code == 400
    assert 'consecutive' in r.json()['detail']


# ---------- edits ----------

def test_lane_override_edit(client):
    s = _create(client)
    sid = s['id']
    r = client.post(f'/api/sessions/{sid}/edits', json={
        'lane_overrides': [{'mix': 2, 'positive': False}],
        'deleted_bands': [], 'added_bands': [],
    })
    assert r.status_code == 200, r.text
    s2 = r.json()
    a = s2['typing']['loci']['A']
    assert a['positives'] == [1, 3]
    assert ['A*01:01', 'A*02:01'] not in a['best_pairs']
    lane2 = [l for l in s2['analysis']['lanes'] if l['mix'] == 2][0]
    assert lane2['positive'] is False
    assert s2['edits']['lane_overrides'] == [{'mix': 2, 'positive': False}]


def test_band_delete_edit(client):
    s = _create(client)
    sid = s['id']
    # delete the product band in lane 3 (y~30, x~125)
    prod3 = [b for b in s['bands'] if b['centroid_y'] < 40 and 115 < b['centroid_x'] < 135]
    assert len(prod3) == 1
    r = client.post(f'/api/sessions/{sid}/edits', json={
        'lane_overrides': [], 'deleted_bands': [prod3[0]['id']], 'added_bands': [],
    })
    s2 = r.json()
    a = s2['typing']['loci']['A']
    assert a['positives'] == [1, 2]
    assert a['score'] == 1  # no pair explains the pattern perfectly anymore
    assert len(s2['bands']) == 8  # raw bands preserved; deletion is an edit


def test_band_add_edit(client):
    s = _create(client)
    sid = s['id']
    r = client.post(f'/api/sessions/{sid}/edits', json={
        'lane_overrides': [], 'deleted_bands': [],
        'added_bands': [{'x': 175.0, 'y': 30.0}],  # lane 4 product position
    })
    s2 = r.json()
    a = s2['typing']['loci']['A']
    assert a['positives'] == [1, 2, 3, 4]
    assert a['score'] == 1


def test_match_tol_update(client):
    s = _create(client)
    sid = s['id']
    r = client.post(f'/api/sessions/{sid}/edits', json={
        'lane_overrides': [], 'deleted_bands': [], 'added_bands': [],
        'match_tol': 0.001,  # product sizes can no longer match
    })
    s2 = r.json()
    assert s2['params']['match_tol'] == 0.001
    assert s2['typing']['loci']['A']['positives'] == []


def test_edit_history_is_audited(client, tmp_path):
    s = _create(client)
    sid = s['id']
    for _ in range(2):
        client.post(f'/api/sessions/{sid}/edits', json={
            'lane_overrides': [{'mix': 2, 'positive': False}],
            'deleted_bands': [], 'added_bands': [],
        })
    hist = tmp_path / 'data' / 'sessions' / sid / 'edits_history.jsonl'
    lines = hist.read_text().strip().splitlines()
    assert len(lines) == 2
    entry = json.loads(lines[0])
    assert 'ts' in entry and entry['edits']['lane_overrides'][0]['mix'] == 2


# ---------- persistence ----------

def test_session_survives_app_restart(client, tmp_path, monkeypatch):
    s = _create(client)
    sid = s['id']
    client.post(f'/api/sessions/{sid}/edits', json={
        'lane_overrides': [{'mix': 2, 'positive': False}],
        'deleted_bands': [], 'added_bands': [],
    })
    # fresh app over the same data dir
    monkeypatch.setattr(sessions_mod, 'GelGenieONNX', _fake_model_factory)
    cfg = AppConfig(model_path=str(tmp_path / 'model.onnx'),
                    data_dir=str(tmp_path / 'data'))
    with TestClient(create_app(cfg)) as c2:
        r = c2.get(f'/api/sessions/{sid}')
        assert r.status_code == 200
        s2 = r.json()
        assert s2['typing']['loci']['A']['positives'] == [1, 3]
        assert s2['edits']['lane_overrides'] == [{'mix': 2, 'positive': False}]
        listed = c2.get('/api/sessions').json()
        assert [e['id'] for e in listed] == [sid]


# ---------- tray store ----------

def test_tray_store_upload_list_and_reuse(client):
    r = client.post('/api/trays', files={
        'tray': ('synthtray.json', _synth_tray_json(), 'application/json')})
    assert r.status_code == 200
    names = [t['name'] for t in client.get('/api/trays').json()]
    assert 'synthtray' in names
    # create a session referencing the stored tray by name (no file upload)
    r = client.post('/api/sessions', data={'tray_name': 'synthtray'}, files={
        'image': ('gel.png', _synth_image_bytes(), 'image/png'),
    })
    assert r.status_code == 200, r.text
    assert r.json()['typing']['loci']['A']['best_pairs'] == [['A*01:01', 'A*02:01']]


def test_session_without_tray_400(client):
    r = client.post('/api/sessions', files={
        'image': ('gel.png', _synth_image_bytes(), 'image/png')})
    assert r.status_code == 400


# ---------- uploads that broke in real use (regression tests) ----------

def test_xlsx_tray_upload_via_api(client, tmp_path):
    # openpyxl validates by extension: the store must keep a .xlsx suffix
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.append(['HLA-A*'])
    for mix, spec in [(1, 'A*01:01'), (2, 'A*02:01'),
                      (3, 'A*01:01, A*02:01'), (4, 'A*03:01'), (5, 'A*11:01')]:
        ws.append([mix, 330, 256, '', spec, ''])
    p = tmp_path / 'vendor.xlsx'
    wb.save(p)
    r = client.post('/api/sessions', files={
        'image': ('gel.png', _synth_image_bytes(), 'image/png'),
        'tray': ('vendor.xlsx', p.read_bytes(),
                 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'),
    })
    assert r.status_code == 200, r.text
    s = r.json()
    assert s['typing']['loci']['A']['best_pairs'] == [['A*01:01', 'A*02:01']]
    # the xlsx arrived normalized into the tray store
    assert 'vendor' in [t['name'] for t in client.get('/api/trays').json()]


def test_garbage_tray_upload_400(client):
    r = client.post('/api/sessions', files={
        'image': ('gel.png', _synth_image_bytes(), 'image/png'),
        'tray': ('broken.xlsx', b'not a spreadsheet at all', 'application/octet-stream'),
    })
    assert r.status_code == 400
    assert 'could not parse tray' in r.json()['detail']


def test_garbage_image_upload_400(client):
    r = client.post('/api/sessions', files={
        'image': ('gel.png', b'not an image', 'image/png'),
        'tray': ('tray.json', _synth_tray_json(), 'application/json'),
    })
    assert r.status_code == 400
    assert 'could not read image' in r.json()['detail']
    # failed session left no orphan on disk
    assert client.get('/api/sessions').json() == []


# ---------- downloads ----------

def test_typing_json_and_report_downloads(client):
    s = _create(client)
    sid = s['id']
    r = client.get(f'/api/sessions/{sid}/typing.json')
    assert r.status_code == 200
    payload = r.json()
    assert payload['typing']['loci']['A']['best_pairs'] == [['A*01:01', 'A*02:01']]
    r = client.get(f'/api/sessions/{sid}/report.txt')
    assert r.status_code == 200
    text = r.text
    assert 'HLA-A' in text and 'A*01:01 + A*02:01' in text
    assert 'attachment' in r.headers.get('content-disposition', '')


# ---------- static UI ----------

def test_ui_assets_served(client):
    r = client.get('/ui/')
    assert r.status_code == 200
    assert 'SSP Typing' in r.text and 'app.js' in r.text
    for asset in ('app.js', 'style.css'):
        r = client.get(f'/ui/{asset}')
        assert r.status_code == 200 and len(r.content) > 100


def test_root_redirects_to_ui(client):
    r = client.get('/', follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers['location'].endswith('/ui/')


# ---------- Phase-3 review hardening ----------

def test_edits_reject_nan_and_bad_types(client):
    s = _create(client)
    sid = s['id']
    base = {'lane_overrides': [], 'deleted_bands': [], 'added_bands': []}
    # NaN/Inf would poison meta.json and brick every JSON response.
    # httpx refuses to serialize non-finite floats, so send raw JSON literals
    # (which Python's json.loads accepts, as would any buggy/malicious client)
    for lit in ('NaN', 'Infinity', '-Infinity'):
        raw = ('{"lane_overrides": [], "deleted_bands": [], "added_bands": [],'
               f' "match_tol": {lit}}}')
        r = client.post(f'/api/sessions/{sid}/edits', content=raw,
                        headers={'Content-Type': 'application/json'})
        assert r.status_code == 400, lit
    raw = ('{"lane_overrides": [], "deleted_bands": [],'
           ' "added_bands": [{"x": NaN, "y": 1}]}')
    r = client.post(f'/api/sessions/{sid}/edits', content=raw,
                    headers={'Content-Type': 'application/json'})
    assert r.status_code == 400
    # malformed types: clean 400s, not 500s
    for body in (
        {**base, 'match_tol': {'a': 1}},
        {**base, 'match_tol': 'abc'},
        {**base, 'deleted_bands': [None]},
        {'lane_overrides': [{'mix': None, 'positive': True}],
         'deleted_bands': [], 'added_bands': []},
        {'lane_overrides': [], 'deleted_bands': [],
         'added_bands': [{'x': 'q', 'y': 1}]},
    ):
        r = client.post(f'/api/sessions/{sid}/edits', json=body)
        assert r.status_code == 400, body
    # session still fully usable after the rejections
    r = client.get(f'/api/sessions/{sid}')
    assert r.status_code == 200
    assert r.json()['typing']['loci']['A']['score'] == 0


def test_synthetic_delete_ids_filtered(client):
    s = _create(client)
    sid = s['id']
    r = client.post(f'/api/sessions/{sid}/edits', json={
        'lane_overrides': [], 'deleted_bands': [],
        'added_bands': [{'x': 225.0, 'y': 30.0}]})   # lane 5 product position
    assert r.json()['typing']['loci']['A']['positives'] == [1, 2, 3, 5]
    # deleting by a synthetic id is a filtered no-op; the added band stays
    r2 = client.post(f'/api/sessions/{sid}/edits', json={
        'lane_overrides': [], 'deleted_bands': [1000000],
        'added_bands': [{'x': 225.0, 'y': 30.0}]})
    assert r2.status_code == 200
    assert r2.json()['edits']['deleted_bands'] == []
    assert r2.json()['typing']['loci']['A']['positives'] == [1, 2, 3, 5]
    # removing it properly via added_bands works
    r3 = client.post(f'/api/sessions/{sid}/edits', json={
        'lane_overrides': [], 'deleted_bands': [], 'added_bands': []})
    assert r3.json()['typing']['loci']['A']['positives'] == [1, 2, 3]


def test_corrupt_state_json_rebuilds(client, tmp_path):
    s = _create(client)
    sid = s['id']
    (tmp_path / 'data' / 'sessions' / sid / 'state.json').write_text('{corrupt')
    r = client.get(f'/api/sessions/{sid}')
    assert r.status_code == 200   # rebuilt from bands/tray/edits, not 500
    assert r.json()['typing']['loci']['A']['best_pairs'] == [['A*01:01', 'A*02:01']]


def test_image_media_type_allowlisted(client):
    r = client.post('/api/sessions', files={
        'image': ('gel.html', _synth_image_bytes(), 'text/html'),
        'tray': ('tray.json', _synth_tray_json(), 'application/json'),
    })
    assert r.status_code == 200
    sid = r.json()['id']
    r = client.get(f'/api/sessions/{sid}/image')
    assert r.status_code == 200
    assert r.headers['content-type'].startswith('application/octet-stream')
    assert r.headers['x-content-type-options'] == 'nosniff'


def test_tray_store_same_name_different_content_kept(client):
    def _tray_bytes(spec):
        return json.dumps(Tray(name='t', lanes=[
            TrayLane(mix=1, locus='A', product_bp=[330], control_bp=256,
                     serotypes='', specificities=spec)]).to_dict()).encode()
    t1, t2 = _tray_bytes('A*01:01'), _tray_bytes('A*02:01')
    up = lambda b: client.post('/api/trays',
                               files={'tray': ('t.json', b, 'application/json')})
    assert up(t1).status_code == 200
    assert up(t2).status_code == 200       # different content -> suffixed name
    assert up(t2).status_code == 200       # identical re-upload -> no new entry
    names = [t['name'] for t in client.get('/api/trays').json()]
    assert len(names) == 2 and 't' in names
    assert any(n.startswith('t-') for n in names)


def test_negative_image_session_detects_polarity(client):
    from io import BytesIO
    neg = 255 - imageio.imread(BytesIO(_synth_image_bytes()))
    buf = BytesIO()
    imageio.imwrite(buf, neg, format='png')
    r = client.post('/api/sessions', files={
        'image': ('neg.png', buf.getvalue(), 'image/png'),
        'tray': ('tray.json', _synth_tray_json(), 'application/json')})
    assert r.status_code == 200, r.text
    s = r.json()
    assert s['params']['invert'] is True and s['params']['invert_auto'] is True
    assert len(s['bands']) == 8
    assert s['typing']['loci']['A']['positives'] == [1, 2, 3]
    pos = _create(client)
    assert pos['params']['invert'] is False and pos['params']['invert_auto'] is True


def test_forced_polarity_is_recorded(client):
    r = client.post('/api/sessions', data={'invert': 'false'}, files={
        'image': ('gel.png', _synth_image_bytes(), 'image/png'),
        'tray': ('tray.json', _synth_tray_json(), 'application/json')})
    assert r.status_code == 200, r.text
    assert r.json()['params']['invert'] is False
    assert r.json()['params']['invert_auto'] is False


# ---------- bands classified by hand ----------

def _band_at(state, mix, y):
    lane = next(l for l in state['analysis']['lanes'] if l['mix'] == mix)
    return next(b for b in lane['bands'] if abs(b['centroid_y'] - y) < 5)


def test_band_kind_edit_retypes_and_round_trips(client):
    s = _create(client)
    sid = s['id']
    product = _band_at(s, 1, 30)      # lane 1's product
    r = client.post(f'/api/sessions/{sid}/edits', json={
        'band_kinds': [{'id': product['id'], 'kind': 'unexplained'}]})
    assert r.status_code == 200, r.text
    s2 = r.json()
    assert s2['typing']['loci']['A']['positives'] == [2, 3]
    b = _band_at(s2, 1, 30)
    assert (b['kind'], b['manual']) == ('unexplained', True)
    assert s2['edits']['band_kinds'] == [{'id': product['id'], 'kind': 'unexplained'}]
    # the edit survives a reload and is audited
    assert client.get(f'/api/sessions/{sid}').json()['edits'] == s2['edits']
    assert 'set by hand' in client.get(f'/api/sessions/{sid}/report.txt').text


def test_added_band_carries_its_own_kind(client):
    s = _create(client)
    r = client.post(f"/api/sessions/{s['id']}/edits", json={
        'added_bands': [{'x': 175.0, 'y': 15.0, 'kind': 'product'}]})  # lane 4
    assert r.status_code == 200, r.text
    s2 = r.json()
    assert s2['typing']['loci']['A']['positives'] == [1, 2, 3, 4]
    assert s2['edits']['added_bands'] == [{'x': 175.0, 'y': 15.0, 'kind': 'product'}]


def test_band_kind_edits_are_validated(client):
    s = _create(client)
    url = f"/api/sessions/{s['id']}/edits"
    bad = [{'band_kinds': [{'id': 1, 'kind': 'positive'}]},
           {'band_kinds': [{'id': 'x', 'kind': 'product'}]},
           {'band_kinds': [{'kind': 'product'}]},
           {'band_kinds': [{'id': 1_000_000, 'kind': 'product'}]},
           {'added_bands': [{'x': 1, 'y': 2, 'kind': 'nope'}]}]
    for body in bad:
        r = client.post(url, json=body)
        assert r.status_code == 400, body
    assert client.get(f"/api/sessions/{s['id']}").json()['edits']['band_kinds'] == []
