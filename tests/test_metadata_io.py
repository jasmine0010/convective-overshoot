import json

import numpy as np
import pytest

from diagnostics import get_boussinesq_cz_reference_gradient
from metadata_io import load_run_metadata, write_json_atomic


def test_atomic_json_writer_replaces_valid_metadata(tmp_path):
    path = tmp_path / 'info.json'
    path.write_text('{"old": true}')

    write_json_atomic(path, {'value': 1.5, 'flag': bool(np.bool_(True))})

    assert json.loads(path.read_text()) == {'value': 1.5, 'flag': True}


def test_atomic_json_writer_preserves_original_on_serialization_failure(tmp_path):
    path = tmp_path / 'info.json'
    original = '{"preserved": true}'
    path.write_text(original)

    with pytest.raises(TypeError):
        write_json_atomic(path, {'bad': np.bool_(True)})

    assert path.read_text() == original
    assert list(tmp_path.iterdir()) == [path]



def test_load_run_metadata_returns_independent_raw_and_normalized_views(tmp_path):
    path = tmp_path / 'info.json'
    source = {
        'run_id': 'current',
        'params': {'Lz': 3.0, 'g': 4.0, 'alpha': 0.5, 'custom': {'value': 4}},
        'background': {'g': 2.0, 'alpha': 0.25, 'kappa_C': 0.004},
        'timing': {
            'release_time_C1': 10.0,
            'release_time_C2': 20.0,
            'release_time_C3': 30.0,
        },
        'depths': {'var_grad': 0.2, 'KE': 0.1},
    }
    original = json.dumps(source, indent=2)
    path.write_text(original)

    info, normalized = load_run_metadata(path)

    assert path.read_text() == original
    assert info == source
    assert normalized['params']['g'] == 4.0
    assert normalized['params']['alpha'] == 0.5
    assert normalized['background']['g'] == 2.0
    assert normalized['background']['alpha'] == 0.25
    assert normalized['background']['kappa_C'] == 0.004
    assert normalized['timing'] == source['timing']
    assert normalized['depths'] == source['depths']

    info['params']['custom']['value'] = 99
    assert normalized['params']['custom']['value'] == 4
    normalized['depths']['KE'] = 99
    assert info['depths']['KE'] == 0.1
    assert json.loads(path.read_text()) == source


def test_load_run_metadata_resolves_parameter_only_physics_and_release_defaults(tmp_path):
    path = tmp_path / 'info.json'
    source = {
        'params': {'Lz': 3.0, 'g': 4.0, 'alpha': 0.5, 'kappa_C': 0.007},
        'background': {'N2_cz': -2.0},
        'depths': {'var_grad': 0.2},
    }
    path.write_text(json.dumps(source))

    info, normalized = load_run_metadata(path)

    assert info == source
    assert 'timing' not in info
    assert 'g' not in info['background']
    assert 'alpha' not in info['background']
    assert 'kappa_C' not in info['background']
    assert normalized['background']['g'] == 4.0
    assert normalized['background']['alpha'] == 0.5
    assert normalized['background']['kappa_C'] == 0.007
    assert normalized['params']['g'] == 4.0
    assert normalized['params']['alpha'] == 0.5
    assert normalized['timing'] == {
        'release_time_C1': 0.0,
        'release_time_C2': None,
        'release_time_C3': None,
    }
    assert normalized['depths'] == source['depths']
    assert get_boussinesq_cz_reference_gradient(normalized['background']) == -1.0


def test_load_run_metadata_populates_normalized_params_from_background(tmp_path):
    path = tmp_path / 'info.json'
    source = {
        'params': {},
        'background': {'g': 3.0, 'alpha': 0.25},
        'timing': {'release_time_C1': 8.0, 'release_time_C2': 9.0, 'release_time_C3': 10.0},
    }
    path.write_text(json.dumps(source))

    info, normalized = load_run_metadata(path)

    assert info == source
    assert normalized['params']['g'] == 3.0
    assert normalized['params']['alpha'] == 0.25
    assert normalized['background']['g'] == 3.0
    assert normalized['background']['alpha'] == 0.25
    assert normalized['timing'] == source['timing']


def test_raw_metadata_round_trip_does_not_persist_inferred_defaults(tmp_path):
    path = tmp_path / 'info.json'
    source = {'params': {'Lz': 3.0}, 'background': {'N2_cz': -1.0}, 'depths': {'KE': 0.1}}
    path.write_text(json.dumps(source))

    info, normalized = load_run_metadata(path)
    assert normalized['params']['g'] == 1.0
    assert normalized['params']['alpha'] == 1.0
    assert normalized['background']['g'] == 1.0
    assert normalized['background']['alpha'] == 1.0
    assert normalized['background']['kappa_C'] == 0.001
    assert normalized['timing']['release_time_C1'] == 0.0
    assert normalized['timing']['release_time_C2'] is None
    assert normalized['timing']['release_time_C3'] is None

    info['derived'] = {'value': 2.0}
    write_json_atomic(path, info)

    saved = json.loads(path.read_text())
    assert saved == {**source, 'derived': {'value': 2.0}}
    assert 'g' not in saved['params']
    assert 'alpha' not in saved['params']
    assert 'g' not in saved['background']
    assert 'alpha' not in saved['background']
    assert 'kappa_C' not in saved['background']
    assert 'timing' not in saved


@pytest.mark.parametrize('source, message', [
    ([], '^Metadata must be a JSON object$'),
    ({'background': {}}, "^Metadata requires 'params' object$"),
    ({'params': {}}, "^Metadata requires 'background' object$"),
    ({'params': [], 'background': {}}, "^Metadata requires 'params' object$"),
    ({'params': {}, 'background': []}, "^Metadata requires 'background' object$"),
    ({'params': {}, 'background': {}, 'timing': []}, "^Metadata 'timing' must be an object or null$"),
])
def test_load_run_metadata_rejects_invalid_shared_structure(tmp_path, source, message):
    path = tmp_path / 'info.json'
    path.write_text(json.dumps(source))

    with pytest.raises(ValueError, match=message):
        load_run_metadata(path)
