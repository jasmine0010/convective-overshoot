import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path


def load_run_metadata(path):
    """Load run metadata with the compatibility defaults used by active drivers."""
    path = Path(path)
    with path.open(encoding='utf-8') as file:
        loaded = json.load(file)

    if not isinstance(loaded, dict):
        raise ValueError('Metadata must be a JSON object')

    normalized = deepcopy(loaded)
    for key in ('params', 'background'):
        if not isinstance(normalized.get(key), dict):
            raise ValueError(f'Metadata requires {key!r} object')

    params = normalized['params']
    background = normalized['background']
    timing = normalized.get('timing')
    if timing is None:
        timing = {}
    elif not isinstance(timing, dict):
        raise ValueError("Metadata 'timing' must be an object or null")
    normalized['timing'] = timing

    for key in ('g', 'alpha'):
        params.setdefault(key, background.get(key, 1.0))
        background.setdefault(key, params[key])
    background.setdefault('kappa_C', params.get('kappa_C', 0.001))
    timing.setdefault('release_time_C1', 0.0)
    timing.setdefault('release_time_C2', None)
    timing.setdefault('release_time_C3', None)
    return loaded, normalized


def write_json_atomic(path, data):
    """Serialize and atomically replace a JSON file."""
    path = Path(path)
    payload = json.dumps(data, indent=2, allow_nan=False)

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            dir=path.parent,
            prefix=f'.{path.name}.',
            suffix='.tmp',
            delete=False,
        ) as file:
            temp_path = Path(file.name)
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())

        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
