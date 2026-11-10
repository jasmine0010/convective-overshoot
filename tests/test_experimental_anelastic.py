import ast
import configparser
import importlib
import os
from pathlib import Path
import subprocess
import sys
import types
from types import SimpleNamespace
from unittest import mock

import metadata_io


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'experimental' / 'anelastic'


def import_solver_with_inert_dedalus(monkeypatch):
    dedalus = types.ModuleType('dedalus')
    public = types.ModuleType('dedalus.public')
    dedalus.public = public
    monkeypatch.setitem(sys.modules, 'dedalus', dedalus)
    monkeypatch.setitem(sys.modules, 'dedalus.public', public)
    monkeypatch.setattr(sys, 'dont_write_bytecode', True)
    sys.modules.pop('experimental.anelastic.solver', None)
    return importlib.import_module('experimental.anelastic.solver')


def test_package_layout_and_checkpoint_cleanup():
    assert (ROOT / 'experimental' / '__init__.py').read_bytes() == b''
    assert (PACKAGE / '__init__.py').read_bytes() == b''
    for path in ('solver.py', 'run.py', 'plot_legacy.py', 'diagnostics.py', 'README.md'):
        assert (PACKAGE / path).is_file()
    for path in ('anelastic.py', 'plot_anelastic.py', 'runs/anelastic_run.py'):
        assert not (ROOT / path).exists()
    for path in (
        '.ipynb_checkpoints/anelastic-checkpoint.py',
        '.ipynb_checkpoints/plot_anelastic-checkpoint.py',
        'runs/.ipynb_checkpoints/anelastic_run-checkpoint.py',
    ):
        assert not (ROOT / path).exists()
    assert not (PACKAGE / 'archive').exists()
    ast.parse((PACKAGE / 'plot_legacy.py').read_text())


def test_experimental_imports_are_inert(monkeypatch, tmp_path):
    import scipy.integrate

    def forbidden(*args, **kwargs):
        raise AssertionError('Import caused a side effect')

    for name in (
        'experimental.anelastic.diagnostics',
        'experimental.anelastic.run',
        'experimental.anelastic.solver',
        'experimental.anelastic',
        'experimental',
    ):
        sys.modules.pop(name, None)

    dedalus = types.ModuleType('dedalus')
    public = types.ModuleType('dedalus.public')
    dedalus.public = public
    monkeypatch.setitem(sys.modules, 'dedalus', dedalus)
    monkeypatch.setitem(sys.modules, 'dedalus.public', public)
    monkeypatch.setattr(sys, 'dont_write_bytecode', True)
    monkeypatch.setattr(subprocess, 'run', forbidden)
    monkeypatch.setattr(Path, 'mkdir', forbidden)
    monkeypatch.setattr(os, 'makedirs', forbidden)
    monkeypatch.setattr(metadata_io, 'write_json_atomic', forbidden)
    monkeypatch.chdir(tmp_path)

    before = set(tmp_path.rglob('*'))
    for name in (
        'experimental',
        'experimental.anelastic',
        'experimental.anelastic.solver',
        'experimental.anelastic.run',
        'experimental.anelastic.diagnostics',
    ):
        importlib.import_module(name)

    assert set(tmp_path.rglob('*')) == before
    assert 'experimental.anelastic.plot_legacy' not in sys.modules
    for path in (
        ROOT / 'experimental' / '__init__.py',
        PACKAGE / '__init__.py',
        PACKAGE / 'solver.py',
        PACKAGE / 'run.py',
        PACKAGE / 'diagnostics.py',
    ):
        assert 'matplotlib' not in path.read_text()


def test_runner_is_guarded_and_builds_mpi_solver_command(monkeypatch):
    runner = importlib.import_module('experimental.anelastic.run')
    tree = ast.parse((PACKAGE / 'run.py').read_text())
    assert any(isinstance(node, ast.If) and "__name__ == '__main__'" in ast.unparse(node.test) for node in tree.body)

    run_mock = mock.Mock()
    monkeypatch.setattr(runner.subprocess, 'run', run_mock)
    runner.main()

    run_mock.assert_called_once()
    call = run_mock.call_args
    command = call.args[0]
    assert command[:3] == ['mpiexec', '-n', '8']
    assert command[3:6] == [sys.executable, '-m', 'experimental.anelastic.solver']
    assert call.kwargs == {'check': True, 'cwd': ROOT}


def test_solver_paths_preserve_project_data_destinations(monkeypatch):
    solver = import_solver_with_inert_dedalus(monkeypatch)
    common = dict(run_id='sample', Ra=1e7, S=5, L_cz=1, Nx=256, Nz=128)

    run_path = solver.get_run_folder(SimpleNamespace(mode='run', **common))
    test_path = solver.get_run_folder(SimpleNamespace(mode='test', **common))

    assert run_path == ROOT / 'data' / 'runs' / 'sample_Ra1e+07_S5_Lcz1_256x128'
    assert test_path == ROOT / 'data' / 'tests' / 'sample_Ra1e+07_S5_Lcz1_256x128'


def test_shared_diagnostics_has_no_anelastic_model_state():
    source = (ROOT / 'diagnostics.py').read_text()
    tree = ast.parse(source)
    assignments = {
        target.id
        for node in tree.body if isinstance(node, ast.Assign)
        for target in node.targets if isinstance(target, ast.Name)
    }
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

    assert 'model' not in assignments
    assert 'TASKS' not in assignments
    assert 'compute_Nu_anelastic' not in functions
    assert 'anelastic' not in source
    thermal = ast.get_source_segment(source, functions['compute_turbulent_thermal_diffusivity'])
    assert "p['theta_havg']" in thermal
    assert 'TASKS' not in thermal and 'model' not in thermal and 'rho0' not in thermal
    for path in ('plot_run.py', 'diffusion_v3.py', 'plot_tracer.py'):
        assert 'diag.model' not in (ROOT / path).read_text()


def test_cache_ignores_and_pytest_collection_are_narrow():
    assert (ROOT / '.gitignore').read_text().splitlines() == [
        '.ipynb_checkpoints/',
        '__pycache__/',
        '.pytest_cache/',
    ]

    config = configparser.ConfigParser()
    config.read(ROOT / 'pytest.ini')
    assert config['pytest']['testpaths'].split() == ['tests']
    assert config['pytest']['python_files'].split() == ['test_*.py']
