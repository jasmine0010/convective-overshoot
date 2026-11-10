import argparse
import ast
from pathlib import Path


root = Path(__file__).resolve().parents[1]


def test_tracer_release_defaults(monkeypatch):
    tree = ast.parse((root/'convection.py').read_text())
    parse_args = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'parse_args')
    namespace = {'argparse': argparse}
    exec(compile(ast.Module(body=[parse_args], type_ignores=[]), 'convection.py', 'exec'), namespace)
    monkeypatch.setattr('sys.argv', ['convection.py', '--run-id', 'synthetic', '--mode', 'run', '--stop-sim-time', '2600'])
    args = namespace['parse_args']()
    assert (args.release_time_C1, args.release_time_C2, args.release_time_C3) == (1600, 2200, 2200)


def test_fit_window_and_launchers_without_running_simulations():
    tree = ast.parse((root/'diffusion_v4.py').read_text())
    offsets = next(node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'training_offsets' for target in node.targets))
    assert tuple(1600+offset for offset in ast.literal_eval(offsets)) == (2000, 2150)
    for launcher in ('initial_run.py', 'forced_ts.py'):
        tree = ast.parse((root/'runs'/launcher).read_text())
        end = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'stop_sim_time' for target in node.targets))
        assert ast.literal_eval(end) == 2600
