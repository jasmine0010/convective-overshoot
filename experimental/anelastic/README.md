# Experimental anelastic workflow

Status: experimental and not currently supported for production runs or analysis.

## Commands

From the project root:

- Solver: python -m experimental.anelastic.solver ...
- Configured runner: python -m experimental.anelastic.run

The runner uses the active Python interpreter and launches the solver through MPI. Both modules resolve the repository root from their own file location, so outputs still go to the existing project-root data/tests or data/runs directories.

## Legacy analysis

plot_legacy.py is retained only as a reference and is not currently runnable. Do not use it to interpret new runs without reviewing its scientific assumptions and updating its stale diagnostics calls.

Known incompatibilities:

- The solver writes F_entropy_havg and F_enthalpy_havg, while the legacy analysis expects F_conv_havg.
- Several plotting and derived-diagnostic calls use obsolete shared-diagnostics signatures.
- The intended anelastic convective heat-flux definition has not been selected.
- Existing output names under the project-root analysis directory are preserved in the legacy source.

Before resuming this work, choose and document the heat-flux semantics, reconcile the expected HDF5 tasks, update the stale calls, and add dedicated anelastic tests.

## Checkpoints

No archive directory was needed. The plotter and runner checkpoints were byte-identical copies. The solver checkpoint contained the same solver logic with superseded direct JSON writes; the current source uses the shared atomic JSON writer. All three generated-equivalent checkpoint files were removed.
