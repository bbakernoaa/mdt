"""Allocation-script generation tests (feature 001-prep-fixed-allocation, US3).

Contract: specs/001-prep-fixed-allocation/contracts/cli.md (`mdt script`).
Scenarios per quickstart.md §A: directives/body match, deterministic output,
missing-section error, no submission side effects (FR-011).
"""

from unittest.mock import MagicMock

import pytest

from mdt.config import ConfigParser
from mdt.scriptgen import render_allocation_script

SCRIPT_YAML = """\
data:
  obs:
    type: aeronet
    kwargs: {files: dummy}
execution:
  orchestrator: dask
  dask_interface: hsn0
  allocation:
    nodes: 2
    ntasks_per_node: 4
    cpus_per_task: 8
    memory: 200GB
    walltime: "02:00:00"
    account: bil-fire3
    partition: batch
    qos: normal
    slurm_cluster: c6
    python: /envs/mdt/bin/python
    prologue: ["cd /repo", "export FOO=1"]
  clusters:
    compute: {mode: allocation}
"""


def _config(tmp_path, body=SCRIPT_YAML):
    path = tmp_path / "wf.yaml"
    path.write_text(body)
    return ConfigParser(str(path)), path


def test_script_directives_match_contract(tmp_path):
    """Generated script carries the contract's SBATCH directives and srun line."""
    config, path = _config(tmp_path)
    script = render_allocation_script(config, str(path))
    assert script.startswith("#!/bin/bash")
    for directive in (
        "#SBATCH -N 2",
        "#SBATCH --ntasks-per-node=4",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=200GB",
        "#SBATCH -t 02:00:00",
        "#SBATCH -A bil-fire3",
        "#SBATCH -p batch",
        "#SBATCH --qos=normal",
        "#SBATCH -M c6",
    ):
        assert directive in script, directive
    assert "cd /repo" in script
    assert "export FOO=1" in script
    assert f"srun -n 8 /envs/mdt/bin/python -m mdt.allocation {path.resolve()}" in script


def test_script_optional_directives_absent_when_unset(tmp_path):
    """Optional directives are omitted when not configured."""
    body = SCRIPT_YAML.replace("    partition: batch\n", "").replace("    qos: normal\n", "").replace("    slurm_cluster: c6\n", "")
    config, path = _config(tmp_path, body)
    script = render_allocation_script(config, str(path))
    assert "#SBATCH -p" not in script
    assert "#SBATCH --qos" not in script
    assert "#SBATCH -M" not in script


def test_script_deterministic(tmp_path):
    """Rendering the same config twice yields identical text."""
    config, path = _config(tmp_path)
    assert render_allocation_script(config, str(path)) == render_allocation_script(config, str(path))


def test_script_defaults(tmp_path):
    """python/prologue fall back to sys.executable and the working directory."""
    body = SCRIPT_YAML.replace("    python: /envs/mdt/bin/python\n", "").replace('    prologue: ["cd /repo", "export FOO=1"]\n', "")
    config, path = _config(tmp_path, body)
    script = render_allocation_script(config, str(path))
    import sys

    assert f"-m mdt.allocation {path.resolve()}" in script
    assert sys.executable in script


def test_script_missing_allocation_section_errors(tmp_path):
    """Missing execution.allocation raises with guidance."""
    body = SCRIPT_YAML.split("  allocation:")[0] + "  clusters:\n    compute: {mode: local}\n"
    config, path = _config(tmp_path, body)
    with pytest.raises(ValueError, match="execution.allocation"):
        render_allocation_script(config, str(path))


def test_script_missing_required_fields_error(tmp_path):
    """Missing nodes/account raise naming the fields."""
    body = SCRIPT_YAML.replace("    account: bil-fire3\n", "").replace("    nodes: 2\n", "")
    config, path = _config(tmp_path, body)
    with pytest.raises(ValueError, match="nodes|account"):
        render_allocation_script(config, str(path))


def test_script_cli_writes_file_without_submitting(tmp_path, monkeypatch):
    """CLI writes the script and never submits it."""
    from mdt.cli import main

    _, path = _config(tmp_path)
    out = tmp_path / "run.slurm"
    submit = MagicMock()
    monkeypatch.setattr("subprocess.run", submit)
    monkeypatch.setattr("sys.argv", ["mdt", "script", str(path), "-o", str(out)])
    main()
    text = out.read_text()
    assert "#SBATCH -N 2" in text
    assert "srun -n 8" in text
    submit.assert_not_called()


def test_script_cli_missing_section_exit_1(tmp_path, monkeypatch):
    """CLI exits 1 when the allocation section is absent."""
    from mdt.cli import main

    body = SCRIPT_YAML.split("  allocation:")[0] + "  clusters:\n    compute: {mode: local}\n"
    _, path = _config(tmp_path, body)
    monkeypatch.setattr("sys.argv", ["mdt", "script", str(path), "-o", str(tmp_path / "x.slurm")])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


# --- Prep-stage script (--stage prep) ---

PREP_SECTION = """\
  prep_allocation:
    cpus: 4
    walltime: "04:00:00"
    account: bil-fire3
    partition: dtn_f5_f6
    qos: dtn
    slurm_cluster: es
    extra_directives: ["--constraint=f6"]
    python: /envs/mdt/bin/python
    prologue: ["cd /repo"]
"""


def test_prep_script_directives(tmp_path):
    """Prep script is a single DTN task with the configured directives."""
    from mdt.scriptgen import render_prep_script

    body = SCRIPT_YAML.replace("  allocation:", PREP_SECTION + "  allocation:")
    config, path = _config(tmp_path, body)
    script = render_prep_script(config, str(path))
    assert script.startswith("#!/bin/bash")
    assert "#SBATCH -J mdt-prep" in script
    assert "#SBATCH -N 1" in script
    assert "#SBATCH -n 1" in script
    assert "#SBATCH --cpus-per-task=4" in script
    assert "#SBATCH -p dtn_f5_f6" in script
    assert "#SBATCH --qos=dtn" in script
    assert "#SBATCH -M es" in script
    assert "#SBATCH --constraint=f6" in script
    assert "srun" not in script  # prep is a single task, no runner bootstrap
    assert f"/envs/mdt/bin/python -m mdt prep {path.resolve()}" in script


def test_prep_script_defaults_from_allocation(tmp_path):
    """Prep defaults are DTN-safe and fall back to the allocation account."""
    from mdt.scriptgen import render_prep_script

    config, path = _config(tmp_path)  # no prep_allocation section
    script = render_prep_script(config, str(path))
    assert "#SBATCH --cpus-per-task=4" in script  # DTN-safe default
    assert "#SBATCH -A bil-fire3" in script  # falls back to allocation.account
    assert "-m mdt prep" in script


def test_prep_script_force_flag(tmp_path):
    """prep_allocation.force adds --force to the prep command."""
    from mdt.scriptgen import render_prep_script

    body = SCRIPT_YAML.replace("  allocation:", "  prep_allocation:\n    force: true\n  allocation:")
    config, path = _config(tmp_path, body)
    script = render_prep_script(config, str(path))
    assert f"-m mdt prep {path.resolve()} --force" in script


def test_prep_script_requires_account(tmp_path):
    """Prep generation without any account raises."""
    from mdt.scriptgen import render_prep_script

    body = SCRIPT_YAML.replace("    account: bil-fire3\n", "")
    config, path = _config(tmp_path, body)
    with pytest.raises(ValueError, match="account"):
        render_prep_script(config, str(path))


def test_prep_script_cli_default_output(tmp_path, monkeypatch):
    """--stage prep defaults the output to mdt_prep.slurm."""
    from mdt.cli import main

    body = SCRIPT_YAML.replace("  allocation:", PREP_SECTION + "  allocation:")
    _, path = _config(tmp_path, body)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["mdt", "script", str(path), "--stage", "prep"])
    main()
    assert (tmp_path / "mdt_prep.slurm").exists()
