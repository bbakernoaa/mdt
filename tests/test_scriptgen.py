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
    body = SCRIPT_YAML.replace("    partition: batch\n", "").replace("    qos: normal\n", "").replace("    slurm_cluster: c6\n", "")
    config, path = _config(tmp_path, body)
    script = render_allocation_script(config, str(path))
    assert "#SBATCH -p" not in script
    assert "#SBATCH --qos" not in script
    assert "#SBATCH -M" not in script


def test_script_deterministic(tmp_path):
    config, path = _config(tmp_path)
    assert render_allocation_script(config, str(path)) == render_allocation_script(config, str(path))


def test_script_defaults(tmp_path):
    body = SCRIPT_YAML.replace("    python: /envs/mdt/bin/python\n", "").replace("    prologue: [\"cd /repo\", \"export FOO=1\"]\n", "")
    config, path = _config(tmp_path, body)
    script = render_allocation_script(config, str(path))
    import sys

    assert f"-m mdt.allocation {path.resolve()}" in script
    assert sys.executable in script


def test_script_missing_allocation_section_errors(tmp_path):
    body = SCRIPT_YAML.split("  allocation:")[0] + "  clusters:\n    compute: {mode: local}\n"
    config, path = _config(tmp_path, body)
    with pytest.raises(ValueError, match="execution.allocation"):
        render_allocation_script(config, str(path))


def test_script_missing_required_fields_error(tmp_path):
    body = SCRIPT_YAML.replace("    account: bil-fire3\n", "").replace("    nodes: 2\n", "")
    config, path = _config(tmp_path, body)
    with pytest.raises(ValueError, match="nodes|account"):
        render_allocation_script(config, str(path))


def test_script_cli_writes_file_without_submitting(tmp_path, monkeypatch):
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
    from mdt.cli import main

    body = SCRIPT_YAML.split("  allocation:")[0] + "  clusters:\n    compute: {mode: local}\n"
    _, path = _config(tmp_path, body)
    monkeypatch.setattr("sys.argv", ["mdt", "script", str(path), "-o", str(tmp_path / "x.slurm")])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1