"""Tests for ConfigParser orchestrator and ecFlow configuration support."""

import copy
import re
import textwrap

import pytest

from mdt.config import ConfigParser


@pytest.fixture
def _write_yaml(tmp_path):
    """Factory that writes a YAML string to a temp file and returns its path."""

    def _factory(yaml_text: str):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(textwrap.dedent(yaml_text))
        return cfg_file

    return _factory


class TestOrchestratorProperty:
    """Verify ConfigParser.orchestrator returns the correct value."""

    def test_default_orchestrator_when_field_absent(self, _write_yaml):
        """When no orchestrator field is in the YAML, default to 'prefect'."""
        path = _write_yaml("""\
            data:
              merra2: {type: merra2}
            execution:
              clusters:
                compute:
                  mode: local
        """)
        cfg = ConfigParser(path)
        assert cfg.orchestrator == "prefect"

    @pytest.mark.parametrize(
        ("body", "message"),
        [
            ("data: []", "'data' section must be a dictionary"),
            ("data:\n  model: {type: cmaq, kwargs: invalid}", "data.model.kwargs must be a mapping"),
            ("data:\n  model: {type: cmaq}\nexecution: []", "'execution' section must be a dictionary"),
            ("data:\n  model: {type: cmaq}\nexecution:\n  pairing_mode: parallel", "execution.pairing_mode must be 'serial' or 'concurrent'"),
            ("data:\n  model: {type: cmaq}\ncombine:\n  all: {sources: bad}", "combine.all.sources must be a non-empty list"),
            ("data:\n  model: {type: cmaq}\nexecution:\n  clusters: []", "execution.clusters must be a mapping"),
            ("data:\n  model: {type: cmaq}\nexecution:\n  clusters:\n    compute: {mode: local, workers: 0}", "workers must be a positive integer"),
            ("data:\n  model: {type: cmaq}\nexecution:\n  clusters:\n    compute: {mode: mystery}", "mode must be one of"),
            (
                "data:\n  model: {type: cmaq}\nexecution:\n  default_cluster: missing\n  clusters:\n    compute: {mode: local}",
                "is not defined in execution.clusters",
            ),
            ("data:\n  model: {type: cmaq}\nexecution:\n  clusters: {}", "must contain at least one cluster"),
            ("data:\n  model: {type: cmaq}\nstatistics:\n  stats: {input: pair_model}\n", "non-empty list of metric names"),
            ("data:\n  model: {type: cmaq}\nstatistics:\n  stats: {input: pair_model, metrics: []}\n", "non-empty list of metric names"),
        ],
    )
    def test_malformed_sections_and_pairing_mode_are_rejected(self, _write_yaml, body, message):
        with pytest.raises(ValueError, match=re.escape(message)):
            ConfigParser(_write_yaml(body))

    def test_execution_property_does_not_mutate_source_config(self, _write_yaml):
        path = _write_yaml("""\
            data:
              model: {type: cmaq}
            execution:
              default_cluster: compute
              clusters:
                compute: {mode: local}
        """)
        cfg = ConfigParser(path)
        before = copy.deepcopy(cfg.config["execution"])
        first = cfg.execution
        second = cfg.execution
        assert first == second
        assert cfg.config["execution"] == before

    @pytest.mark.parametrize("pairing_mode", ["serial", "concurrent"])
    def test_pairing_mode_accepts_supported_values(self, _write_yaml, pairing_mode):
        path = _write_yaml(f"data:\n  model: {{type: cmaq}}\nexecution:\n  pairing_mode: {pairing_mode}\n")
        cfg = ConfigParser(path)
        assert cfg.execution["pairing_mode"] == pairing_mode

    def test_orchestrator_ecflow_from_yaml(self, _write_yaml):
        """When orchestrator is set to 'ecflow', property returns 'ecflow'."""
        path = _write_yaml("""\
            data:
              merra2: {type: merra2}
            execution:
              orchestrator: ecflow
              clusters:
                compute:
                  mode: local
        """)
        cfg = ConfigParser(path)
        assert cfg.orchestrator == "ecflow"

    def test_orchestrator_prefect_explicit(self, _write_yaml):
        """When orchestrator is explicitly set to 'prefect', property returns 'prefect'."""
        path = _write_yaml("""\
            data:
              merra2: {type: merra2}
            execution:
              orchestrator: prefect
              clusters:
                compute:
                  mode: local
        """)
        cfg = ConfigParser(path)
        assert cfg.orchestrator == "prefect"


class TestEcFlowKeysAccessible:
    """Verify ecFlow-specific keys are accessible from the execution property."""

    def test_ecflow_keys_present_in_execution(self, _write_yaml):
        """Verify ecFlow-specific keys are present in the execution property."""
        path = _write_yaml("""\
            data:
              merra2: {type: merra2}
            execution:
              orchestrator: ecflow
              ecflow_host: ecflow-server.local
              ecflow_port: 4000
              suite_name: my_suite
              task_script_dir: /tmp/ecf_scripts/
              clusters:
                compute:
                  mode: hera
        """)
        cfg = ConfigParser(path)
        ex = cfg.execution

        assert ex["ecflow_host"] == "ecflow-server.local"
        assert ex["ecflow_port"] == 4000
        assert ex["suite_name"] == "my_suite"
        assert ex["task_script_dir"] == "/tmp/ecf_scripts/"

    def test_ecflow_keys_with_defaults_via_get(self, _write_yaml):
        """EcFlowEngine uses .get() with defaults; verify that works on the dict."""
        path = _write_yaml("""\
            data:
              merra2: {type: merra2}
            execution:
              orchestrator: ecflow
              clusters:
                compute:
                  mode: local
        """)
        cfg = ConfigParser(path)
        ex = cfg.execution

        # Keys not in YAML — .get() should return the defaults EcFlowEngine uses
        assert ex.get("ecflow_host", "localhost") == "localhost"
        assert int(ex.get("ecflow_port", 3141)) == 3141
        assert ex.get("suite_name", "mdt") == "mdt"
        assert ex.get("task_script_dir", "./ecflow_tasks/") == "./ecflow_tasks/"

    def test_ecflow_keys_preserved_in_backwards_compat_mode(self, _write_yaml):
        """When 'mode' is present without 'clusters', ecFlow keys must survive."""
        path = _write_yaml("""\
            data:
              merra2: {type: merra2}
            execution:
              orchestrator: ecflow
              ecflow_host: myhost
              ecflow_port: 5000
              suite_name: test_suite
              task_script_dir: ./scripts/
              mode: local
        """)
        cfg = ConfigParser(path)
        ex = cfg.execution

        # ecFlow keys should still be at the top level
        assert ex.get("ecflow_host") == "myhost"
        assert ex.get("ecflow_port") == 5000
        assert ex.get("suite_name") == "test_suite"
        assert ex.get("task_script_dir") == "./scripts/"
        assert ex.get("orchestrator") == "ecflow"
        # Backwards-compat wrapping should still work
        assert "clusters" in ex
        assert "compute" in ex["clusters"]

    def test_orchestrator_key_accessible_from_execution(self, _write_yaml):
        """Verify the orchestrator key is accessible from the execution property."""
        path = _write_yaml("""\
            data:
              merra2: {type: merra2}
            execution:
              orchestrator: ecflow
              clusters:
                compute:
                  mode: local
        """)
        cfg = ConfigParser(path)
        assert cfg.execution.get("orchestrator") == "ecflow"
        assert cfg.orchestrator == "ecflow"


def test_pairing_configuration_requires_source_and_target(_write_yaml):
    """Reject pairings that omit either endpoint."""
    path = _write_yaml("""\
        data:
          obs: {type: aeronet}
        pairing:
          eval: {target: obs}
    """)
    with pytest.raises(ValueError, match="must specify 'source'.*and 'target'"):
        ConfigParser(path)


@pytest.mark.parametrize(
    ("time_axis", "message"),
    [
        ("not-a-map", "must be a mapping"),
        ({"bogus": "time"}, "unknown role"),
        ({"lead_time": []}, "non-empty list"),
        ({"valid_time": ["time", ""]}, "non-empty list"),
    ],
)
def test_config_rejects_invalid_time_axis(_write_time_axis_config, time_axis, message):
    """Reject malformed coordinate role mappings."""
    with pytest.raises(ValueError, match=message):
        ConfigParser(_write_time_axis_config(time_axis))


@pytest.fixture
def _write_time_axis_config(tmp_path):
    import yaml

    def write(time_axis):
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump({"data": {"model": {"type": "gefs", "kwargs": {"time_axis": time_axis}}}}))
        return path

    return write


def test_config_accepts_time_axis_aliases_and_multi_source_pairing(tmp_path):
    """Allow role aliases and a valid list of source datasets."""
    import yaml

    config = {
        "data": {"a": {"type": "gefs", "kwargs": {"time_axis": {"valid_time": "valid", "lead_time": ["step", "lead"]}}}, "b": {"type": "gefs"}},
        "pairing": {"eval": {"sources": ["a", "b"], "target": "a", "model_dim": "experiment"}},
    }
    path = tmp_path / "valid.yaml"
    path.write_text(yaml.safe_dump(config))
    parsed = ConfigParser(path)
    assert parsed.pairing["eval"]["sources"] == ["a", "b"]


@pytest.mark.parametrize("sources", [[], ["model", ""]])
def test_config_rejects_invalid_multi_model_sources(tmp_path, sources):
    """Require non-empty names for each configured model source."""
    import yaml

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"data": {"model": {"type": "gefs"}}, "pairing": {"eval": {"sources": sources, "target": "obs"}}}))
    with pytest.raises(ValueError, match="sources must be a non-empty list"):
        ConfigParser(path)


# --- Feature 001-prep-fixed-allocation: allocation mode + prep section (T006) ---


def test_allocation_mode_valid(_write_yaml):
    path = _write_yaml("""\
        data:
          model: {type: gefs}
        execution:
          orchestrator: dask
          allocation:
            nodes: 1
            ntasks_per_node: 4
            cpus_per_task: 8
            memory: 200GB
            walltime: "02:00:00"
            account: bil-fire3
            partition: batch
            slurm_cluster: c6
            interface: hsn0
            prologue: ["cd /repo"]
          clusters:
            compute: {mode: allocation}
        prep:
          store_dir: ./zarr_stores
          summary_file: prep_summary.json
    """)
    cfg = ConfigParser(path)
    assert cfg.execution["allocation"]["nodes"] == 1
    assert cfg.config["prep"]["store_dir"] == "./zarr_stores"


def test_allocation_with_local_cluster_allowed(_write_yaml):
    path = _write_yaml("""\
        data:
          model: {type: gefs}
        execution:
          orchestrator: dask
          clusters:
            compute: {mode: allocation}
            service: {mode: local}
    """)
    ConfigParser(path)  # must not raise


@pytest.mark.parametrize(
    ("body", "message"),
    [
        # world size < 3
        (
            "data:\n  m: {type: gefs}\nexecution:\n  orchestrator: dask\n  allocation: {nodes: 1, ntasks_per_node: 2}\n  clusters:\n    c: {mode: allocation}",
            r"nodes\*ntasks_per_node must be >= 3",
        ),
        # mixed elastic + allocation
        (
            "data:\n  m: {type: gefs}\nexecution:\n  orchestrator: dask\n  clusters:\n    c: {mode: allocation}\n    d: {mode: gaea}",
            "cannot be mixed with elastic cluster modes",
        ),
        # wrong orchestrator
        (
            "data:\n  m: {type: gefs}\nexecution:\n  clusters:\n    c: {mode: allocation}",
            "must be 'dask' when any cluster uses mode 'allocation'",
        ),
        # bad allocation key type
        (
            "data:\n  m: {type: gefs}\nexecution:\n  orchestrator: dask\n  allocation: {nodes: 0}\n  clusters:\n    c: {mode: allocation}",
            "execution.allocation.nodes must be a positive integer",
        ),
        # unknown allocation key
        (
            "data:\n  m: {type: gefs}\nexecution:\n  orchestrator: dask\n  allocation: {bogus: 1}\n  clusters:\n    c: {mode: allocation}",
            "unknown execution.allocation keys",
        ),
        # bad prologue
        (
            "data:\n  m: {type: gefs}\nexecution:\n  orchestrator: dask\n  allocation: {prologue: notalist}\n  clusters:\n    c: {mode: allocation}",
            "prologue must be a list",
        ),
        # bad prep key
        (
            "data:\n  m: {type: gefs}\nprep: {store_dir: ''}",
            "prep.store_dir must be a non-empty path string",
        ),
        # unknown prep key
        (
            "data:\n  m: {type: gefs}\nprep: {bogus: 1}",
            "unknown prep keys",
        ),
        # prep section wrong type
        (
            "data:\n  m: {type: gefs}\nprep: []",
            "'prep' section must be a dictionary",
        ),
    ],
)
def test_allocation_and_prep_validation_errors(_write_yaml, body, message):
    path = _write_yaml(body)
    with pytest.raises(ValueError, match=message):
        ConfigParser(path)
