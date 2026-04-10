"""Tests for ConfigParser orchestrator and ecFlow configuration support."""

import textwrap

import pytest

from mdt.config import ConfigParser


@pytest.fixture()
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
              merra2: {}
            execution:
              clusters:
                compute:
                  mode: local
        """)
        cfg = ConfigParser(path)
        assert cfg.orchestrator == "prefect"

    def test_orchestrator_ecflow_from_yaml(self, _write_yaml):
        """When orchestrator is set to 'ecflow', property returns 'ecflow'."""
        path = _write_yaml("""\
            data:
              merra2: {}
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
              merra2: {}
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
        path = _write_yaml("""\
            data:
              merra2: {}
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
              merra2: {}
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
              merra2: {}
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
        path = _write_yaml("""\
            data:
              merra2: {}
            execution:
              orchestrator: ecflow
              clusters:
                compute:
                  mode: local
        """)
        cfg = ConfigParser(path)
        assert cfg.execution.get("orchestrator") == "ecflow"
        assert cfg.orchestrator == "ecflow"
