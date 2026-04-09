"""Unit tests for CLI orchestrator selection logic.

Validates: Requirements 7.1, 7.2, 7.3
"""

from unittest.mock import MagicMock, patch

import networkx as nx
import pytest


@pytest.fixture()
def mock_dag():
    """Return a trivial DAG for CLI tests."""
    return nx.DiGraph()


@pytest.fixture()
def mock_config_factory():
    """Factory that builds a mock ConfigParser with a configurable orchestrator."""

    def _make(orchestrator="prefect"):
        cfg = MagicMock()
        cfg.orchestrator = orchestrator
        return cfg

    return _make


@pytest.fixture()
def mock_engine_class():
    """Return a mock engine class whose instances have an .execute() method."""
    engine_cls = MagicMock()
    engine_cls.return_value.execute.return_value = {"status": "ok"}
    return engine_cls


class TestCLIOrchestratorSelection:
    """Verify the CLI resolves the orchestrator in the correct priority order."""

    @patch("mdt.cli.EngineRegistry")
    @patch("mdt.cli.DAGBuilder")
    @patch("mdt.cli.load_config")
    def test_cli_arg_overrides_yaml_config(
        self, mock_load_config, mock_dag_builder, mock_registry, mock_config_factory, mock_dag, mock_engine_class
    ):
        """CLI --orchestrator ecflow overrides YAML config that says prefect."""
        # YAML says "prefect", CLI says "ecflow" → ecflow wins
        config = mock_config_factory(orchestrator="prefect")
        mock_load_config.return_value = config
        mock_dag_builder.return_value.build.return_value = mock_dag
        mock_registry.get_engine.return_value = mock_engine_class

        with patch("sys.argv", ["mdt", "run", "config.yaml", "--orchestrator", "ecflow"]):
            from mdt.cli import main

            main()

        mock_registry.get_engine.assert_called_once_with("ecflow")

    @patch("mdt.cli.EngineRegistry")
    @patch("mdt.cli.DAGBuilder")
    @patch("mdt.cli.load_config")
    def test_yaml_config_used_when_cli_arg_absent(
        self, mock_load_config, mock_dag_builder, mock_registry, mock_config_factory, mock_dag, mock_engine_class
    ):
        """No --orchestrator flag → YAML config value is used."""
        config = mock_config_factory(orchestrator="ecflow")
        mock_load_config.return_value = config
        mock_dag_builder.return_value.build.return_value = mock_dag
        mock_registry.get_engine.return_value = mock_engine_class

        with patch("sys.argv", ["mdt", "run", "config.yaml"]):
            from mdt.cli import main

            main()

        mock_registry.get_engine.assert_called_once_with("ecflow")

    @patch("mdt.cli.EngineRegistry")
    @patch("mdt.cli.DAGBuilder")
    @patch("mdt.cli.load_config")
    def test_default_falls_back_to_prefect(
        self, mock_load_config, mock_dag_builder, mock_registry, mock_config_factory, mock_dag, mock_engine_class
    ):
        """No --orchestrator flag and YAML defaults to 'prefect' → prefect is used."""
        config = mock_config_factory(orchestrator="prefect")
        mock_load_config.return_value = config
        mock_dag_builder.return_value.build.return_value = mock_dag
        mock_registry.get_engine.return_value = mock_engine_class

        with patch("sys.argv", ["mdt", "run", "config.yaml"]):
            from mdt.cli import main

            main()

        mock_registry.get_engine.assert_called_once_with("prefect")
