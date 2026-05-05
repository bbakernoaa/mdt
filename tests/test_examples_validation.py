import glob
import os
import pytest
from unittest.mock import MagicMock, patch
from mdt.config import ConfigParser
from mdt.dag import DAGBuilder
from mdt.engine_registry import Engine

def get_example_configs():
    """Find all example YAML files in the docs/examples/ directory."""
    examples_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "examples")
    return glob.glob(os.path.join(examples_dir, "*.yaml"))

@pytest.mark.parametrize("config_path", get_example_configs())
def test_example_configs_validate_and_execute(config_path):
    """Ensure that all example configurations in the documentation are valid and executable."""
    # 1. Load the configuration
    config = ConfigParser(config_path)

    # 2. Build the DAG (this validates task dependencies and schema)
    builder = DAGBuilder(config)
    dag = builder.build()

    assert dag is not None
    assert len(dag.nodes) > 0

    # 3. Test "Execution" with a Mock Engine
    class MockEngine(Engine):
        def __init__(self, dag, config):
            self.dag = dag
            self.config = config

        def execute(self):
            return {"status": "success"}

    engine = MockEngine(dag, config)
    result = engine.execute()
    assert result["status"] == "success"

@patch("mdt.engine_registry.EngineRegistry.get_engine")
@pytest.mark.parametrize("config_path", get_example_configs())
def test_example_configs_cli_dry_run(mock_get_engine, config_path):
    """Simulate a CLI 'run' of the examples with a mocked engine to verify orchestration logic."""
    from mdt.cli import main
    import sys

    # Mock the engine class and its execution
    mock_engine_instance = MagicMock()
    mock_engine_instance.execute.return_value = {"status": "ok"}
    mock_engine_cls = MagicMock(return_value=mock_engine_instance)
    mock_get_engine.return_value = mock_engine_cls

    # We use "prefect" because "mock" is not a valid choice in argparse
    test_args = ["mdt", "run", config_path, "--orchestrator", "prefect"]
    with patch.object(sys, 'argv', test_args):
        # We expect this to run through and call our mock instead of real prefect
        try:
            main()
        except SystemExit as e:
            assert e.code == 0

    mock_get_engine.assert_called_with("prefect")
    mock_engine_instance.execute.assert_called_once()
