import glob
import os
import pytest
from mdt.config import ConfigParser
from mdt.dag import DAGBuilder

def get_example_configs():
    """Find all example YAML files in the docs/examples/ directory."""
    examples_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "examples")
    return glob.glob(os.path.join(examples_dir, "*.yaml"))

@pytest.mark.parametrize("config_path", get_example_configs())
def test_example_configs_validate(config_path):
    """Ensure that all example configurations in the documentation are valid."""
    # 1. Load the configuration
    config = ConfigParser(config_path)

    # 2. Build the DAG (this validates task dependencies and schema)
    builder = DAGBuilder(config)
    dag = builder.build()

    assert dag is not None
    assert len(dag.nodes) > 0
