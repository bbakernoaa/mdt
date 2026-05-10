import sys
import types

# MockModule for statistics aliases testing
class MockModule:
    """Mock module for testing metric discovery."""

    def __init__(self):
        """Initialize mock metrics."""
        self.mb = "METRIC_MB"
        self.stats = types.ModuleType("stats")
        self.stats.rmse = "METRIC_RMSE"


def test_find_metric_aliases(mocker):
    """Test that find_metric correctly identifies aliases like BIAS."""
    mock_monet_stats = MockModule()
    mocker.patch.dict(sys.modules, {"monet_stats": mock_monet_stats})

    from mdt.tasks.statistics import _find_metric

    # Test case-insensitive match (aliased to mb)
    metric = _find_metric(mock_monet_stats, "MB")
    assert metric == "METRIC_MB"

    # Test alias BIAS
    metric_bias = _find_metric(mock_monet_stats, "BIAS")
    assert metric_bias == "METRIC_MB"

    # Test case-insensitive submodule search
    metric_rmse = _find_metric(mock_monet_stats, "RMSE")
    assert metric_rmse == "METRIC_RMSE"


if __name__ == "__main__":
    # Manual run support if needed, but intended for pytest
    import pytest
    pytest.main([__file__])
