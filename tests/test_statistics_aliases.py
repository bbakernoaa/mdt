import sys
from unittest.mock import MagicMock

# Mock monet_stats before importing compute_statistics
mock_monet_stats = MagicMock()
mock_monet_stats.mb = MagicMock()
mock_monet_stats.__name__ = "monet_stats"
sys.modules["monet_stats"] = mock_monet_stats

# Mock other dependencies if needed
sys.modules["yaml"] = MagicMock()

from mdt.tasks.statistics import _find_metric

def test_find_metric_aliases():
    # Test case-insensitive match
    metric = _find_metric(mock_monet_stats, "MB")
    assert metric is mock_monet_stats.mb

    # Test alias BIAS
    metric_bias = _find_metric(mock_monet_stats, "BIAS")
    assert metric_bias is mock_monet_stats.mb

    # Test alias MBIAS
    metric_mbias = _find_metric(mock_monet_stats, "MBIAS")
    assert metric_mbias is mock_monet_stats.mb

    # Test non-existent metric
    assert _find_metric(mock_monet_stats, "NONEXISTENT") is None

if __name__ == "__main__":
    test_find_metric_aliases()
    print("All tests passed!")
