import sys
from unittest.mock import MagicMock

# Mock monet_stats
mock_monet_stats = MagicMock()
mock_monet_stats.mb = MagicMock()
sys.modules["monet_stats"] = mock_monet_stats

from mdt.tasks.statistics import _find_metric

def test_find_metric():
    # Test case-insensitive match
    metric = _find_metric(mock_monet_stats, "MB")
    print(f"Found MB: {metric is mock_monet_stats.mb}")

    # Test alias BIAS (should fail currently)
    metric_bias = _find_metric(mock_monet_stats, "BIAS")
    print(f"Found BIAS: {metric_bias is not None}")
    if metric_bias is mock_monet_stats.mb:
        print("BIAS correctly mapped to mb")
    else:
        print("BIAS NOT mapped to mb")

if __name__ == "__main__":
    test_find_metric()
