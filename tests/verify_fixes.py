import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd  # noqa: F401
import xarray as xr


def test_find_metric_aliases(mocker):
    """Test that metric aliases are correctly identified."""
    mock_monet_stats = MagicMock()
    mock_monet_stats.mb = "MB_METRIC"
    mocker.patch.dict(sys.modules, {"monet_stats": mock_monet_stats})

    from mdt.tasks.statistics import _find_metric

    metric = _find_metric(mock_monet_stats, "BIAS")
    assert metric == "MB_METRIC"


def test_execute_metric_signature_robustness():
    """Test that _execute_metric is robust to functions without signatures."""

    def no_sig_func(obs, mod):
        return obs - mod

    # Manually remove signature if it's a mock or something
    data = xr.Dataset({"obs": (("x"), [1]), "mod": (("x"), [0])})

    from mdt.tasks.statistics import _execute_metric

    res = _execute_metric(data, no_sig_func, {"obs_var": "obs", "mod_var": "mod"})
    assert res == 1


def test_calculate_reduction_kwargs_filtering(mocker):
    """Test that calculate_reduction correctly filters kwargs."""
    mock_monet_stats = MagicMock()
    mock_monet_stats.weighted_spatial_mean = lambda obj, **kwargs: obj.mean()
    mocker.patch.dict(sys.modules, {"monet_stats": mock_monet_stats})

    from mdt.tasks.reductions import calculate_reduction

    data = xr.DataArray([[1.0, 2.0]], coords={"lat": [0], "lon": [0, 1]}, dims=("lat", "lon"))
    # This should not raise TypeError when calling .mean() in Step B
    res = calculate_reduction(data, method="mean", dim=["lat", "lon"], force_weighted=True, some_extra="val")
    assert res == 1.5


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
