import sys
from unittest.mock import MagicMock
import numpy as np
import xarray as xr
import pandas as pd

# Mock monet_stats
mock_monet_stats = MagicMock()
sys.modules["monet_stats"] = mock_monet_stats

# Mock monetio and monet
sys.modules["monetio"] = MagicMock()
sys.modules["monet"] = MagicMock()

from mdt.tasks.statistics import compute_statistics, _find_metric, _execute_metric
from mdt.tasks.reductions import calculate_reduction

def test_find_metric_aliases():
    mock_monet_stats.mb = MagicMock()
    metric = _find_metric(mock_monet_stats, "BIAS")
    assert metric is mock_monet_stats.mb
    print("test_find_metric_aliases passed")

def test_execute_metric_signature_robustness():
    def no_sig_func(obs, mod):
        return obs - mod

    # Manually remove signature if it's a mock or something
    data = xr.Dataset({"obs": (("x"), [1]), "mod": (("x"), [0])})
    res = _execute_metric(data, no_sig_func, {"obs_var": "obs", "mod_var": "mod"})
    assert res == 1
    print("test_execute_metric_signature_robustness passed")

def test_calculate_reduction_kwargs_filtering():
    def mock_weighted_spatial_mean(obj, **kwargs):
        if "lat_dim" in kwargs or "lon_dim" in kwargs:
             # In my implementation Step A still gets them, Step B filters them.
             # Wait, Step A explicitly passes them as named args.
             pass
        return obj.mean()

    mock_monet_stats.weighted_spatial_mean = mock_weighted_spatial_mean

    data = xr.DataArray([[1.0, 2.0]], coords={"lat": [0], "lon": [0, 1]}, dims=("lat", "lon"))
    # This should not raise TypeError when calling .mean() in Step B
    res = calculate_reduction(data, method="mean", dim=["lat", "lon"], force_weighted=True, some_extra="val")
    assert res == 1.5
    print("test_calculate_reduction_kwargs_filtering passed")

if __name__ == "__main__":
    test_find_metric_aliases()
    test_execute_metric_signature_robustness()
    test_calculate_reduction_kwargs_filtering()
    print("All custom verifications passed!")
