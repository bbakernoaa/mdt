import dask.array as da
import numpy as np
import xarray as xr

from mdt.tasks.statistics import _find_metric, compute_statistics


def test_find_metric_robustness(mocker):
    """Test _find_metric with various naming conventions and submodules."""

    class MockStats:
        def my_metric(self):
            pass

    class MockModule:
        def __init__(self):
            self.stats = MockStats()

        def top_metric(self):
            pass

        def lower_metric(self):
            pass

    mock_mod = MockModule()

    # Top level case insensitive
    assert _find_metric(mock_mod, "top_metric") == mock_mod.top_metric
    assert _find_metric(mock_mod, "LOWER_METRIC") == mock_mod.lower_metric

    # Submodule discovery
    assert _find_metric(mock_mod, "my_metric") == mock_mod.stats.my_metric

    # Not found
    assert _find_metric(mock_mod, "nonexistent") is None


def test_execute_metric_manual_fallback_rmse(mocker):
    """Test manual fallback for RMSE when native weights support is missing."""
    # 1. Setup Eager Data
    lat = np.array([10, 20])
    lon = np.array([100, 110])
    ds = xr.Dataset(
        {
            "obs": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]]),
            "mod": (("lat", "lon"), [[1.1, 2.1], [3.1, 4.1]]),
            "w": (("lat", "lon"), [[1.0, 1.0], [0.5, 0.5]]),
        },
        coords={"lat": lat, "lon": lon},
    )

    # 2. Mock a metric function that DOES NOT have 'weights' in signature
    def mock_rmse_no_weights(obs, mod, axis=None):
        return ((mod - obs) ** 2).mean()

    mock_rmse_no_weights.__name__ = "RMSE"

    # We need to mock monet_stats.weighted_spatial_mean as it's used in fallback
    import monet_stats

    mocker.patch("monet_stats.weighted_spatial_mean", side_effect=monet_stats.weighted_spatial_mean)

    # Mock discovery to return our no-weights function
    mocker.patch("mdt.tasks.statistics._find_metric", return_value=mock_rmse_no_weights)

    # 3. Execute
    kwargs = {"obs_var": "obs", "mod_var": "mod", "weights": "w", "dim": ["lat", "lon"]}
    results = compute_statistics("test_fallback", ["RMSE"], ds, kwargs)

    # 4. Verify result (manual calculation)
    # diff = [0.1, 0.1, 0.1, 0.1], diff^2 = [0.01, 0.01, 0.01, 0.01]
    # weights = [1, 1, 0.5, 0.5]
    # sum(diff^2 * w) = 0.01*1 + 0.01*1 + 0.01*0.5 + 0.01*0.5 = 0.03
    # sum(w) = 3.0
    # weighted_mean = 0.03 / 3.0 = 0.01
    # sqrt(0.01) = 0.1
    assert np.isclose(results["RMSE"].values, 0.1)
    assert "Computed RMSE" in results["RMSE"].attrs["history"]


def test_execute_metric_manual_fallback_corr(mocker):
    """Test manual fallback for CORR when native weights support is missing."""
    # 1. Setup Eager Data
    ds = xr.Dataset(
        {
            "obs": (("x"), [1.0, 2.0, 3.0]),
            "mod": (("x"), [2.0, 4.0, 6.0]),
            "w": (("x"), [1.0, 1.0, 1.0]),
        },
        coords={"x": [0, 1, 2]},
    )

    def mock_corr_no_weights(obs, mod, axis=None):
        return 1.0

    mock_corr_no_weights.__name__ = "CORR"

    mocker.patch("mdt.tasks.statistics._find_metric", return_value=mock_corr_no_weights)

    # 3. Execute
    kwargs = {"obs_var": "obs", "mod_var": "mod", "weights": "w", "dim": "x"}
    results = compute_statistics("test_fallback_corr", ["CORR"], ds, kwargs)

    # Perfectly correlated
    assert np.isclose(results["CORR"].values, 1.0)


def test_execute_metric_lazy_fallback(mocker):
    """Verify that manual fallback preserves laziness."""
    ds_lazy = xr.Dataset(
        {
            "obs": (("x"), da.random.random(10, chunks=5)),
            "mod": (("x"), da.random.random(10, chunks=5)),
            "w": (("x"), da.ones(10, chunks=5)),
        },
        coords={"x": np.arange(10)},
    )

    def mock_rmse_no_weights(obs, mod, axis=None):
        return 0.0

    mock_rmse_no_weights.__name__ = "RMSE"
    mocker.patch("mdt.tasks.statistics._find_metric", return_value=mock_rmse_no_weights)

    kwargs = {"obs_var": "obs", "mod_var": "mod", "weights": "w", "dim": "x"}
    results = compute_statistics("test_lazy_fallback", ["RMSE"], ds_lazy, kwargs)

    assert hasattr(results["RMSE"].data, "dask")
