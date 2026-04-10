import numpy as np
import xarray as xr

from mdt.tasks.statistics import compute_statistics


def test_weighted_metrics_aero_protocol(mocker):
    """Double-Check Test: Verify weighted MB and MAE for Eager and Lazy backends."""
    # 1. Setup Data (2D Spatial Grid)
    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(-180, 180, 20)
    obs_raw = np.random.rand(10, 20)
    mod_raw = np.random.rand(10, 20)
    weights_raw = np.random.rand(10, 20)

    ds_eager = xr.Dataset(
        {
            "obs": (("lat", "lon"), obs_raw),
            "mod": (("lat", "lon"), mod_raw),
            "w": (("lat", "lon"), weights_raw),
        },
        coords={"lat": lat, "lon": lon},
    )

    ds_lazy = ds_eager.chunk({"lat": 5, "lon": 10})

    # 2. Setup Mock for monet-stats
    # We need mock functions that do NOT have 'weights' in signature to trigger fallback
    def mock_mb(obs, mod, **kwargs):
        return (mod - obs).mean()

    mock_mb.__name__ = "MB"

    def mock_mae(obs, mod, **kwargs):
        return np.abs(mod - obs).mean()

    mock_mae.__name__ = "MAE"

    # We also need to mock weighted_spatial_mean as it's used in the fallback
    def mock_weighted_mean(data, weights=None, **kwargs):
        if weights is not None:
            return (data * weights).sum() / weights.sum()
        return data.mean()

    # Mock monet_stats module
    import sys

    mock_monet_stats = mocker.MagicMock()
    mock_monet_stats.weighted_spatial_mean.side_effect = mock_weighted_mean
    mocker.patch.dict(sys.modules, {"monet_stats": mock_monet_stats})

    # Mock _find_metric to return our specific mocks
    def side_effect(module, name):
        if name.upper() in ["MB", "BIAS", "MBIAS"]:
            return mock_mb
        if name.upper() == "MAE":
            return mock_mae
        return None

    mocker.patch("mdt.tasks.statistics._find_metric", side_effect=side_effect)

    metrics = ["MB", "MAE", "MBIAS"]
    kwargs = {"obs_var": "obs", "mod_var": "mod", "weights": "w"}

    # 3. Execute Eager
    res_eager = compute_statistics("test_eager", metrics, ds_eager, kwargs)

    # 4. Execute Lazy
    res_lazy = compute_statistics("test_lazy", metrics, ds_lazy, kwargs)

    # 5. Assertions
    for m in metrics:
        # Check Laziness
        assert hasattr(res_lazy[m].data, "dask"), f"Result for {m} should be lazy"

        # Double-Check: Eager == Lazy
        xr.testing.assert_allclose(res_eager[m], res_lazy[m].compute())

        # Verify result is indeed weighted
        if m in ["MB", "MBIAS"]:
            expected = ((ds_eager.mod - ds_eager.obs) * ds_eager.w).sum() / ds_eager.w.sum()
        elif m == "MAE":
            expected = (np.abs(ds_eager.mod - ds_eager.obs) * ds_eager.w).sum() / ds_eager.w.sum()

        xr.testing.assert_allclose(res_eager[m], expected)

    print("\n✅ Weighted Metrics Double-Check Passed: Eager == Lazy")
