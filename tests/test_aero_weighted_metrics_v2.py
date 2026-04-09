import sys

import numpy as np
import pytest
import xarray as xr

from mdt.tasks.statistics import compute_statistics


def test_weighted_metrics_aero_protocol_double_check(mocker):
    """
    Aero Protocol Double-Check: Weighted MAE and MB.

    Verifies that Eager (NumPy) and Lazy (Dask) backends yield identical
    results for weighted MAE and MB using the orchestrator's ACTUAL production logic.
    """
    # 1. Setup Synthetic Grid Data
    lats = np.linspace(-90, 90, 10)
    lons = np.linspace(-180, 180, 20)
    rng = np.random.default_rng(42)
    obs_data = rng.standard_normal((len(lats), len(lons)))
    mod_data = obs_data * 0.8 + rng.standard_normal((len(lats), len(lons))) * 0.2

    # Eager Dataset (NumPy)
    ds_eager = xr.Dataset(
        {
            "obs": (("lat", "lon"), obs_data),
            "mod": (("lat", "lon"), mod_data),
        },
        coords={"lat": lats, "lon": lons},
        attrs={"history": "Initial data"},
    )
    # Add weights
    weights = np.cos(np.deg2rad(ds_eager.lat))
    ds_eager.coords["w"] = weights

    # 2. Lazy Dataset (Dask)
    ds_lazy = ds_eager.chunk({"lat": 5, "lon": 10})

    # 3. Mock monet_stats to trigger fallback logic
    # We create a mock module that has MAE and MB functions,
    # but their signature does NOT include 'weights', forcing the fallback.
    mock_monet_stats = mocker.MagicMock()

    def mae(obs, mod):
        return (mod - obs).mean()  # noqa: N802

    def mb(obs, mod):
        return (mod - obs).mean()  # noqa: N802

    # monet_stats discovery logic in mdt looks for case-insensitive matches
    # We set __name__ to trigger the fallback logic in statistics.py
    mae.__name__ = "MAE"
    mb.__name__ = "MB"

    mock_monet_stats.MAE = mae
    mock_monet_stats.MB = mb

    # We must also mock weighted_spatial_mean as it is the tool used in fallback
    import monet_stats

    mock_monet_stats.weighted_spatial_mean = monet_stats.weighted_spatial_mean

    # Patch sys.modules to inject our mock
    mocker.patch.dict(sys.modules, {"monet_stats": mock_monet_stats})

    # 4. Execution & Assertions for MAE
    kwargs = {"obs_var": "obs", "mod_var": "mod", "weights": "w"}

    # Eager MAE
    res_eager_mae = compute_statistics("test_eager", ["MAE"], ds_eager, kwargs)["MAE"]
    # Lazy MAE
    res_lazy_mae = compute_statistics("test_lazy", ["MAE"], ds_lazy, kwargs)["MAE"]

    assert hasattr(res_lazy_mae.data, "dask"), "MAE result should be lazy-backed"
    xr.testing.assert_allclose(res_eager_mae, res_lazy_mae.compute(), atol=1e-15)
    assert "Computed MAE" in res_eager_mae.attrs["history"]

    # 5. Execution & Assertions for MB
    # Eager MB
    res_eager_mb = compute_statistics("test_eager", ["MB"], ds_eager, kwargs)["MB"]
    # Lazy MB
    res_lazy_mb = compute_statistics("test_lazy", ["MB"], ds_lazy, kwargs)["MB"]

    assert hasattr(res_lazy_mb.data, "dask"), "MB result should be lazy-backed"
    xr.testing.assert_allclose(res_eager_mb, res_lazy_mb.compute(), atol=1e-15)
    assert "Computed MB" in res_eager_mb.attrs["history"]

    # 6. Manual weighted verification for MB (ensures production logic is correct)
    diff = ds_eager.mod - ds_eager.obs
    w_2d = ds_eager.w.broadcast_like(ds_eager.obs)
    expected_mb = (diff * w_2d).sum() / w_2d.sum()
    xr.testing.assert_allclose(res_eager_mb, expected_mb, atol=1e-15)

    print("\n✅ Production Fallback Logic Verified: Eager == Lazy (Dask)")


if __name__ == "__main__":
    pytest.main([__file__])
