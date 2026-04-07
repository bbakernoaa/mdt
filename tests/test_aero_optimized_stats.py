import numpy as np
import pytest
import xarray as xr
from mdt.tasks.statistics import compute_statistics


def test_weighted_rmse_aero_protocol_double_check():
    """
    Aero Protocol Double-Check: Weighted RMSE.

    Verifies that Eager (NumPy) and Lazy (Dask) backends yield identical
    results for weighted RMSE using the orchestrator's fallback logic.
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

    # 3. Execution
    metrics = ["RMSE"]
    kwargs = {"obs_var": "obs", "mod_var": "mod", "weights": "w"}

    res_eager = compute_statistics("test_eager", metrics, ds_eager, kwargs)["RMSE"]
    res_lazy = compute_statistics("test_lazy", metrics, ds_lazy, kwargs)["RMSE"]

    # 4. Assertions
    # Verify laziness
    assert hasattr(res_lazy.data, "dask"), "Result should be lazy-backed for lazy input"

    # Verify identical results (Double-Check Rule)
    xr.testing.assert_allclose(res_eager, res_lazy.compute(), atol=1e-15)

    # Verify provenance
    assert "Computed RMSE" in res_eager.attrs["history"]
    print("\n✅ Weighted RMSE Double-Check Passed: Eager == Lazy (Dask)")


def test_native_weighted_mb_aero_protocol():
    """Verify that metrics with native weights support (like MB/Bias) are called correctly."""
    # 1. Setup Synthetic Grid Data
    lats = np.linspace(-90, 90, 10)
    lons = np.linspace(-180, 180, 20)
    rng = np.random.default_rng(42)
    obs_data = rng.standard_normal((len(lats), len(lons)))
    mod_data = obs_data + 0.5  # Constant bias

    ds = xr.Dataset(
        {
            "obs": (("lat", "lon"), obs_data),
            "mod": (("lat", "lon"), mod_data),
        },
        coords={"lat": lats, "lon": lons},
    )
    # Add uniform weights
    ds.coords["w"] = xr.DataArray(np.ones(len(lats)), coords={"lat": lats}, dims=["lat"])

    # 2. Execution
    metrics = ["MB"]
    kwargs = {"obs_var": "obs", "mod_var": "mod", "weights": "w"}

    # MB in monet-stats has native weights support
    results = compute_statistics("test_mb", metrics, ds, kwargs)
    res = results["MB"]

    # 3. Assertions
    # For uniform weights and constant bias, MB should be exactly 0.5
    np.testing.assert_allclose(res.values, 0.5, atol=1e-15)
    assert "Computed MB" in res.attrs["history"]
    print("\n✅ Native Weighted MB Passed")


if __name__ == "__main__":
    pytest.main([__file__])
