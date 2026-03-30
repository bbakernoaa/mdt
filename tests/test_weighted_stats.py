import numpy as np
import xarray as xr
from mdt.tasks.statistics import compute_statistics


def test_compute_statistics_weighted_double_check(mocker):
    """
    Aero Protocol: Double-Check Test for Weighted Statistics.

    Verifies that weighted statistics yield identical results for Eager (NumPy)
    and Lazy (Dask) backends and preserve laziness.
    """
    # 1. Setup Eager Data (NumPy)
    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(-180, 180, 10)

    # Create lat weights
    weights = np.cos(np.deg2rad(lat))
    weights_2d = np.broadcast_to(weights[:, np.newaxis], (10, 10))

    ds_eager = xr.Dataset(
        {
            "obs": (("lat", "lon"), np.random.rand(10, 10)),
            "mod": (("lat", "lon"), np.random.rand(10, 10)),
            "w": (("lat", "lon"), weights_2d),
        },
        coords={"lat": lat, "lon": lon},
        attrs={"history": "Initial data"},
    )

    # 2. Setup Lazy Data (Dask)
    ds_lazy = ds_eager.chunk({"lat": 5, "lon": 5})

    # 3. Execute Eager (NumPy)
    metrics = ["MB"]
    kwargs = {"obs_var": "obs", "mod_var": "mod", "weights": "w", "dim": ("lat", "lon")}
    results_eager = compute_statistics("test_eager", metrics, ds_eager, kwargs)
    res_eager = results_eager["MB"]

    # 4. Execute Lazy (Dask)
    results_lazy = compute_statistics("test_lazy", metrics, ds_lazy, kwargs)
    res_lazy = results_lazy["MB"]

    # 5. Assertions
    # Verify laziness (Aero Protocol Rule 1.2)
    assert hasattr(res_lazy.data, "dask"), "Result should be Dask-backed for lazy input"

    # Verify identical results (Double-Check Rule)
    np.testing.assert_allclose(res_eager.values, res_lazy.compute().values)

    # Manual verification of weighted mean bias
    diff = ds_eager.mod - ds_eager.obs
    expected = (diff * ds_eager.w).sum() / ds_eager.w.where(diff.notnull()).sum()
    np.testing.assert_allclose(res_eager.values, expected.values)

    # Provenance check
    assert "Computed MB" in res_eager.attrs["history"]

    print("\n✅ Aero Protocol Double-Check Passed: Weighted Eager == Lazy (Dask)")


def test_robust_dimension_discovery(mocker):
    """
    Verify robust spatial dimension discovery.

    Uses metadata (units/axis) and ensures Aero Protocol
    Double-Check (Eager vs Lazy).
    """
    # Setup data with non-standard dimension names but standard attributes
    data_mod = np.random.rand(5, 5)
    data_obs = np.random.rand(5, 5)
    lat_coords = np.linspace(-90, 90, 5)
    lon_coords = np.linspace(-180, 180, 5)

    ds_eager = xr.Dataset(
        {"mod": (("y", "x"), data_mod), "obs": (("y", "x"), data_obs)},
        coords={
            "y": ("y", lat_coords, {"units": "degrees_north"}),
            "x": ("x", lon_coords, {"axis": "X"}),
        },
    )

    # Weights
    weights = xr.DataArray(np.random.rand(5, 5), dims=("y", "x"))

    # Track B: Lazy Data
    ds_lazy = ds_eager.chunk({"y": 3, "x": 3})

    # To truly test the dispatcher, we mock the core monet_stats call but let our logic run
    def mock_w_mean(basis, weights=None, lat_dim="lat", lon_dim="lon"):
        # Simulate area-weighted average
        return (basis * weights).mean()

    mocker.patch("monet_stats.weighted_spatial_mean", side_effect=mock_w_mean)

    # Mock a metric function that would be found
    def mock_bias(obs, mod):
        return mod - obs

    mock_bias.__name__ = "MB"
    mocker.patch("mdt.tasks.statistics._find_metric", return_value=mock_bias)

    kwargs = {"obs_var": "obs", "mod_var": "mod", "weights": weights, "dim": ("x", "y")}

    # 1. Execute Eager
    res_eager = compute_statistics("test_robust_eager", ["MB"], ds_eager, kwargs)["MB"]

    # 2. Execute Lazy
    res_lazy = compute_statistics("test_robust_lazy", ["MB"], ds_lazy, kwargs)["MB"]

    # 3. Assertions
    # Verify laziness
    assert hasattr(res_lazy.data, "dask"), "Result must be Dask-backed for lazy input"

    # Double-Check
    xr.testing.assert_allclose(res_eager, res_lazy.compute())

    # Provenance Tracking check (Aero Protocol Rule 2.3)
    assert "Weighted MB computed" in res_eager.attrs["history"]
    assert "Weighted MB computed" in res_lazy.attrs["history"]

    # Verify Robust Discovery kwargs were passed correctly (inspecting mock)
    # The last call was the lazy one
    import monet_stats

    last_call = monet_stats.weighted_spatial_mean.call_args
    assert last_call.kwargs["lat_dim"] == "y"
    assert last_call.kwargs["lon_dim"] == "x"

    print("\n✅ Aero Protocol: Robust Dimension Discovery & Double-Check Verified")
