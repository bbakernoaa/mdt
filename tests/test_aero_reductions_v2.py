import numpy as np
import pytest
import xarray as xr

from mdt.tasks.reductions import calculate_reduction, spatial_mean


def test_calculate_reduction_double_check():
    """
    Aero Protocol: Double-Check Test for generalized reductions.

    Verifies that various reduction methods produce identical results for
    Eager (NumPy) and Lazy (Dask) backends.
    """
    # 1. Setup Eager Data (NumPy)
    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(-180, 180, 10)
    time = np.arange(5)
    data = np.random.rand(5, 10, 10)

    da_eager = xr.DataArray(
        data, coords={"time": time, "lat": lat, "lon": lon}, dims=("time", "lat", "lon"), name="test_data", attrs={"history": "Initial data"}
    )

    # 2. Setup Lazy Data (Dask)
    da_lazy = da_eager.chunk({"time": 1, "lat": 5, "lon": 5})

    methods = ["mean", "max", "min", "std", "sum"]

    for method in methods:
        # 3. Execute Eager (NumPy) - Temporal reduction
        res_eager = calculate_reduction(da_eager, method=method, dim="time")

        # 4. Execute Lazy (Dask) - Temporal reduction
        res_lazy = calculate_reduction(da_lazy, method=method, dim="time")

        # 5. Assertions
        # Verify laziness: The underlying data should be a Dask array
        assert hasattr(res_lazy.data, "dask"), f"Result for {method} should be lazy-backed"

        # Verify identical results
        np.testing.assert_allclose(res_eager.values, res_lazy.compute().values, err_msg=f"Eager and Lazy results differ for method={method}")

        # Verify provenance
        assert f"method='{method}'" in res_eager.attrs["history"]
        assert "dim=time" in res_eager.attrs["history"]

    print("\n✅ Aero Protocol Double-Check Passed: Eager == Lazy (Dask) for multiple reductions")


def test_spatial_mean_delegation():
    """Verify that spatial_mean correctly delegates to calculate_reduction with weighting."""
    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(-180, 180, 10)
    data = np.random.rand(10, 10)

    da = xr.DataArray(data, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"), name="test_data")

    # This should trigger the specialized weighted spatial mean logic
    res = spatial_mean(da)

    # Check provenance for the monet-stats indicator (added by monet-stats or mdt)
    # Actually mdt adds it if it detects spatial.
    assert "area-weighted via monet-stats" in res.attrs["history"]
    # Verify it reduced to a scalar
    assert res.dims == ()


def test_calculate_reduction_combined_dims():
    """Verify that calculate_reduction correctly handles combined spatial + non-spatial dims."""
    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(-180, 180, 10)
    time = np.arange(5)
    data = np.random.rand(5, 10, 10)

    da = xr.DataArray(data, coords={"time": time, "lat": lat, "lon": lon}, dims=("time", "lat", "lon"), name="test_data")

    # Reduce over both time and spatial dimensions
    res = calculate_reduction(da, method="mean", dim=["time", "lat", "lon"])

    # Assertions
    assert "area-weighted via monet-stats" in res.attrs["history"]
    # All dimensions should be gone (reduced to a scalar)
    assert res.dims == ()
    assert not any(d in res.dims for d in ["time", "lat", "lon"])


def test_spatial_mean_nonstandard_names():
    """Verify that spatial_mean still applies weighting for non-standard dimension names."""
    y = np.linspace(-90, 90, 10)
    x = np.linspace(-180, 180, 10)
    data = np.random.rand(10, 10)

    da = xr.DataArray(data, coords={"y": y, "x": x}, dims=("y", "x"), name="test_data")

    # Call spatial_mean with non-standard names
    res = spatial_mean(da, lat_dim="y", lon_dim="x")

    # Verify weighting was applied (force_weighted should trigger it)
    assert "area-weighted via monet-stats" in res.attrs["history"]
    assert res.dims == ()


def test_unsupported_method():
    """Verify that unsupported methods raise ValueError."""
    da = xr.DataArray([1, 2, 3], dims="x")
    with pytest.raises(ValueError, match="Unsupported reduction method"):
        calculate_reduction(da, method="not_a_method")
