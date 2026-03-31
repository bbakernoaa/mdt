import numpy as np
import xarray as xr

from mdt.tasks.reductions import spatial_mean


def test_spatial_mean_aero_protocol():
    """
    Aero Protocol: Double-Check Test for spatial_mean.

    Verifies that spatial_mean produces identical results for Eager (NumPy)
    and Lazy (Dask) backends and preserves laziness.
    """
    # 1. Setup Eager Data (NumPy)
    lat = np.linspace(-90, 90, 18)
    lon = np.linspace(-180, 180, 36)
    data = np.random.rand(len(lat), len(lon))

    da_eager = xr.DataArray(
        data,
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
        name="test_data",
        attrs={"units": "m", "history": "initial"},
    )

    # 2. Setup Lazy Data (Dask)
    da_lazy = da_eager.chunk({"lat": 9, "lon": 18})

    # 3. Execute Eager
    res_eager = spatial_mean(da_eager)

    # 4. Execute Lazy
    res_lazy = spatial_mean(da_lazy)

    # 5. Assertions
    # Verify laziness (Aero Protocol Rule 1.2)
    assert hasattr(res_lazy.data, "dask"), "Result must be Dask-backed for lazy input"

    # Verify identical results (Double-Check Rule)
    # Using compute() for comparison
    np.testing.assert_allclose(res_eager.values, res_lazy.compute().values)

    # Manual verification (Weighted Mean with Cosine Latitude)
    weights = xr.DataArray(np.cos(np.deg2rad(lat)), coords={"lat": lat}, dims="lat")
    expected = da_eager.weighted(weights).mean(("lat", "lon"))
    np.testing.assert_allclose(res_eager.values, expected.values)

    # Provenance Check
    assert "Computed area-weighted spatial mean" in res_eager.attrs["history"]


def test_spatial_mean_robust_dimension_discovery():
    """Verify that spatial_mean correctly finds non-standard dimension names."""
    # Custom names
    lats = np.linspace(-90, 90, 10)
    lons = np.linspace(-180, 180, 20)

    # Case 1: Detect by name variant (lowercase 'latitude')
    da1 = xr.DataArray(
        np.ones((10, 20)),
        coords={"latitude": lats, "longitude": lons},
        dims=("latitude", "longitude"),
    )
    res1 = spatial_mean(da1)
    assert "Computed area-weighted spatial mean over latitude and longitude" in res1.attrs["history"]

    # Case 2: Detect by attributes (non-standard names)
    da2 = xr.DataArray(
        np.ones((10, 20)),
        coords={"y_coord": (("y_coord"), lats, {"units": "degrees_north"}), "x_coord": (("x_coord"), lons, {"units": "degrees_east"})},
        dims=("y_coord", "x_coord"),
    )
    res2 = spatial_mean(da2)
    assert "Computed area-weighted spatial mean over y_coord and x_coord" in res2.attrs["history"]


def test_spatial_mean_dataset():
    """Verify spatial_mean works on xarray.Dataset."""
    ds = xr.Dataset(
        {
            "v1": (("lat", "lon"), np.random.rand(10, 10)),
            "v2": (("lat", "lon"), np.random.rand(10, 10)),
        },
        coords={"lat": np.linspace(-90, 90, 10), "lon": np.linspace(-180, 180, 10)},
    )

    res = spatial_mean(ds)
    assert isinstance(res, xr.Dataset)
    assert "v1" in res.data_vars
    assert "v2" in res.data_vars
    assert "lat" not in res.dims
    assert "lon" not in res.dims
