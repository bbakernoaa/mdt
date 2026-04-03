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

    print("\n✅ Aero Protocol Double-Check Passed: Weighted MB Eager == Lazy (Dask)")


def test_compute_statistics_weighted_correlation_double_check():
    """Aero Protocol: Double-Check Test for Weighted Correlation."""
    # 1. Setup Data
    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(-180, 180, 10)
    weights = np.cos(np.deg2rad(lat))
    weights_2d = np.broadcast_to(weights[:, np.newaxis], (10, 10))

    ds_eager = xr.Dataset(
        {
            "obs": (("lat", "lon"), np.random.rand(10, 10)),
            "mod": (("lat", "lon"), np.random.rand(10, 10)),
            "w": (("lat", "lon"), weights_2d),
        },
        coords={"lat": lat, "lon": lon},
    )
    ds_lazy = ds_eager.chunk({"lat": 5, "lon": 5})

    # 2. Execute
    metrics = ["correlation"]
    kwargs = {"obs_var": "obs", "mod_var": "mod", "weights": "w", "dim": ("lat", "lon")}

    results_eager = compute_statistics("test_corr_eager", metrics, ds_eager, kwargs)
    res_eager = results_eager["correlation"]

    results_lazy = compute_statistics("test_corr_lazy", metrics, ds_lazy, kwargs)
    res_lazy = results_lazy["correlation"]

    # 3. Assertions
    assert hasattr(res_lazy.data, "dask"), "Result should be Dask-backed"
    np.testing.assert_allclose(res_eager.values, res_lazy.compute().values)

    # Manual verification of weighted correlation
    w = ds_eager.w
    x = ds_eager.mod
    y = ds_eager.obs

    def w_mean(arr, weights):
        return (arr * weights).sum() / weights.sum()

    ex = w_mean(x, w)
    ey = w_mean(y, w)
    exy = w_mean(x * y, w)
    ex2 = w_mean(x**2, w)
    ey2 = w_mean(y**2, w)

    cov = exy - ex * ey
    var_x = ex2 - ex**2
    var_y = ey2 - ey**2
    expected = cov / (np.sqrt(var_x) * np.sqrt(var_y))

    np.testing.assert_allclose(res_eager.values, expected.values)
    print("\n✅ Aero Protocol Double-Check Passed: Weighted CORR Eager == Lazy (Dask)")
