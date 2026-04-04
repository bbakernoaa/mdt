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
