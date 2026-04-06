import numpy as np
import xarray as xr

from mdt.tasks.reductions import calculate_reduction
from mdt.tasks.statistics import compute_statistics


def test_weighted_stats_double_check():
    """
    Aero Protocol: Double-Check Test.

    Verifies weighted statistics produce identical results for Eager (NumPy) and Lazy (Dask) backends.
    """
    # 1. Create Synthetic Data
    lat = np.linspace(-90, 90, 18)
    lon = np.linspace(-180, 180, 36)

    data_vars = {
        "obs": (["lat", "lon"], np.random.rand(18, 36)),
        "mod": (["lat", "lon"], np.random.rand(18, 36)),
        "w": (["lat", "lon"], np.cos(np.deg2rad(lat))[:, np.newaxis] * np.ones(36)),
    }

    ds_eager = xr.Dataset(data_vars, coords={"lat": lat, "lon": lon})
    ds_lazy = ds_eager.chunk({"lat": 9, "lon": 18})

    metrics = ["rmse", "pearsonr"]
    kwargs = {"obs_var": "obs", "mod_var": "mod", "weights": "w"}

    # 2. Compute Eager (NumPy)
    results_eager = compute_statistics("test_eager", metrics, ds_eager, kwargs)

    # 3. Compute Lazy (Dask)
    results_lazy = compute_statistics("test_lazy", metrics, ds_lazy, kwargs)

    # 4. Assert Equivalence
    for m in metrics:
        xr.testing.assert_allclose(results_eager[m], results_lazy[m])
        # Ensure lazy result is actually lazy
        assert hasattr(results_lazy[m].data, "dask")


def test_reduction_double_check():
    """Aero Protocol: Double-Check Test for reductions."""
    lat = np.linspace(-90, 90, 18)
    lon = np.linspace(-180, 180, 36)
    da_eager = xr.DataArray(np.random.rand(18, 36), coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
    da_lazy = da_eager.chunk({"lat": 9, "lon": 18})

    # Weighted Mean
    res_eager = calculate_reduction(da_eager, method="mean", dim=["lat", "lon"], force_weighted=True)
    res_lazy = calculate_reduction(da_lazy, method="mean", dim=["lat", "lon"], force_weighted=True)

    xr.testing.assert_allclose(res_eager, res_lazy)
    assert hasattr(res_lazy.data, "dask")
