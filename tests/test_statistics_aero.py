import numpy as np
import pytest
import xarray as xr

from mdt.tasks.statistics import compute_statistics


@pytest.fixture
def sample_data():
    """Create a synthetic dataset for testing."""
    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(-180, 180, 20)
    # Use fixed seed for reproducibility
    rng = np.random.default_rng(42)
    obs_data = rng.standard_normal((10, 20))
    mod_data = obs_data + 0.1 * rng.standard_normal((10, 20))

    ds_numpy = xr.Dataset(
        {
            "obs": (["lat", "lon"], obs_data),
            "mod": (["lat", "lon"], mod_data),
        },
        coords={"lat": lat, "lon": lon},
    )

    # Add area weights (cosine of latitude)
    weights = np.cos(np.deg2rad(ds_numpy.lat))
    # Broadcast to full shape for some tests
    ds_numpy["weights"] = weights + 0 * ds_numpy.obs

    ds_dask = ds_numpy.chunk({"lat": 5, "lon": 10})

    return ds_numpy, ds_dask


@pytest.mark.parametrize("metric", ["MB", "RMSE", "correlation"])
def test_statistics_double_check(sample_data, metric):
    """
    Aero Protocol: Double-Check Test.

    Verify that Eager (NumPy) and Lazy (Dask) backends produce identical results
    and that the Dask backend remains lazy.
    """
    ds_numpy, ds_dask = sample_data
    metrics = [metric]
    # Use weighted version to trigger the refactored logic paths
    kwargs = {"obs_var": "obs", "mod_var": "mod", "weights": "weights", "dim": ["lat", "lon"]}

    # 1. Compute Eager (NumPy)
    res_dict_numpy = compute_statistics(f"test_{metric}_eager", metrics, ds_numpy, kwargs)
    res_numpy = res_dict_numpy[metric]

    # 2. Compute Lazy (Dask)
    res_dict_dask = compute_statistics(f"test_{metric}_lazy", metrics, ds_dask, kwargs)
    res_dask = res_dict_dask[metric]

    # 3. Assertions
    # Ensure values match
    np.testing.assert_allclose(res_numpy.values, res_dask.values, rtol=1e-6)

    # Ensure laziness is preserved for Dask input
    # Note: result might be a scalar wrapped in DataArray
    if hasattr(res_dask, "data"):
        assert hasattr(res_dask.data, "dask"), f"Dask laziness lost for {metric}"

    # Ensure history was updated (Provenance Tracking)
    assert "history" in res_numpy.attrs
    assert metric in res_numpy.attrs["history"]


def test_statistics_no_weights_fallback(sample_data):
    """Verify that standard monet-stats calls work when no weights are provided."""
    ds_numpy, ds_dask = sample_data
    metric = "RMSE"
    metrics = [metric]
    kwargs = {"obs_var": "obs", "mod_var": "mod"}

    res_numpy = compute_statistics("test_rmse_eager", metrics, ds_numpy, kwargs)[metric]
    res_dask = compute_statistics("test_rmse_lazy", metrics, ds_dask, kwargs)[metric]

    np.testing.assert_allclose(res_numpy.values, res_dask.values)
    # Standard monet-stats might compute immediately if not using apply_ufunc internally,
    # but let's check if it preserved laziness.
    # Current monet-stats rmse uses xarray operations which should be lazy.
    if hasattr(res_dask, "data") and not np.isscalar(res_dask.data):
        assert hasattr(res_dask.data, "dask")
