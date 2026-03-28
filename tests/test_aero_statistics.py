import numpy as np
import xarray as xr

from mdt.tasks.statistics import compute_statistics


def test_compute_statistics_aero_protocol_double_check(mocker):
    """
    Aero Protocol: Double-Check Test.

    Verifies that compute_statistics yields identical results for Eager (NumPy)
    and Lazy (Dask) backends when using xr.apply_ufunc.
    """
    # 1. Setup Eager Data (NumPy)
    size = 100
    obs_np = np.random.rand(size)
    mod_np = np.random.rand(size)

    ds_eager = xr.Dataset(
        {
            "obs": (("x"), obs_np),
            "mod": (("x"), mod_np),
        },
        attrs={"history": "Initial data"},
    )

    # 2. Setup Lazy Data (Dask)
    # We chunk the data to ensure it's Dask-backed
    ds_lazy = ds_eager.chunk({"x": 25})

    # 3. Setup Mock Metric (monet-stats compatible)
    # This function handles Xarray (NumPy or Dask) directly.
    def mock_rmse(obs, mod, **kwargs):
        return ((mod - obs) ** 2).mean() ** 0.5

    # Mock the discovery of the metric
    mocker.patch("mdt.tasks.statistics._find_metric", return_value=mock_rmse)
    mocker.patch("importlib.import_module")  # Mock import of monet_stats

    # 4. Execute Eager (NumPy)
    metrics = ["RMSE"]
    kwargs = {"obs_var": "obs", "mod_var": "mod"}
    results_eager = compute_statistics("test_eager", metrics, ds_eager, kwargs)
    res_eager = results_eager["RMSE"]

    # 5. Execute Lazy (Dask)
    results_lazy = compute_statistics("test_lazy", metrics, ds_lazy, kwargs)
    res_lazy = results_lazy["RMSE"]

    # 6. Assertions
    # Verify laziness (Aero Protocol Rule 1.2)
    assert hasattr(res_lazy.data, "dask"), "Result should be Dask-backed for lazy input"

    # Verify identical results (Double-Check Rule)
    # We compute the lazy result for comparison
    np.testing.assert_allclose(res_eager.values, res_lazy.compute().values)

    # Verify provenance (Aero Protocol Rule 2.3)
    assert "Computed RMSE" in res_eager.attrs["history"]
    assert "Computed RMSE" in res_lazy.attrs["history"]

    print("\n✅ Aero Protocol Double-Check Passed: Eager == Lazy (Dask)")
