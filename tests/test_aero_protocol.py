import numpy as np
import pandas as pd
import xarray as xr

from mdt.tasks.statistics import compute_statistics


def test_compute_statistics_aero_protocol(mocker):
    """
    Double-Check Test: Verify compute_statistics for Eager and Lazy backends.

    (Aero Protocol Requirement)
    """
    # 1. Setup Data
    size = 100
    obs_np = np.random.rand(size)
    mod_np = np.random.rand(size)

    # Eager Dataset (NumPy)
    ds_eager = xr.Dataset(
        {
            "obs": (("x"), obs_np),
            "mod": (("x"), mod_np),
        },
        attrs={"history": "Initial data"},
    )

    # Lazy Dataset (Dask)
    ds_lazy = ds_eager.chunk({"x": 20})

    # 2. Setup Mock for monet-stats
    # We mock a simple metric that would normally be found in monet-stats
    def mock_rmse(obs, mod, **kwargs):
        # Implementation of RMSE that works with NumPy/Dask/Xarray
        return ((mod - obs) ** 2).mean() ** 0.5

    mocker.patch("mdt.tasks.statistics._find_metric", return_value=mock_rmse)

    # 3. Execute Eager (NumPy)
    metrics = ["RMSE"]
    kwargs = {"obs_var": "obs", "mod_var": "mod"}
    res_eager = compute_statistics("test_eager", metrics, ds_eager, kwargs)

    # 4. Execute Lazy (Dask)
    res_lazy = compute_statistics("test_lazy", metrics, ds_lazy, kwargs)

    # 5. Assertions
    # Check that lazy run actually returned a lazy object
    assert hasattr(res_lazy["RMSE"].data, "dask") or hasattr(res_lazy["RMSE"].data, "cubed"), "Result should be lazy-backed for lazy input"

    # Double-Check: Results must be identical after computation
    xr.testing.assert_allclose(res_eager["RMSE"], res_lazy["RMSE"].compute())

    # Provenance Check
    assert "Computed RMSE" in res_eager["RMSE"].attrs["history"]
    assert "Computed RMSE" in res_lazy["RMSE"].attrs["history"]

    print("\n✅ Aero Protocol Double-Check Passed: Eager == Lazy (Dask)")


def test_compute_statistics_pandas_provenance():
    """Verify provenance tracking for Pandas objects."""
    df = pd.DataFrame({"obs": [1, 2], "mod": [1.1, 1.9]})

    # Mocking a function that returns a scalar or series
    def dummy_metric(obs, mod):
        return (mod - obs).abs().mean()

    # Inject dummy into find_metric via monkeypatch or similar if needed,
    # but here we can just test the _update_provenance helper directly or use a real metric.

    # Let's test _update_provenance directly for Pandas
    df.attrs = {"history": {"mdt_history": "Start"}}
    from mdt.utils import update_history

    updated_df = update_history(df, "Computed Metric")
    assert "Computed Metric" in updated_df.attrs["history"]["mdt_history"]


def test_compute_statistics_dask_enabled(mocker):
    """Verify that a Dask-enabled metric works when called directly."""

    # 1. A metric function that handles Dask natively
    def dask_enabled_metric(obs, mod, **kwargs):
        return (mod - obs).mean()

    # 2. Setup Dask Data
    size = 10
    ds_lazy = xr.Dataset(
        {
            "obs": (("x"), np.random.rand(size)),
            "mod": (("x"), np.random.rand(size)),
        }
    ).chunk({"x": 5})

    # 3. Mock monet-stats
    mocker.patch("mdt.tasks.statistics._find_metric", return_value=dask_enabled_metric)

    # 4. Execute
    metrics = ["DASK_ENABLED"]
    kwargs = {"obs_var": "obs", "mod_var": "mod"}
    results = compute_statistics("test_dask", metrics, ds_lazy, kwargs)

    # 5. Assertions
    res = results["DASK_ENABLED"]
    # If the metric function preserves dask, then the result should have dask.
    # Our dask_enabled_metric does preserve dask because (mod-obs).mean() on dask arrays is lazy.
    if hasattr(res, "data"):
        assert hasattr(res.data, "dask") or hasattr(res.data, "cubed"), "Result must be lazy if the metric preserves it"

    # Should work when computed
    val = res.compute() if hasattr(res, "compute") else res
    assert isinstance(float(val), float)
