import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

from mdt.tasks.data import load_data
from mdt.tasks.pairing import pair_data
from mdt.tasks.statistics import compute_statistics


def test_load_data_provenance(mocker):
    """Test load_data and ensure it adds provenance metadata."""
    # Create dummy dataset
    ds = xr.Dataset({"val": (("x"), [1, 2, 3])})

    # Mock monetio.load
    mocker.patch("monetio.load", return_value=ds)

    # Load data
    res = load_data("test_data", "cmaq", {"files": "dummy.nc"})

    # Check history
    assert "Loaded dataset 'test_data'" in res.attrs["history"]


def test_pair_data_double_check(mocker):
    """Test pair_data with both Eager (NumPy) and Lazy (Dask) data (Aero Protocol)."""
    # Setup mock data (NumPy)
    data = np.random.rand(10, 10)
    lat = np.linspace(0, 10, 10)
    lon = np.linspace(0, 10, 10)
    ds_eager = xr.Dataset({"val": (("lat", "lon"), data)}, coords={"lat": lat, "lon": lon})

    # Setup mock data (Dask)
    ds_lazy = ds_eager.chunk({"lat": 5, "lon": 5})

    import monet.util.interp_util as iu

    # Use mocker.patch with create=True if it doesn't exist
    mocker.patch.object(iu, "points_to_dataset", side_effect=lambda src, tgt, **kwargs: src, create=True)

    # Eager run
    res_eager = pair_data("test_eager", "interpolate", ds_eager, ds_eager, {})

    # Lazy run
    res_lazy = pair_data("test_lazy", "interpolate", ds_lazy, ds_lazy, {})

    # Check that the lazy run is still lazy
    assert hasattr(res_lazy.val.data, "dask")

    # Double-Check: Results must be identical after compute
    xr.testing.assert_allclose(res_eager, res_lazy.compute())

    # Check history update
    assert "Paired using method 'interpolate'" in res_eager.attrs["history"]


def test_compute_statistics_double_check(mocker):
    """Test compute_statistics with both Eager (NumPy) and Lazy (Dask) data (Aero Protocol)."""
    # Metrics and kwargs
    metrics = ["RMSE"]
    kwargs = {"obs_var": "obs", "mod_var": "mod"}

    # Mock monet_stats.RMSE
    def mock_rmse(obs, mod, **kwargs):
        if hasattr(obs, "data") and hasattr(obs.data, "dask"):
            # Return a Lazy DataArray
            res = xr.DataArray(da.from_array(0.5, chunks=()), attrs={"history": "initial"})
            return res
        elif hasattr(obs, "attrs"):
            # Return a NumPy DataArray
            return xr.DataArray(0.5, attrs={"history": "initial"})
        return 0.5

    import monet_stats

    # We need to patch BOTH RMSE and rmse if the code searches for them
    mocker.patch.object(monet_stats, "RMSE", side_effect=mock_rmse, create=True)
    mocker.patch.object(monet_stats, "rmse", side_effect=mock_rmse, create=True)

    # 1. Eager Run (Pandas)
    df = pd.DataFrame({"obs": np.random.rand(10), "mod": np.random.rand(10)})
    results_df = compute_statistics("test_stats", metrics, df, kwargs)
    assert results_df["RMSE"] == 0.5

    # 2. Eager Run (Xarray NumPy)
    ds_eager = xr.Dataset({"obs": (("x"), np.random.rand(10)), "mod": (("x"), np.random.rand(10))})
    results_eager = compute_statistics("test_eager_stats", metrics, ds_eager, kwargs)
    assert float(results_eager["RMSE"]) == 0.5
    assert "Computed RMSE" in results_eager["RMSE"].attrs["history"]

    # 3. Lazy Run (Xarray Dask)
    ds_lazy = xr.Dataset(
        {
            "obs": (("x"), da.from_array(np.random.rand(10), chunks=5)),
            "mod": (("x"), da.from_array(np.random.rand(10), chunks=5)),
        }
    )
    results_lazy = compute_statistics("test_lazy_stats", metrics, ds_lazy, kwargs)

    # Ensure result is still lazy if input was lazy and result is an xarray object
    if hasattr(results_lazy["RMSE"], "data"):
        assert hasattr(results_lazy["RMSE"].data, "dask")

    # Double-Check: Results must be identical
    assert float(results_lazy["RMSE"].compute()) == float(results_eager["RMSE"])
    assert "Computed RMSE" in results_lazy["RMSE"].attrs["history"]
