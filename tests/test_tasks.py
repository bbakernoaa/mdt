import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

from mvs.tasks.data import load_data
from mvs.tasks.pairing import pair_data
from mvs.tasks.statistics import compute_statistics


def test_load_data_provenance(mocker):
    """Test load_data and ensure it adds provenance metadata."""
    # Create dummy dataset
    ds = xr.Dataset({"val": (("x"), [1, 2, 3])})

    # Mock monetio.datasets.cmaq.open_dataset
    mock_module = mocker.Mock()
    mock_module.open_dataset.return_value = ds
    mocker.patch("importlib.import_module", return_value=mock_module)

    # Load data
    res = load_data("test_data", "cmaq", {"fname": "dummy.nc"})

    # Check history
    assert "Loaded dataset 'test_data'" in res.attrs["history"]


def test_pair_data_double_check(mocker):
    """Test pair_data with both Eager (NumPy) and Lazy (Dask) data."""
    # Setup mock data (NumPy)
    data = np.random.rand(10, 10)
    lat = np.linspace(0, 10, 10)
    lon = np.linspace(0, 10, 10)
    ds_eager = xr.Dataset({"val": (("lat", "lon"), data)}, coords={"lat": lat, "lon": lon})

    # Setup mock data (Dask)
    ds_lazy = ds_eager.chunk({"lat": 5, "lon": 5})

    # Mock the monet.util.interp_util.points_to_dataset
    mocker.patch("monet.util.interp_util.points_to_dataset", return_value=ds_eager)

    # Eager run
    res_eager = pair_data("test_eager", "interpolate", ds_eager, ds_eager, {})

    # Lazy run
    _ = pair_data("test_lazy", "interpolate", ds_lazy, ds_lazy, {})

    # Check history update
    assert "Paired using method 'interpolate'" in res_eager.attrs["history"]


def test_compute_statistics_double_check(mocker):
    """Test compute_statistics with both Eager (NumPy) and Lazy (Dask) data."""
    # Setup mock data
    df = pd.DataFrame({"obs": np.random.rand(10), "mod": np.random.rand(10)})

    # Mock monet.util.stats.rmse
    mocker.patch("monet.util.stats.rmse", return_value=0.5)

    metrics = ["rmse"]
    kwargs = {"obs_var": "obs", "mod_var": "mod"}

    # Eager run (DataFrame)
    results = compute_statistics("test_stats", metrics, df, kwargs)
    assert results["rmse"] == 0.5

    # Lazy run (Dataset with Dask)
    ds_lazy = xr.Dataset({"obs": (("x"), da.from_array(np.random.rand(10), chunks=5)), "mod": (("x"), da.from_array(np.random.rand(10), chunks=5))})

    # Mock return value as a DataArray to check history
    res_da = xr.DataArray(0.5, attrs={"history": "initial"})
    mocker.patch("monet.util.stats.rmse", return_value=res_da)

    results = compute_statistics("test_lazy_stats", metrics, ds_lazy, kwargs)
    assert float(results["rmse"]) == 0.5
    assert "Computed rmse" in results["rmse"].attrs["history"]
