import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

from mdt.tasks.data import load_data
from mdt.tasks.pairing import combine_paired_data, pair_data
from mdt.tasks.statistics import compute_statistics


def test_merra2_icap_aeronet_workflow(mocker):
    """Test the full comparison workflow for MERRA-2, ICAP-MME, and AERONET.

    This test uses real Reader classes from monetio (mocked for I/O).
    """
    # 1. Setup Mock Data
    times = pd.date_range("2023-01-01", periods=2, freq="h")
    lats = np.linspace(-90, 90, 10)
    lons = np.linspace(-180, 180, 10)

    # MERRA-2 Mock Dataset (Gridded)
    merra2_ds = xr.Dataset(
        {"TOTEXTTAU": (("time", "lat", "lon"), np.random.rand(2, 10, 10))},
        coords={"time": times, "lat": lats, "lon": lons},
    )

    # ICAP-MME Mock Dataset (Gridded)
    icap_ds = xr.Dataset(
        {"modeaod550": (("time", "lat", "lon"), np.random.rand(2, 10, 10))},
        coords={"time": times, "lat": lats, "lon": lons},
    )

    # AERONET Mock Dataset (Point/UGRID)
    # 5 sites
    aeronet_df = pd.DataFrame(
        {
            "time": np.repeat(times, 5),
            "siteid": np.tile([f"site_{i}" for i in range(5)], 2),
            "latitude": np.tile(np.linspace(-80, 80, 5), 2),
            "longitude": np.tile(np.linspace(-170, 170, 5), 2),
            "AOD_550nm": np.random.rand(10),
        }
    )

    # 2. Mock monetio.load
    def mock_load(source, **kwargs):
        if source == "merra2":
            return merra2_ds
        elif source == "icap_mme":
            return icap_ds
        elif source == "aeronet":
            # Convert DF to Xarray UGRID via AERONETReader logic (simplified here)
            ds = aeronet_df.set_index(["time", "siteid"]).to_xarray()
            ds.attrs["history"] = "Mocked AERONET"
            return ds
        return None

    mocker.patch("monetio.load", side_effect=mock_load)

    # 3. Load Data Tasks
    ds_merra2 = load_data("m2", "merra2", {"dates": "2023-01-01"})
    ds_icap = load_data("icap", "icap_mme", {"date": "2023-01-01"})
    ds_obs = load_data("obs", "aeronet", {"dates": "2023-01-01"})

    assert "TOTEXTTAU" in ds_merra2.data_vars
    assert "modeaod550" in ds_icap.data_vars
    assert "AOD_550nm" in ds_obs.data_vars

    # 4. Pairing Tasks (Interpolation)
    # Mocking interpolation to just return the model data at the same site locations
    # (In a real run, monet.util.interp_util would be used)
    def mock_interp(src, tgt, **kwargs):
        # Return a dataset with the model variable mapped to the observation points
        # For simplicity in this test, we just create a compatible structure
        sites = tgt.siteid.values
        times = tgt.time.values
        varname = list(src.data_vars)[0]
        paired = xr.Dataset(
            {varname: (("time", "siteid"), np.random.rand(len(times), len(sites)))},
            coords={"time": times, "siteid": sites},
        )
        # Merge observations for comparison
        paired["obs"] = tgt["AOD_550nm"]
        return paired

    mocker.patch("monet.util.interp_util.nearest_point_swathdefinition", side_effect=mock_interp, create=True)

    pair_m2 = pair_data("p_m2", "interpolate", ds_merra2, ds_obs, {})
    pair_icap = pair_data("p_icap", "interpolate", ds_icap, ds_obs, {})

    assert "TOTEXTTAU" in pair_m2.data_vars
    assert "modeaod550" in pair_icap.data_vars
    assert "obs" in pair_m2.data_vars

    # 5. Combine Task
    # Rename variables to match for statistics
    pair_m2 = pair_m2.rename({"TOTEXTTAU": "AOD"})
    pair_icap = pair_icap.rename({"modeaod550": "AOD"})

    combined = combine_paired_data({"merra2": pair_m2, "icap": pair_icap}, dim="model")

    assert "model" in combined.dims
    assert combined.sizes["model"] == 2

    # 6. Statistics Task
    metrics = ["RMSE", "MB"]
    stats_kwargs = {"obs_var": "obs", "mod_var": "AOD"}

    # Mock monet_stats
    import monet_stats

    mocker.patch.object(monet_stats, "RMSE", return_value=xr.DataArray(0.1, attrs={"history": ""}))
    mocker.patch.object(monet_stats, "MB", return_value=xr.DataArray(0.05, attrs={"history": ""}))

    # Note: MDT statistics task handles dimensions (e.g. computing per model)
    # Here we test the high-level task execution
    results = compute_statistics("workflow_stats", metrics, combined, stats_kwargs)

    assert "RMSE" in results
    assert "MB" in results


def test_merra2_lazy_loading(mocker):
    """Verify that MERRA-2 loading remains lazy (Aero Protocol)."""
    # Mock xarray.open_mfdataset to return a lazy dataset
    lazy_data = da.random.random((2, 10, 10), chunks=(1, 5, 5))
    ds_lazy = xr.Dataset(
        {"TOTEXTTAU": (("time", "lat", "lon"), lazy_data)},
        coords={
            "time": pd.date_range("2023-01-01", periods=2),
            "lat": np.linspace(-90, 90, 10),
            "lon": np.linspace(-180, 180, 10),
        },
    )

    # We need to mock the Reader.open_dataset or the driver
    from monetio.readers.merra2 import MERRA2Reader

    mocker.patch.object(MERRA2Reader, "open_dataset", return_value=ds_lazy)

    ds = load_data("m2_lazy", "merra2", {"dates": "2023-01-01"})

    # Check if TOTEXTTAU is lazy
    assert hasattr(ds.TOTEXTTAU.data, "dask")
    assert "Loaded dataset 'm2_lazy'" in ds.attrs["history"]
