import numpy as np
import pandas as pd
import xarray as xr
import pytest
from mdt.tasks.statistics import compute_statistics
from mdt.tasks.plotting import generate_plot

def test_compute_statistics_weighted_fallbacks(mocker):
    """Test manual weighted fallbacks for MB and MAE."""
    import monet_stats

    # Mock monet_stats.weighted_spatial_mean
    mock_wsm = mocker.patch("monet_stats.weighted_spatial_mean", return_value=xr.DataArray(0.1))

    # Mock MB and MAE to NOT have 'weights' in their signature to trigger fallback
    # We can do this by creating a function without 'weights' parameter
    def mock_mb_no_weights(obs, mod, axis=None): return 0.5
    def mock_mae_no_weights(obs, mod, axis=None): return 0.5

    # We need to make sure _find_metric finds these
    mocker.patch.object(monet_stats, "MB", mock_mb_no_weights, create=True)
    mocker.patch.object(monet_stats, "MAE", mock_mae_no_weights, create=True)

    # Force the names for the fallback logic
    mock_mb_no_weights.__name__ = "MB"
    mock_mae_no_weights.__name__ = "MAE"

    ds = xr.Dataset({
        "obs": (("lat", "lon"), np.random.rand(10, 10)),
        "mod": (("lat", "lon"), np.random.rand(10, 10)),
        "w": (("lat", "lon"), np.random.rand(10, 10))
    }, coords={"lat": np.arange(10), "lon": np.arange(10)})

    kwargs = {"obs_var": "obs", "mod_var": "mod", "weights": "w"}

    # Test MB fallback
    res = compute_statistics("test_mb", ["MB"], ds, kwargs)
    assert float(res["MB"]) == 0.1
    assert mock_wsm.called

    # Test MAE fallback
    mock_wsm.reset_mock()
    res = compute_statistics("test_mae", ["MAE"], ds, kwargs)
    assert float(res["MAE"]) == 0.1
    assert mock_wsm.called

def test_generate_plot_extended(mocker):
    """Test generate_plot for timeseries and taylor types."""
    import monet_plots

    # Mock Plot classes
    mock_ts_plot = mocker.Mock()
    mock_taylor_plot = mocker.Mock()

    mocker.patch("monet_plots.TimeSeriesPlot", return_value=mock_ts_plot)
    mocker.patch("monet_plots.TaylorDiagramPlot", return_value=mock_taylor_plot)

    df = pd.DataFrame({"time": pd.date_range("2023-01-01", periods=10, freq="h"), "obs": np.random.rand(10), "mod": np.random.rand(10)})

    # Test Static TimeSeries
    generate_plot("test_ts", "timeseries", df, {"savename": "ts.png"}, track="A")
    monet_plots.TimeSeriesPlot.assert_called_once()
    mock_ts_plot.save.assert_called_with("ts.png")

    # Test Static Taylor
    generate_plot("test_taylor", "taylor", df, {"savename": "taylor.png"}, track="A")
    monet_plots.TaylorDiagramPlot.assert_called_once()
    mock_taylor_plot.save.assert_called_with("taylor.png")

    # Test Interactive TimeSeries
    mock_ts_plot.hvplot.return_value = "interactive_plot"
    res = generate_plot("test_ts_int", "timeseries", df, {}, track="B")
    assert res == "interactive_plot"
    mock_ts_plot.hvplot.assert_called_once()
