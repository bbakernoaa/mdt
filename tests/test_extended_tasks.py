import numpy as np
import pandas as pd
import xarray as xr
import pytest
from mdt.tasks.statistics import compute_statistics
from mdt.tasks.plotting import generate_plot

def test_compute_statistics_no_manual_fallbacks(mocker):
    """Test that manual weighted fallbacks are NOT used anymore."""
    import monet_stats

    # Mock monet_stats.weighted_spatial_mean to ensure it's NOT called by the orchestrator fallback
    mock_wsm = mocker.patch("monet_stats.weighted_spatial_mean")

    # Mock MB to NOT have 'weights' in its signature
    def mock_mb_no_weights(obs, mod, axis=None): return xr.DataArray(0.5)
    mocker.patch.object(monet_stats, "MB", mock_mb_no_weights, create=True)
    mock_mb_no_weights.__name__ = "MB"

    ds = xr.Dataset({
        "obs": (("lat", "lon"), np.random.rand(10, 10)),
        "mod": (("lat", "lon"), np.random.rand(10, 10)),
        "w": (("lat", "lon"), np.random.rand(10, 10))
    }, coords={"lat": np.arange(10), "lon": np.arange(10)})

    kwargs = {"obs_var": "obs", "mod_var": "mod", "weights": "w"}

    # Test MB - should call the mock_mb_no_weights directly (unweighted) and NOT the fallback wsm
    res = compute_statistics("test_mb", ["MB"], ds, kwargs)
    assert float(res["MB"]) == 0.5
    assert not mock_wsm.called

def test_generate_plot_dynamic_discovery(mocker):
    """Test generate_plot for dynamic discovery of monet-plots classes."""
    import monet_plots

    # Mock a new plot class that doesn't have a priority mapping
    mock_custom_plot = mocker.Mock()
    mocker.patch.object(monet_plots, "CustomPlot", mock_custom_plot, create=True)

    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})

    # Test discovery of 'Custom' (should find CustomPlot)
    generate_plot("test_custom", "Custom", df, {"savename": "custom.png"}, track="A")
    monet_plots.CustomPlot.assert_called_once()
