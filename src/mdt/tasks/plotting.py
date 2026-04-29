import logging
from typing import Any, Dict, Union

import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def generate_plot(
    name: str,
    plot_type: str,
    input_data: Union[xr.Dataset, xr.DataArray, pd.DataFrame, Dict[str, Any]],
    kwargs: Dict[str, Any],
    track: str = "A",
) -> Any:
    """
    Generate visualizations using the Two-Track Rule.

    Track A (Publication): matplotlib + cartopy.
    Track B (Exploration): hvplot / geoviews.

    Parameters
    ----------
    name : str
        The identifier for this plotting task.
    plot_type : str
        The type of plot to generate (e.g., 'spatial', 'scatter', 'timeseries').
    input_data : xarray.Dataset, xarray.DataArray, pandas.DataFrame, or dict
        The data to plot.
    kwargs : dict
        Additional keyword arguments to pass to the plotting function.
    track : str, optional
        Visualization track: 'A' for Static, 'B' for Interactive. Default is 'A'.

    Returns
    -------
    object
        The plot object (Matplotlib figure or HoloViews object).

    Examples
    --------
    >>> fig = generate_plot("my_plot", "spatial", ds, {"savename": "spatial.png"}, track="A")
    """
    logger.info("Generating plot '%s' [Track %s] of type: %s", name, track, plot_type)

    if track == "A":
        return _generate_static_plot(name, plot_type, input_data, kwargs)
    elif track == "B":
        return _generate_interactive_plot(name, plot_type, input_data, kwargs)
    else:
        raise ValueError(f"Unknown track '{track}'. Use 'A' (Static) or 'B' (Interactive).")


def _generate_static_plot(name, plot_type, input_data, kwargs) -> Any:
    """Track A: Static plotting with monet-plots (Matplotlib + Cartopy)."""
    import monet_plots

    savename = kwargs.pop("savename", f"{name}.png")

    # Orchestrator Rule: Dispatch to monet_plots class-based API
    if "spatial" in plot_type.lower():
        # Use SpatialImshowPlot for 2D grids (standard for model/obs)
        plot_obj = monet_plots.SpatialImshowPlot(input_data, **kwargs)
        plot_obj.plot(**kwargs)
        plot_obj.save(savename)
        logger.info("Saved Track A plot '%s' to %s via monet-plots", name, savename)
        plot_obj.close()
        return plot_obj
    elif "scatter" in plot_type.lower():
        plot_obj = monet_plots.ScatterPlot(input_data, **kwargs)
        plot_obj.plot(**kwargs)
        plot_obj.save(savename)
        plot_obj.close()
        return plot_obj
    elif "timeseries" in plot_type.lower():
        # TimeSeriesPlot usually expects a DataFrame
        data = input_data.to_dataframe() if hasattr(input_data, "to_dataframe") else input_data
        plot_obj = monet_plots.TimeSeriesPlot(data, **kwargs)
        plot_obj.save(savename)
        plot_obj.close()
        return plot_obj
    elif "taylor" in plot_type.lower():
        # TaylorDiagramPlot usually expects a DataFrame
        data = input_data.to_dataframe() if hasattr(input_data, "to_dataframe") else input_data
        plot_obj = monet_plots.TaylorDiagramPlot(data, **kwargs)
        plot_obj.save(savename)
        plot_obj.close()
        return plot_obj

    # Fallback to general base class if specific class not found/needed
    raise NotImplementedError(f"Static plot type '{plot_type}' not yet implemented in mdt orchestrator.")


def _generate_interactive_plot(name, plot_type, input_data, kwargs) -> Any:
    """Track B: Interactive plotting with monet-plots (HvPlot/GeoViews)."""
    import monet_plots

    # Orchestrator Rule: Dispatch to monet_plots class-based API
    if "spatial" in plot_type.lower():
        plot_obj = monet_plots.SpatialImshowPlot(input_data, **kwargs)
        logger.info("Generated Track B interactive plot '%s' via monet-plots", name)
        return plot_obj.hvplot(**kwargs)
    elif "timeseries" in plot_type.lower():
        data = input_data.to_dataframe() if hasattr(input_data, "to_dataframe") else input_data
        plot_obj = monet_plots.TimeSeriesPlot(data, **kwargs)
        logger.info("Generated Track B interactive plot '%s' via monet-plots", name)
        return plot_obj.hvplot(**kwargs)

    raise NotImplementedError(f"Interactive plot type '{plot_type}' not yet implemented in mdt orchestrator.")
