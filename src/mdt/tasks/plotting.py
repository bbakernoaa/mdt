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


def _find_plot_class(plot_type: str) -> Any:
    """
    Dynamically discover a plot class in monet_plots based on plot_type.

    Parameters
    ----------
    plot_type : str
        The name of the plot type (e.g., 'spatial', 'scatter', 'timeseries').

    Returns
    -------
    Any
        The plot class from monet_plots.

    Raises
    ------
    ValueError
        If no matching plot class is found.
    """
    import monet_plots

    # Priority mapping for common names
    mapping = {
        "spatial": "SpatialImshowPlot",
        "scatter": "ScatterPlot",
        "timeseries": "TimeSeriesPlot",
        "taylor": "TaylorDiagramPlot",
    }

    class_name = mapping.get(plot_type.lower())
    if class_name and hasattr(monet_plots, class_name):
        cls = getattr(monet_plots, class_name)
        logger.debug(f"Found plot class '{class_name}' for type '{plot_type}' via mapping.")
        return cls

    # Fallback: Try to find a class that matches the plot_type string (case-insensitive)
    for attr in dir(monet_plots):
        if attr.lower() == plot_type.lower() or attr.lower() == f"{plot_type.lower()}plot":
            cls = getattr(monet_plots, attr)
            logger.debug(f"Found plot class '{attr}' for type '{plot_type}' via fallback search.")
            return cls

    raise ValueError(f"Could not find a plot class in monet_plots for type: '{plot_type}'")


def _generate_static_plot(
    name: str,
    plot_type: str,
    input_data: Union[xr.Dataset, xr.DataArray, pd.DataFrame, Dict[str, Any]],
    kwargs: Dict[str, Any],
) -> Any:
    """Track A: Static plotting with monet-plots (Matplotlib + Cartopy)."""
    savename = kwargs.pop("savename", f"{name}.png")

    try:
        plot_class = _find_plot_class(plot_type)
        if not callable(plot_class):
            raise ValueError(f"Discovered plot type '{plot_type}' is not a valid callable class.")

        # MDT Architecture Rule: Data remains in its native format (Xarray) during orchestration.
        # Plot classes in monet-plots are responsible for any internal data conversions.
        data = input_data

        # Filter kwargs: separate constructor args from plot() args
        # For now, we assume most kwargs can go to both, but we MUST exclude 'savename'
        # which we already popped.
        plot_obj = plot_class(data, **kwargs)

        # Dispatch to plot() method if it exists (for standard Matplotlib rendering)
        if hasattr(plot_obj, "plot"):
            # We pass kwargs again to plot() as many monet-plots use this pattern
            plot_obj.plot(**kwargs)

        plot_obj.save(savename)
        logger.info("Saved Track A plot '%s' to %s via monet-plots", name, savename)

        if hasattr(plot_obj, "close"):
            plot_obj.close()

        return plot_obj

    except ValueError as e:
        raise NotImplementedError(f"Static plot type '{plot_type}' not supported: {e}") from e


def _generate_interactive_plot(
    name: str,
    plot_type: str,
    input_data: Union[xr.Dataset, xr.DataArray, pd.DataFrame, Dict[str, Any]],
    kwargs: Dict[str, Any],
) -> Any:
    """Track B: Interactive plotting with monet-plots (HvPlot/GeoViews)."""
    # Pop savename if it exists, though Track B usually doesn't save to file immediately
    kwargs.pop("savename", None)

    try:
        plot_class = _find_plot_class(plot_type)
        if not callable(plot_class):
            raise ValueError(f"Discovered plot type '{plot_type}' is not a valid callable class.")

        # MDT Architecture Rule: Data remains in its native format (Xarray) during orchestration.
        # Plot classes in monet-plots are responsible for any internal data conversions.
        data = input_data

        plot_obj = plot_class(data, **kwargs)

        if not hasattr(plot_obj, "hvplot"):
            raise AttributeError(f"Plot class '{plot_class.__name__}' does not support interactive hvplot.")

        logger.info("Generated Track B interactive plot '%s' via monet-plots", name)
        return plot_obj.hvplot(**kwargs)

    except (ValueError, AttributeError) as e:
        raise NotImplementedError(f"Interactive plot type '{plot_type}' not supported: {e}") from e
