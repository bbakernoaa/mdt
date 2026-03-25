import logging
from typing import Any, Dict, Union

import matplotlib.pyplot as plt
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
    """Track A: Static plotting with Matplotlib and Cartopy."""
    import importlib

    import cartopy.crs as ccrs

    module_path = f"monet_plots.{plot_type}"
    try:
        plot_module = importlib.import_module(module_path)
    except ImportError:
        logger.error("monet-plots module %s not found.", module_path)
        raise

    # Look for the plotting function
    func = getattr(plot_module, "plot", None)
    if func is None:
        possible_funcs = [
            attr
            for attr in dir(plot_module)
            if plot_type in attr.lower() and callable(getattr(plot_module, attr))
        ]
        if possible_funcs:
            func = getattr(plot_module, possible_funcs[0])
        else:
            raise AttributeError(f"Could not find a plotting function in {module_path}")

    # Enforce Cartopy for spatial plots
    is_spatial = "spatial" in plot_type.lower()
    projection = None
    if is_spatial:
        if "map_kwargs" not in kwargs:
            kwargs["map_kwargs"] = {}
        if "projection" not in kwargs["map_kwargs"]:
            kwargs["map_kwargs"]["projection"] = ccrs.PlateCarree()
        projection = kwargs["map_kwargs"]["projection"]
        if "transform" not in kwargs:
            kwargs["transform"] = ccrs.PlateCarree()

    if "ax" not in kwargs:
        subplot_kw = {}
        if projection:
            subplot_kw["projection"] = projection
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (10, 8)), subplot_kw=subplot_kw)
        kwargs["ax"] = ax

    savename = kwargs.pop("savename", f"{name}.png")

    plot_obj = func(input_data, **kwargs)

    if "ax" in kwargs and kwargs["ax"] is not None:
        fig = kwargs["ax"].figure
        fig.savefig(savename, bbox_inches="tight")
        logger.info("Saved Track A plot '%s' to %s", name, savename)
        plt.close(fig)

    return plot_obj


def _generate_interactive_plot(name, plot_type, input_data, kwargs) -> Any:
    """Track B: Interactive plotting with HvPlot/GeoViews."""
    import hvplot.pandas  # noqa: F401
    import hvplot.xarray  # noqa: F401

    # Mandatory: rasterize=True for large grids in Track B
    if "rasterize" not in kwargs:
        kwargs["rasterize"] = True

    # Use hvplot dispatch
    if hasattr(input_data, "hvplot"):
        # If it's a spatial plot, we might need geoviews
        if "spatial" in plot_type.lower() or "geo" in kwargs.get("features", []):
            kwargs["geo"] = True

        plot_obj = input_data.hvplot(**kwargs)
        logger.info("Generated Track B interactive plot '%s'", name)
        return plot_obj
    else:
        raise TypeError(f"Input data of type {type(input_data)} does not support hvplot.")
