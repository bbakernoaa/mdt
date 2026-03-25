import importlib
import logging

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def generate_plot(name, plot_type, input_data, kwargs):
    """
    Generates visualizations using monet-plots.

    Parameters
    ----------
    name : str
        The identifier for this plotting task.
    plot_type : str
        The type of plot to generate (e.g., 'spatial', 'scatter', 'timeseries').
    input_data : xarray.Dataset, pandas.DataFrame, or dict
        The data to plot (raw data, paired data, or computed statistics).
    kwargs : dict
        Additional keyword arguments to pass to the plotting function
        (e.g., `ax`, `savename`, `vmin`, `vmax`, `figsize`).

    Returns
    -------
    object
        The matplotlib plot or figure object returned by monet-plots.

    Raises
    ------
    ImportError
        If monet-plots is not installed, or the requested plot module does not exist.
    AttributeError
        If an appropriate plotting function cannot be found within the module.
    """
    logger.info(f"Generating plot '{name}' of type: {plot_type}")

    try:
        # monet-plots provides modules based on plot type, e.g. scatter, spatial
        # Dynamically import the plot module
        module_path = f"monet_plots.{plot_type}"
        plot_module = importlib.import_module(module_path)

        # Typically monet-plots will have a standard function like `make_spatial_plot` or `plot`
        if hasattr(plot_module, "plot"):
            func = plot_module.plot
        else:
            # Look for a function containing the plot_type string
            possible_funcs = [attr for attr in dir(plot_module) if plot_type in attr.lower() and callable(getattr(plot_module, attr))]
            if possible_funcs:
                func = getattr(plot_module, possible_funcs[0])
            else:
                raise AttributeError(f"Could not find a plotting function for {plot_type} in {module_path}")

        # Create figure and axis if not provided
        if "ax" not in kwargs:
            fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (10, 8)))
            kwargs["ax"] = ax

        savename = kwargs.pop("savename", f"{name}.png")

        # Generate the plot
        logger.debug(f"Calling {func.__name__} with kwargs: {kwargs}")
        plot_obj = func(input_data, **kwargs)

        # Save the figure
        if "ax" in kwargs and kwargs["ax"] is not None:
            fig = kwargs["ax"].figure
            fig.savefig(savename, bbox_inches="tight")
            logger.info(f"Saved plot '{name}' to {savename}")
            plt.close(fig)

        logger.info(f"Successfully generated plot '{name}'")
        return plot_obj

    except ImportError:
        logger.error("monet-plots is not installed or the specific plot type module could not be found.")
        raise
    except Exception as e:
        logger.error(f"Failed to generate plot '{name}': {e}")
        raise
