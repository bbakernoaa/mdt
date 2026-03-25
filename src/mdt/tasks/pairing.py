import logging
from typing import Union

import pandas as pd
import xarray as xr

# Monet pairing is usually handled by `monet.models.*` or `monet.obs.*` depending on the object,
# or through `monet.util.interp_util` regridding.

logger = logging.getLogger(__name__)


def pair_data(
    name: str,
    method: str,
    source_data: Union[xr.Dataset, xr.DataArray, pd.DataFrame],
    target_data: Union[xr.Dataset, xr.DataArray, pd.DataFrame],
    kwargs: dict,
) -> Union[xr.Dataset, xr.DataArray, pd.DataFrame]:
    """
    Dynamically pair two datasets using monet regridding or interpolation.

    Parameters
    ----------
    name : str
        The identifier for this pairing task.
    method : str
        The regridding method to use (e.g., 'interpolate', 'regrid', 'point_to_grid').
    source_data : xarray.Dataset or xarray.DataArray or pandas.DataFrame
        The source data object (typically a model).
    target_data : xarray.Dataset or xarray.DataArray or pandas.DataFrame
        The target data object or grid (typically observations or a reference grid).
    kwargs : dict
        Additional keyword arguments to pass to the underlying pairing function.

    Returns
    -------
    xarray.Dataset or xarray.DataArray or pandas.DataFrame
        The resulting paired dataset.

    Raises
    ------
    ValueError
        If an unknown pairing method is specified.

    Examples
    --------
    >>> paired = pair_data(
    ...     "my_pairing", "interpolate", model_ds, obs_df, {"interp_kw": "val"}
    ... )
    """
    logger.info("Pairing data '%s' using method '%s'", name, method)

    import monet.util.interp_util as interp_util

    try:
        if method == "interpolate":
            # Check for a standard interpolation function in monet
            func = getattr(interp_util, "points_to_dataset", None)
            if func is None:
                # Fallback to nearest neighbor if generic not found
                func = getattr(interp_util, "nearest_point_swathdefinition", None)

            if func:
                paired_data = func(source_data, target_data, **kwargs)
            else:
                # If everything fails, this is likely a custom regridding needed
                raise AttributeError("Could not find suitable interpolation function in monet.")

        elif method == "regrid":
            import xregrid

            regridder = xregrid.Regridder(source_data, target_data, **kwargs)
            paired_data = regridder.regrid(source_data)

        else:
            raise ValueError(f"Unknown pairing method '{method}'.")

        # Provenance Tracking
        if hasattr(paired_data, "attrs"):
            history = paired_data.attrs.get("history", "")
            new_history = f"Paired using method '{method}' with params {kwargs}."
            paired_data.attrs["history"] = f"{history}\n{new_history}".strip()

        logger.info("Successfully paired data '%s'", name)
        return paired_data

    except ImportError as e:
        logger.error("Required package for pairing not found: %s", e)
        raise
    except Exception as e:
        logger.error("Failed to pair data '%s': %s", name, e)
        raise
