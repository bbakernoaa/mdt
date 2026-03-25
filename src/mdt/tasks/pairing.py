import logging
from typing import Union

import pandas as pd
import xarray as xr

# Monet pairing is usually handled by `monet.models.*` or `monet.obs.*` depending on the object,
# or through `monet.util.interp_util` regridding, using xregrid and esmpy.

logger = logging.getLogger(__name__)


def pair_data(
    name: str,
    method: str,
    source_data: Union[xr.Dataset, xr.DataArray, pd.DataFrame],
    target_data: Union[xr.Dataset, xr.DataArray, pd.DataFrame],
    kwargs: dict,
) -> Union[xr.Dataset, xr.DataArray, pd.DataFrame]:
    """
    Dynamically pairs two datasets using monet regridding or interpolation.

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
    >>> paired = pair_data("my_pairing", "interpolate", model_ds, obs_df, {"interp_kw": "val"})
    """
    logger.info(f"Pairing data '{name}' using method '{method}'")

    import monet.util.interp_util as interp_util
    import xregrid

    try:
        if method == "interpolate":
            # E.g., model to point observations
            # Often monet handles this via the model object itself, but let's assume
            # a generic regridding approach using xregrid or monet.utils.
            # This is a generic placeholder for the actual monet logic.
            # In monet, the target is often an observation dataframe, and the source
            # is a model xarray dataset.
            # Using points_to_dataset as a proxy for interpolation in this refactor
            paired_data = interp_util.points_to_dataset(source_data, target_data, **kwargs)

        elif method == "regrid":
            # E.g., model to model, using xregrid (esmpy backend)
            regridder = xregrid.Regridder(source_data, target_data, **kwargs)
            paired_data = regridder.regrid(source_data)

        else:
            raise ValueError(f"Unknown pairing method '{method}'.")

        # Provenance Tracking
        if hasattr(paired_data, "attrs"):
            history = paired_data.attrs.get("history", "")
            new_history = f"Paired using method '{method}' with params {kwargs}."
            paired_data.attrs["history"] = f"{history}\n{new_history}".strip()

        logger.info(f"Successfully paired data '{name}'")
        return paired_data

    except Exception as e:
        logger.error(f"Failed to pair data '{name}': {e}")
        raise
