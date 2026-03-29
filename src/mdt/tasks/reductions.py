import logging
from typing import Union

import numpy as np
import xarray as xr

from mdt.utils import update_history

logger = logging.getLogger(__name__)


def spatial_mean(
    obj: Union[xr.DataArray, xr.Dataset],
    lon_dim: str = "lon",
    lat_dim: str = "lat",
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Compute the area-weighted spatial mean using cosine of latitude.

    This function is backend-agnostic and adheres to the Aero Protocol:
    it works with both NumPy-backed and Dask-backed Xarray objects without
    forcing computation.

    Parameters
    ----------
    obj : xr.DataArray or xr.Dataset
        The input data to average spatially.
    lon_dim : str, optional
        The name of the longitude dimension. Default is 'lon'.
    lat_dim : str, optional
        The name of the latitude dimension. Default is 'lat'.

    Returns
    -------
    xr.DataArray or xr.Dataset
        The area-weighted spatial mean of the input object.

    Examples
    --------
    >>> ds = xr.tutorial.open_dataset("air_temperature")
    >>> sm = spatial_mean(ds)
    """
    logger.info(f"Computing area-weighted spatial mean over dimensions ({lat_dim}, {lon_dim})")

    # 1. Calculate weights based on cosine of latitude
    # Xarray handles the broadcasting of weights automatically.
    weights = np.cos(np.deg2rad(obj[lat_dim]))
    weights.name = "weights"

    # 2. Apply weighting and compute mean
    # This remains lazy if 'obj' is Dask-backed.
    result = obj.weighted(weights).mean(dim=(lat_dim, lon_dim))

    # 3. Provenance Tracking
    msg = f"Computed area-weighted spatial mean over {lat_dim} and {lon_dim}."
    result = update_history(result, msg)

    logger.info("Successfully computed spatial mean.")
    return result
