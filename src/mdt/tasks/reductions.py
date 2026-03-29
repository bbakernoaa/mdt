import logging
from typing import Union

import monet_stats
import xarray as xr

from mdt.utils import update_history

logger = logging.getLogger(__name__)


def spatial_mean(
    obj: Union[xr.DataArray, xr.Dataset],
    lon_dim: str = "lon",
    lat_dim: str = "lat",
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Compute the area-weighted spatial mean using monet-stats.

    This function delegates to `monet_stats.weighted_spatial_mean`, which
    implements the Aero Protocol (supporting both NumPy and Dask backends).

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
    >>> import xarray as xr
    >>> da = xr.DataArray([[1, 2], [3, 4]], coords={"lat": [0, 1], "lon": [0, 1]}, dims=("lat", "lon"))
    >>> sm = spatial_mean(da)
    """
    logger.info("Computing area-weighted spatial mean via monet_stats.")

    # Delegate to monet-stats for robust implementation
    # It handles cell_area detection, cosine weighting, and Aero Protocol compliance.
    result = monet_stats.weighted_spatial_mean(obj, lat_dim=lat_dim, lon_dim=lon_dim)

    # Provenance Tracking
    msg = f"Computed area-weighted spatial mean over {lat_dim} and {lon_dim} via monet-stats."
    result = update_history(result, msg)

    logger.info("Successfully computed spatial mean.")
    return result
