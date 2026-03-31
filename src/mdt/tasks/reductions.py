import logging
from typing import Optional, Union

import monet_stats
import xarray as xr

from mdt.utils import update_history

logger = logging.getLogger(__name__)


def spatial_mean(
    obj: Union[xr.DataArray, xr.Dataset],
    lon_dim: Optional[str] = None,
    lat_dim: Optional[str] = None,
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Compute the area-weighted spatial mean using monet-stats.

    This function automatically discovers spatial dimensions (if not provided)
    and delegates to `monet_stats.weighted_spatial_mean`, which implements
    the Aero Protocol (supporting both NumPy and Dask backends).

    Parameters
    ----------
    obj : xr.DataArray or xr.Dataset
        The input data to average spatially.
    lon_dim : str, optional
        The name of the longitude dimension. If None, it will be automatically discovered.
    lat_dim : str, optional
        The name of the latitude dimension. If None, it will be automatically discovered.

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
    logger.info("Computing area-weighted spatial mean via monet-stats.")

    # 1. Discover Dimensions (Aero Protocol Rule 2: Maintainability)
    if lat_dim is None:
        lat_dim = _find_lat_dim(obj)
    if lon_dim is None:
        lon_dim = _find_lon_dim(obj)

    if lat_dim is None or lon_dim is None:
        logger.warning(f"Could not discover spatial dimensions (lat={lat_dim}, lon={lon_dim}). Performing simple mean.")
        result = obj.mean()
    else:
        # 2. Delegate to monet-stats (Aero Protocol Rule 1: Backend Agnostic)
        # monet-stats handles cell_area detection and cosine weighting.
        result = monet_stats.weighted_spatial_mean(obj, lat_dim=lat_dim, lon_dim=lon_dim)

    # 3. Provenance Tracking (Aero Protocol Rule 2.3: Scientific Hygiene)
    msg = f"Computed area-weighted spatial mean over {lat_dim} and {lon_dim} via monet-stats."
    result = update_history(result, msg)

    logger.info("Successfully computed spatial mean.")
    return result


def _find_lat_dim(obj: Union[xr.DataArray, xr.Dataset]) -> Optional[str]:
    """Find the latitude dimension name using standard conventions."""
    for dim in obj.dims:
        if "lat" in str(dim).lower():
            return str(dim)
    # Check attributes of coordinates (CF conventions)
    for coord in obj.coords:
        attrs = obj[coord].attrs
        if attrs.get("units") in ["degrees_north", "degree_north", "degree_N", "degrees_N", "degreeN", "degreesN"]:
            return str(coord)
        if attrs.get("axis") == "Y":
            return str(coord)
    return None


def _find_lon_dim(obj: Union[xr.DataArray, xr.Dataset]) -> Optional[str]:
    """Find the longitude dimension name using standard conventions."""
    for dim in obj.dims:
        if "lon" in str(dim).lower():
            return str(dim)
    # Check attributes of coordinates (CF conventions)
    for coord in obj.coords:
        attrs = obj[coord].attrs
        if attrs.get("units") in ["degrees_east", "degree_east", "degree_E", "degrees_E", "degreeE", "degreesE"]:
            return str(coord)
        if attrs.get("axis") == "X":
            return str(coord)
    return None
