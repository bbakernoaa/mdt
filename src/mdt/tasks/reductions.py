import logging
from typing import Optional, Union

import numpy as np
import xarray as xr

try:
    import monet_stats
except ImportError:
    monet_stats = None

from mdt.utils import update_history

logger = logging.getLogger(__name__)


def spatial_mean(
    obj: Union[xr.DataArray, xr.Dataset],
    lon_dim: Optional[str] = None,
    lat_dim: Optional[str] = None,
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Compute the area-weighted spatial mean.

    This function attempts to use `monet_stats.weighted_spatial_mean` if available,
    falling back to a native Xarray implementation with cosine weighting if needed.
    It adheres to the Aero Protocol by supporting both NumPy and Dask backends
    without forcing computation.

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
    logger.info("Computing area-weighted spatial mean.")

    # 1. Discover Dimensions
    if lat_dim is None:
        lat_dim = _find_lat_dim(obj)
    if lon_dim is None:
        lon_dim = _find_lon_dim(obj)

    if lat_dim is None or lon_dim is None:
        logger.warning(f"Could not discover spatial dimensions (lat={lat_dim}, lon={lon_dim}). Performing simple mean.")
        result = obj.mean()
    else:
        # 2. Compute Weighted Mean
        if monet_stats is not None:
            try:
                result = monet_stats.weighted_spatial_mean(obj, lat_dim=lat_dim, lon_dim=lon_dim)
            except Exception as e:
                logger.warning(f"monet_stats.weighted_spatial_mean failed: {e}. Falling back to native Xarray.")
                result = _native_weighted_mean(obj, lat_dim, lon_dim)
        else:
            result = _native_weighted_mean(obj, lat_dim, lon_dim)

    # 3. Provenance Tracking
    msg = f"Computed area-weighted spatial mean over {lat_dim} and {lon_dim}."
    result = update_history(result, msg)

    logger.info("Successfully computed spatial mean.")
    return result


def _find_lat_dim(obj: Union[xr.DataArray, xr.Dataset]) -> Optional[str]:
    """Find the latitude dimension name."""
    for dim in obj.dims:
        if "lat" in str(dim).lower():
            return str(dim)
    # Check attributes of coordinates
    for coord in obj.coords:
        attrs = obj[coord].attrs
        if attrs.get("units") in ["degrees_north", "degree_north", "degree_N", "degrees_N", "degreeN", "degreesN"]:
            return str(coord)
        if attrs.get("axis") == "Y":
            return str(coord)
    return None


def _find_lon_dim(obj: Union[xr.DataArray, xr.Dataset]) -> Optional[str]:
    """Find the longitude dimension name."""
    for dim in obj.dims:
        if "lon" in str(dim).lower():
            return str(dim)
    # Check attributes of coordinates
    for coord in obj.coords:
        attrs = obj[coord].attrs
        if attrs.get("units") in ["degrees_east", "degree_east", "degree_E", "degrees_E", "degreeE", "degreesE"]:
            return str(coord)
        if attrs.get("axis") == "X":
            return str(coord)
    return None


def _native_weighted_mean(obj: Union[xr.DataArray, xr.Dataset], lat_dim: str, lon_dim: str) -> Union[xr.DataArray, xr.Dataset]:
    """Fallback native Xarray implementation of area-weighted mean."""
    weights = np.cos(np.deg2rad(obj[lat_dim]))
    return obj.weighted(weights).mean(dim=(lat_dim, lon_dim))
