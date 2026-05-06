from typing import Any, List, Optional, Tuple, Union

import pandas as pd
import xarray as xr


def update_history(obj: Any, message: str) -> Any:
    """
    Update the provenance history for Xarray or Pandas objects.

    Parameters
    ----------
    obj : Any
        The data object to update (Xarray Dataset/DataArray or Pandas DataFrame/Series).
    message : str
        The message to append to the history.

    Returns
    -------
    Any
        The updated object with the history attribute modified.
    """
    if isinstance(obj, (xr.DataArray, xr.Dataset)):
        if getattr(obj, "attrs", None) is None:
            obj.attrs = {}
        history = obj.attrs.get("history", "")
        obj.attrs["history"] = f"{history}\n{message}".strip()
    elif isinstance(obj, (pd.Series, pd.DataFrame)):
        try:
            if getattr(obj, "attrs", None) is None:
                obj.attrs = {}

            current_attrs = obj.attrs
            history = current_attrs.get("history", "")
            if isinstance(history, str):
                current_attrs["history"] = (history + f"\n{message}").strip()
            elif isinstance(history, dict):
                mdt_hist = str(history.get("mdt_history", ""))
                history["mdt_history"] = (mdt_hist + f"\n{message}").strip()
                current_attrs["history"] = history
            else:
                current_attrs["history"] = message.strip()

            obj.attrs = current_attrs
        except Exception:  # noqa: S110
            # Pandas attrs are experimental and might fail in some versions/objects
            pass
    return obj


def discover_spatial_dims(
    obj: Union[xr.DataArray, xr.Dataset, List[str], str],
    dims: Optional[List[str]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Identify latitude and longitude dimensions from an object or list of names.

    Parameters
    ----------
    obj : xr.DataArray, xr.Dataset, list of str, or str
        The object to inspect for spatial dimensions.
    dims : list of str, optional
        A subset of dimensions to search within. If None, uses all dimensions of obj.

    Returns
    -------
    tuple of (str or None, str or None)
        A tuple containing (lat_dim, lon_dim). Either can be None if not found.

    Examples
    --------
    >>> lat, lon = discover_spatial_dims(da)
    >>> lat, lon = discover_spatial_dims(["latitude", "longitude"])
    """
    search_dims: List[str]
    if isinstance(obj, (xr.DataArray, xr.Dataset)):
        search_dims = [str(d) for d in obj.dims]
    elif isinstance(obj, str):
        search_dims = [obj]
    else:
        search_dims = [str(d) for d in obj]

    if dims is not None:
        search_dims = [d for d in search_dims if d in dims]

    lat_dim = None
    lon_dim = None

    for d in search_dims:
        d_lower = d.lower()
        if "lat" in d_lower and lat_dim is None:
            lat_dim = d
        if "lon" in d_lower and lon_dim is None:
            lon_dim = d

    return lat_dim, lon_dim
