from typing import Any

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
        except Exception:
            # Pandas attrs are experimental and might fail in some versions/objects
            pass
    return obj
