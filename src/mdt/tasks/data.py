import logging
from typing import Any, Dict, Union, cast

import pandas as pd
import xarray as xr

from mdt.utils import update_history

logger = logging.getLogger(__name__)


def load_data(
    name: str,
    dataset_type: str,
    kwargs: Dict[str, Any],
) -> Union[xr.Dataset, pd.DataFrame]:
    """
    Unified entry point for loading data via monetio.load.

    Supports standard loading and VirtualiZarr-based ingestion (kerchunk/icechunk).
    If VirtualiZarr parameters cause a TypeError (e.g., old reader version),
    it automatically falls back to standard loading.

    Parameters
    ----------
    name : str
        A user-defined name for this dataset.
    dataset_type : str
        The dataset type/reader name recognized by monetio (e.g., 'cmaq', 'aeronet').
    kwargs : dict
        Keyword arguments passed to monetio.load.

    Returns
    -------
    xarray.Dataset or pandas.DataFrame
        The loaded data object.

    Notes
    -----
    Aero Protocol: Ensures that data is loaded lazily where supported.
    """
    logger.info("Loading dataset '%s' of type '%s'", name, dataset_type)

    import monetio

    def _prefer_xarray_result(
        loaded_obj: Union[xr.Dataset, pd.DataFrame],
        load_kwargs: Dict[str, Any],
    ) -> Union[xr.Dataset, pd.DataFrame]:
        """Prefer xarray output for point-reader pathways when possible."""
        if isinstance(loaded_obj, xr.Dataset):
            return loaded_obj

        if load_kwargs.get("as_xarray") is False:
            return loaded_obj

        retry_kwargs = dict(load_kwargs)
        retry_kwargs["as_xarray"] = True
        logger.info(
            "Dataset '%s' returned DataFrame; retrying monetio.load with as_xarray=True.",
            name,
        )
        retry_obj = monetio.load(dataset_type, **retry_kwargs)
        if isinstance(retry_obj, xr.Dataset):
            return retry_obj

        logger.warning(
            "Dataset '%s' still returned non-xarray output after as_xarray=True retry.",
            name,
        )
        return loaded_obj

    kwargs = dict(kwargs)

    # Compatibility shim: ICAP config often uses singular `date`, while
    # monetio reader interface expects `dates`.
    if dataset_type == "icap_mme" and "dates" not in kwargs and "date" in kwargs:
        kwargs["dates"] = kwargs.pop("date")

    # Support date-range syntax in YAML without requiring users to enumerate every day.
    # If both start_date and end_date are provided, expand to a daily inclusive list.
    if "start_date" in kwargs and "end_date" in kwargs and "dates" not in kwargs:
        date_freq = kwargs.pop("date_freq", "D")
        start_date = kwargs.pop("start_date")
        end_date = kwargs.pop("end_date")
        date_index = pd.date_range(start=start_date, end=end_date, freq=date_freq)
        if date_index.empty:
            raise ValueError(f"Invalid date range for dataset '{name}': {start_date} -> {end_date}")
        kwargs["dates"] = [d.strftime("%Y-%m-%d") for d in date_index]
        logger.info(
            "Expanded date range for '%s' from %s to %s (freq=%s, n=%d).",
            name,
            start_date,
            end_date,
            date_freq,
            len(kwargs["dates"]),
        )

    def _strip_virtualizarr_kwargs(input_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        vz_keys = [
            "use_virtualizarr",
            "virtualizarr_backend",
            "store_path",
            "icechunk_url",
            "icechunk_repo",
            "use_icechunk",
            "max_scan_attempts",
            "network_timeout",
            "max_concurrent_requests",
            "existing_zarr",
        ]
        return {k: v for k, v in input_kwargs.items() if k not in vz_keys}

    def _normalize_virtualizarr_kwargs(input_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Translate MDT kwargs to MonetIO's preferred VirtualiZarr interface."""
        load_kwargs = dict(input_kwargs)
        backend = load_kwargs.get("virtualizarr_backend")

        # MonetIO deprecates explicit backend arg for virtual paths.
        load_kwargs.pop("virtualizarr_backend", None)
        if backend == "icechunk":
            load_kwargs["use_icechunk"] = True

        # MonetIO prefers icechunk_url; keep legacy key support in config.
        if "icechunk_url" not in load_kwargs and "icechunk_repo" in load_kwargs:
            load_kwargs["icechunk_url"] = load_kwargs["icechunk_repo"]
        load_kwargs.pop("icechunk_repo", None)
        return load_kwargs

    # 1. Attempt Direct Zarr/Icechunk Loading if existing store is pointed to
    existing_zarr = kwargs.get("existing_zarr", False)
    if existing_zarr:
        backend = "icechunk" if kwargs.get("use_icechunk", False) else kwargs.get("virtualizarr_backend", "zarr")
        store_path = kwargs.get("store_path", f"./zarr_stores/{name}/")
        icechunk_url = kwargs.get("icechunk_url") or kwargs.get("icechunk_repo", "")
        zarr_kwargs = kwargs.get("zarr_kwargs", {})

        logger.info(
            "Loading existing %s store for '%s' from %s",
            backend,
            name,
            icechunk_url if backend == "icechunk" else store_path,
        )

        if backend == "icechunk":
            try:
                import icechunk

                repo = icechunk.Repository.open(icechunk_url)
                session = repo.readonly_session()
                ds = xr.open_zarr(session.store, consolidated=False, **zarr_kwargs)
                msg = f"Loaded existing Icechunk store '{name}' from {icechunk_url}."
                ds = update_history(ds, msg)
                return cast(Union[xr.Dataset, pd.DataFrame], ds)
            except Exception as e:
                logger.error("Failed to load existing Icechunk store '%s': %s", name, e)
                raise
        else:
            # Standard Zarr
            try:
                ds = xr.open_zarr(store_path, **zarr_kwargs)
                msg = f"Loaded existing Zarr store '{name}' from {store_path}."
                ds = update_history(ds, msg)
                return cast(Union[xr.Dataset, pd.DataFrame], ds)
            except Exception as e:
                logger.error("Failed to load existing Zarr store '%s': %s", name, e)
                raise

    # 2. Attempt VirtualiZarr Loading if enabled
    use_virtualizarr = kwargs.get("use_virtualizarr", False)
    if use_virtualizarr:
        backend = kwargs.get("virtualizarr_backend", "kerchunk_json")
        store_path = kwargs.get("store_path", f"./zarr_stores/{name}/")
        icechunk_url = kwargs.get("icechunk_url") or kwargs.get("icechunk_repo", "")

        logger.info(
            "Attempting VirtualiZarr load for '%s' (backend=%s, store=%s, icechunk_url=%s)",
            name,
            backend,
            store_path,
            icechunk_url,
        )

        try:
            # Call monetio.load with all VirtualiZarr parameters
            load_kwargs = _normalize_virtualizarr_kwargs(kwargs)
            ds = monetio.load(dataset_type, **load_kwargs)
            ds = _prefer_xarray_result(cast(Union[xr.Dataset, pd.DataFrame], ds), kwargs)
            logger.info("Successfully loaded '%s' via VirtualiZarr pathway.", name)

            # Add MDT provenance tracking
            msg = f"Loaded dataset '{name}' via VirtualiZarr (backend={backend})."
            ds = update_history(ds, msg)
            return cast(Union[xr.Dataset, pd.DataFrame], ds)

        except TypeError as e:
            # Handle cases where reader doesn't support VirtualiZarr kwargs yet
            logger.warning(
                "VirtualiZarr load failed for '%s' due to unexpected arguments: %s. Retrying with standard monetio.load.",
                name,
                e,
            )
            fallback_kwargs = _strip_virtualizarr_kwargs(kwargs)
            logger.info("Retrying standard monetio.load with kwargs: %s", fallback_kwargs)
            ds = monetio.load(dataset_type, **fallback_kwargs)
            ds = _prefer_xarray_result(cast(Union[xr.Dataset, pd.DataFrame], ds), fallback_kwargs)
            msg = f"Loaded dataset '{name}' (VirtualiZarr fallback used)."
            ds = update_history(ds, msg)
            return cast(Union[xr.Dataset, pd.DataFrame], ds)

    # 3. Standard Loading Pathway
    try:
        ds = monetio.load(dataset_type, **kwargs)
        ds = _prefer_xarray_result(cast(Union[xr.Dataset, pd.DataFrame], ds), kwargs)

        # Provenance Tracking
        msg = f"Loaded dataset '{name}' using type '{dataset_type}' with params {kwargs}."
        ds = update_history(ds, msg)

        logger.info("Successfully loaded dataset '%s'", name)
        return cast(Union[xr.Dataset, pd.DataFrame], ds)

    except Exception as e:
        logger.error("Failed to load dataset '%s': %s", name, e)
        raise
