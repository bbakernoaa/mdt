import logging

# Monet pairing is usually handled by `monet.models.*` or `monet.obs.*` depending on the object,
# or through `monet.utils.spatial` regridding, using xregrid and esmpy.

logger = logging.getLogger(__name__)


def pair_data(name, method, source_data, target_data, kwargs):
    """
    Dynamically pairs two datasets using monet regridding or interpolation.

    Parameters
    ----------
    name : str
        The identifier for this pairing task.
    method : str
        The regridding method to use (e.g., 'interpolate', 'regrid', 'point_to_grid').
    source_data : xarray.Dataset or pandas.DataFrame
        The source data object (typically a model).
    target_data : xarray.Dataset or pandas.DataFrame
        The target data object or grid (typically observations or a reference grid).
    kwargs : dict
        Additional keyword arguments to pass to the underlying pairing function.

    Returns
    -------
    xarray.Dataset or pandas.DataFrame
        The resulting paired dataset.

    Raises
    ------
    ValueError
        If an unknown pairing method is specified.
    """
    logger.info(f"Pairing data '{name}' using method '{method}'")

    import monet

    try:
        if method == "interpolate":
            # E.g., model to point observations
            # Often monet handles this via the model object itself, but let's assume
            # a generic regridding approach using xregrid or monet.utils.
            # This is a generic placeholder for the actual monet logic.
            # In monet, the target is often an observation dataframe, and the source
            # is a model xarray dataset.
            paired_data = monet.utils.spatial.interpolate(source_data, target_data, **kwargs)

        elif method == "regrid":
            # E.g., model to model, using xregrid (esmpy backend)
            from monet.utils.spatial import regrid

            paired_data = regrid(source_data, target_data, **kwargs)

        else:
            raise ValueError(f"Unknown pairing method '{method}'.")

        logger.info(f"Successfully paired data '{name}'")
        return paired_data

    except Exception as e:
        logger.error(f"Failed to pair data '{name}': {e}")
        raise
