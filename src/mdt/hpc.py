"""NOAA RDHPCS Profile factory for Dask jobqueue integration."""

import logging

from dask_jobqueue import LSFCluster, PBSCluster, SLURMCluster

logger = logging.getLogger(__name__)


class HPCProfileFactory:
    """
    Factory for generating Dask Jobqueue cluster instances.

    Tailored to specific NOAA RDHPCS platforms (Hera, Jet, Orion, Hercules, Gaea, Ursa, WCOSS2).
    """

    @classmethod
    def create_cluster(cls, mode, **user_kwargs):
        """
        Creates and returns a dynamically configured dask-jobqueue Cluster instance.

        Parameters
        ----------
        mode : str
            The platform or cluster type (e.g., 'hera', 'ursa', 'slurm', 'wcoss2').
        **user_kwargs : dict
            Custom overrides provided by the user via configuration. These take
            precedence over built-in platform defaults.

        Returns
        -------
        dask_jobqueue.JobQueueCluster
            An instantiated dask-jobqueue cluster object (e.g., SLURMCluster,
            PBSCluster, LSFCluster) configured for the requested mode.

        Raises
        ------
        ValueError
            If an unknown execution mode or HPC profile is specified.
        """
        mode = mode.lower()

        # Dispatch to specific HPC profiles
        if mode == "hera":
            return cls._create_hera(user_kwargs)
        elif mode == "jet":
            return cls._create_jet(user_kwargs)
        elif mode == "orion":
            return cls._create_orion(user_kwargs)
        elif mode == "hercules":
            return cls._create_hercules(user_kwargs)
        elif mode == "gaea":
            return cls._create_gaea(user_kwargs)
        elif mode == "ursa":
            return cls._create_ursa(user_kwargs)
        elif mode == "wcoss2":
            return cls._create_wcoss2(user_kwargs)

        # Dispatch to generic schedulers
        elif mode == "slurm":
            return SLURMCluster(**user_kwargs)
        elif mode == "pbs":
            return PBSCluster(**user_kwargs)
        elif mode == "lsf":
            return LSFCluster(**user_kwargs)
        else:
            raise ValueError(f"Unknown execution mode or HPC profile: '{mode}'")

    @classmethod
    def _create_hera(cls, user_kwargs):
        logger.info("Initializing SLURM cluster with RDHPCS Hera defaults.")
        defaults = {
            "cores": 40,
            "memory": "120GB",
            "processes": 1,
            "walltime": "01:00:00",
            "job_extra_directives": ["--qos=batch"],
            "interface": "ib0",  # Example interface, might need adjustment
        }
        # User overrides take precedence
        defaults.update(user_kwargs)
        return SLURMCluster(**defaults)

    @classmethod
    def _create_jet(cls, user_kwargs):
        logger.info("Initializing SLURM cluster with RDHPCS Jet defaults.")
        defaults = {
            "cores": 24,  # Jet typically has variations, setting a safe default
            "memory": "60GB",
            "processes": 1,
            "walltime": "01:00:00",
        }
        defaults.update(user_kwargs)
        return SLURMCluster(**defaults)

    @classmethod
    def _create_orion(cls, user_kwargs):
        logger.info("Initializing SLURM cluster with RDHPCS Orion defaults.")
        defaults = {
            "cores": 40,
            "memory": "180GB",
            "processes": 1,
            "walltime": "01:00:00",
        }
        defaults.update(user_kwargs)
        return SLURMCluster(**defaults)

    @classmethod
    def _create_hercules(cls, user_kwargs):
        logger.info("Initializing SLURM cluster with RDHPCS Hercules defaults.")
        defaults = {
            "cores": 80,
            "memory": "250GB",
            "processes": 1,
            "walltime": "01:00:00",
        }
        defaults.update(user_kwargs)
        return SLURMCluster(**defaults)

    @classmethod
    def _create_gaea(cls, user_kwargs):
        logger.info("Initializing SLURM cluster with RDHPCS Gaea defaults.")
        defaults = {
            "cores": 36,
            "memory": "120GB",
            "processes": 1,
            "walltime": "01:00:00",
        }
        defaults.update(user_kwargs)
        return SLURMCluster(**defaults)

    @classmethod
    def _create_ursa(cls, user_kwargs):
        logger.info("Initializing SLURM cluster with RDHPCS Ursa defaults.")
        defaults = {
            "cores": 36,  # Adjust cores/memory as appropriate for Ursa hardware
            "memory": "120GB",
            "processes": 1,
            "walltime": "01:00:00",
        }
        defaults.update(user_kwargs)
        return SLURMCluster(**defaults)

    @classmethod
    def _create_wcoss2(cls, user_kwargs):
        # WCOSS2 typically uses PBS Pro
        logger.info("Initializing PBS cluster with WCOSS2 defaults.")
        defaults = {
            "cores": 128,
            "memory": "256GB",
            "processes": 1,
            "walltime": "01:00:00",
            "queue": "dev",
        }
        defaults.update(user_kwargs)
        return PBSCluster(**defaults)
