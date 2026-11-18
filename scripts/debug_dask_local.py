"""Minimal repro: DaskEngine on a single local cluster with verbose logs."""

import logging
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

import xarray as xr

import mdt.dask_engine as de
from mdt.config import ConfigParser
from mdt.dag import DAGBuilder

tmp = Path(tempfile.mkdtemp())
cfg_file = tmp / "wf.yaml"
cfg_file.write_text(
    """
data:
  obs:
    type: aeronet
    cluster: compute
    kwargs: {files: dummy}
statistics:
  s:
    input: obs
    metrics: [mb]
    kwargs: {obs_var: o, mod_var: m}
execution:
  orchestrator: dask
  clusters:
    compute: {mode: local, workers: 1}
    service: {mode: local, workers: 1}
"""
)
config = ConfigParser(str(cfg_file))
dag = DAGBuilder(config).build()

obs = xr.Dataset({"o": ("x", [1.0, 2.0]), "m": ("x", [1.5, 2.5])})
de.load_data = lambda name, dataset_type, kwargs: obs

engine = de.DaskEngine(dag, config)
print("EXECUTING", flush=True)
results = engine.execute()
print("RESULTS:", results, flush=True)