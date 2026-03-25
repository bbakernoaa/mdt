# Quick Start Guide

This guide will help you set up and run your first MDT verification workflow.

## Installation

MDT requires several components from the MONET ecosystem. For the best experience, we recommend using a Conda or Mamba environment.

```bash
# Create and activate environment
mamba create -n mdt python=3.9
mamba activate mdt

# Install MONET ecosystem from GitHub
pip install git+https://github.com/noaa-oar-arl/monetio.git
pip install git+https://github.com/noaa-oar-arl/monet.git
pip install git+https://github.com/noaa-oar-arl/monet-stats.git

# Install monet-plots without dependencies to avoid conflicts
pip install git+https://github.com/noaa-oar-arl/monet-plots.git --no-deps

# Install MDT and its direct dependencies
git clone https://github.com/noaa-oar-arl/mvs.git
cd mvs
pip install -e .
```

> **Note:** MDT also requires `pyarrow` for Dask/Pandas compatibility.

## Running MDT

MDT is configured via a YAML file. Once you have a configuration file (e.g., `config.yaml`), you can run the tool using the command-line interface:

```bash
mdt run --config config.yaml
```

## A Minimal Working Example

Create a file named `simple_eval.yaml` with the following content:

```yaml
data:
  my_model:
    type: "cmaq"
    kwargs:
      fname: "path/to/cmaq_output.nc"
  my_obs:
    type: "aeronet"
    kwargs:
      fname: "path/to/aeronet_data.nc"

pairing:
  eval_pair:
    source: "my_model"
    target: "my_obs"
    method: "interpolate"

statistics:
  basic_stats:
    input: "eval_pair"
    metrics: ["rmse", "bias", "corr"]
    kwargs:
      obs_var: "AOD_500"
      mod_var: "AOD_500"

plots:
  spatial_eval:
    input: "eval_pair"
    type: "spatial"
    track: "A"
    kwargs:
      savename: "cmaq_vs_aeronet.png"
```

Then run it:

```bash
mdt run --config simple_eval.yaml
```

MDT will automatically load the data, pair the model output to the observation locations, compute the requested statistics, and save a spatial plot.
