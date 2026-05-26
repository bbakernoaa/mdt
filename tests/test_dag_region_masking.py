"""Unit tests for DAG mask and regions attribute propagation.

Validates Requirements 5.1, 5.2, 5.3, 5.4, 5.5:
- mask attribute present/absent on pairing nodes
- regions attribute present/absent on plot and statistics nodes
"""

from mdt.dag import DAGBuilder


class DummyConfig:
    """Mock MDT config object for DAG builder testing."""

    def __init__(self, data_dict):
        self.data = data_dict
        self.execution = {"default_cluster": "compute"}
        self.pairing = {}
        self.combine = {}
        self.statistics = {}
        self.plots = {}


# --- Pairing mask tests (Requirements 5.1, 5.2) ---


def test_pairing_node_has_mask_when_configured():
    """When a pairing task has mask in config, the DAG node has mask attribute."""
    config = DummyConfig(
        data_dict={
            "gfs_model": {"type": "cmaq"},
            "ish_obs": {"type": "aeronet"},
        }
    )
    config.pairing = {
        "gfs_ish": {
            "source": "gfs_model",
            "target": "ish_obs",
            "method": "bilinear",
            "mask": "land",
        }
    }

    builder = DAGBuilder(config)
    dag = builder.build()

    node = dag.nodes["pair_gfs_ish"]
    assert "mask" in node
    assert node["mask"] == "land"


def test_pairing_node_no_mask_when_not_configured():
    """When a pairing task has no mask key, the DAG node does NOT have mask attribute."""
    config = DummyConfig(
        data_dict={
            "gfs_model": {"type": "cmaq"},
            "ish_obs": {"type": "aeronet"},
        }
    )
    config.pairing = {
        "gfs_ish": {
            "source": "gfs_model",
            "target": "ish_obs",
            "method": "bilinear",
        }
    }

    builder = DAGBuilder(config)
    dag = builder.build()

    node = dag.nodes["pair_gfs_ish"]
    assert "mask" not in node


# --- Plot regions tests (Requirements 5.4, 5.5) ---


def test_plot_node_has_regions_when_configured():
    """When a plot task has regions in kwargs, the DAG node has regions attribute."""
    config = DummyConfig(
        data_dict={
            "gfs_model": {"type": "cmaq"},
            "ish_obs": {"type": "aeronet"},
        }
    )
    config.pairing = {
        "gfs_ish": {
            "source": "gfs_model",
            "target": "ish_obs",
            "mask": "land",
        }
    }
    config.plots = {
        "temp_plot": {
            "input": "gfs_ish",
            "type": "timeseries",
            "kwargs": {
                "savename": "temp_{region}.png",
                "regions": ["North America", "Europe"],
            },
        }
    }

    builder = DAGBuilder(config)
    dag = builder.build()

    node = dag.nodes["plot_temp_plot"]
    assert "regions" in node
    assert node["regions"] == ["North America", "Europe"]


def test_plot_node_no_regions_when_not_configured():
    """When a plot task has no regions in kwargs, the DAG node does NOT have regions attribute."""
    config = DummyConfig(
        data_dict={
            "gfs_model": {"type": "cmaq"},
            "ish_obs": {"type": "aeronet"},
        }
    )
    config.pairing = {
        "gfs_ish": {
            "source": "gfs_model",
            "target": "ish_obs",
        }
    }
    config.plots = {
        "temp_plot": {
            "input": "gfs_ish",
            "type": "timeseries",
            "kwargs": {
                "savename": "temp.png",
            },
        }
    }

    builder = DAGBuilder(config)
    dag = builder.build()

    node = dag.nodes["plot_temp_plot"]
    assert "regions" not in node


# --- Statistics regions tests (Requirements 5.3, 5.5) ---


def test_statistics_node_has_regions_when_configured():
    """When a statistics task has regions in kwargs, the DAG node has regions attribute."""
    config = DummyConfig(
        data_dict={
            "gfs_model": {"type": "cmaq"},
            "ish_obs": {"type": "aeronet"},
        }
    )
    config.pairing = {
        "gfs_ish": {
            "source": "gfs_model",
            "target": "ish_obs",
            "mask": "land",
        }
    }
    config.statistics = {
        "met_stats": {
            "input": "gfs_ish",
            "metrics": ["rmse", "mb"],
            "kwargs": {
                "savename": "stats_{region}.md",
                "regions": ["Asia"],
            },
        }
    }

    builder = DAGBuilder(config)
    dag = builder.build()

    node = dag.nodes["stats_met_stats"]
    assert "regions" in node
    assert node["regions"] == ["Asia"]


def test_statistics_node_no_regions_when_not_configured():
    """When a statistics task has no regions in kwargs, the DAG node does NOT have regions attribute."""
    config = DummyConfig(
        data_dict={
            "gfs_model": {"type": "cmaq"},
            "ish_obs": {"type": "aeronet"},
        }
    )
    config.pairing = {
        "gfs_ish": {
            "source": "gfs_model",
            "target": "ish_obs",
        }
    }
    config.statistics = {
        "met_stats": {
            "input": "gfs_ish",
            "metrics": ["rmse", "mb"],
            "kwargs": {
                "savename": "stats.md",
            },
        }
    }

    builder = DAGBuilder(config)
    dag = builder.build()

    node = dag.nodes["stats_met_stats"]
    assert "regions" not in node


def test_regions_preserves_order():
    """The regions attribute preserves the original list order from config."""
    config = DummyConfig(
        data_dict={
            "gfs_model": {"type": "cmaq"},
            "ish_obs": {"type": "aeronet"},
        }
    )
    config.pairing = {
        "gfs_ish": {
            "source": "gfs_model",
            "target": "ish_obs",
            "mask": "land",
        }
    }
    config.plots = {
        "ordered_plot": {
            "input": "gfs_ish",
            "type": "timeseries",
            "kwargs": {
                "savename": "plot_{region}.png",
                "regions": ["Europe", "Asia", "North America"],
            },
        }
    }

    builder = DAGBuilder(config)
    dag = builder.build()

    node = dag.nodes["plot_ordered_plot"]
    assert node["regions"] == ["Europe", "Asia", "North America"]
