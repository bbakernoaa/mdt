"""Reader discovery works with both supported MonetIO registry interfaces."""

import sys
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mdt.tasks.data import (
    _expand_local_grib_files,
    _grib_forecast_time_preprocess,
    _is_unsupported_keyword_type_error,
    _promote_time_coordinate,
    _reader_accepts_as_xarray,
    _reader_supports_virtualizarr,
    _resolve_reader_class,
    _time_delta,
    _unsupported_option_name,
)


@pytest.mark.parametrize(("name", "as_xarray", "virtualizarr"), [("cmaq", False, True), ("aeronet", True, False)])
def test_installed_reader_capabilities(name, as_xarray, virtualizarr):
    """Pinned MonetIO reader classes expose the correct point/gridded contract."""
    assert isinstance(_resolve_reader_class(name), type)
    assert _reader_accepts_as_xarray(name) is as_xarray
    assert _reader_supports_virtualizarr(name) is virtualizarr


def test_modern_reader_alias_resolution(monkeypatch, mocker):
    """Modern lazy imports resolve aliases through the manifest's canonical name."""
    from monetio.readers.base import READER_REGISTRY, GriddedReader

    class Reader(GriddedReader):
        pass

    lazy = types.ModuleType("monetio._lazy")
    lazy.import_reader = mocker.Mock()
    manifest = types.ModuleType("monetio._manifest")
    manifest.resolve = lambda name: SimpleNamespace(name="canonical_test")
    monkeypatch.setitem(sys.modules, "monetio._lazy", lazy)
    monkeypatch.setitem(sys.modules, "monetio._manifest", manifest)
    monkeypatch.setitem(READER_REGISTRY, "canonical_test", Reader)
    assert _resolve_reader_class("alias_test") is Reader
    lazy.import_reader.assert_called_once_with("alias_test")


def test_local_grib_expansion_is_sorted_and_deduplicated(tmp_path):
    """Existing files reach mocked or real readers in deterministic order."""
    paths = [tmp_path / "b.grib2", tmp_path / "a.grib2"]
    for path in paths:
        path.touch()
    result = _expand_local_grib_files({"files": [str(tmp_path / "*.grib2"), str(paths[0])]}, "gefs", "model")
    assert result["files"] == sorted(str(path) for path in paths)


def test_missing_grib_input_fails_before_reader(tmp_path):
    """Unit reader doubles must not hide a missing local input path."""
    with pytest.raises(FileNotFoundError, match="No local GRIB2 files"):
        _expand_local_grib_files({"files": str(tmp_path / "*.grib2")}, "gefs", "model")


def test_promote_forecast_time_derives_valid_time_from_reference_and_lead():
    """Derive valid timestamps when a reader supplies only cycle and lead."""
    data = xr.Dataset(
        {"value": (("cycle", "step"), [[1, 2]])},
        coords={
            "init": ("cycle", pd.to_datetime(["2025-08-01"]), {"standard_name": "forecast_reference_time"}),
            "step": ("step", [0, 3], {"standard_name": "forecast_period", "units": "hours"}),
        },
    )
    normalized = _promote_time_coordinate(data, "derived")
    assert normalized.sizes["time"] == 2
    assert pd.to_datetime(normalized.time.values).tolist() == [pd.Timestamp("2025-08-01"), pd.Timestamp("2025-08-01T03:00:00")]


@pytest.mark.parametrize("unit", ["h", "hours", "hr", "hrs", "m", "minutes", "s", "seconds"])
def test_time_delta_converts_numeric_unit_aliases(unit):
    """Accept common CF time unit spellings."""
    delta = _time_delta(xr.DataArray([1], attrs={"units": unit}))
    expected_unit = "h" if unit in {"h", "hours", "hr", "hrs"} else "m" if unit in {"m", "minutes"} else "s"
    assert delta.values[0] == np.timedelta64(1, expected_unit)


def test_promote_time_rejects_unmapped_time_axis_at_load_boundary(mocker):
    """Require user supplied time axis policy to be a mapping."""
    from mdt.tasks.data import load_data

    mocker.patch("mdt.tasks.data._reader_accepts_as_xarray", return_value=False)
    with pytest.raises(ValueError, match="time_axis must be a mapping"):
        load_data("bad", "gefs", {"time_axis": ["time"]})


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (TypeError("unexpected keyword argument 'product'"), "product"),
        (TypeError("backend does not support option 'source'"), "source"),
        (RuntimeError("bad"), None),
    ],
)
def test_rejected_option_message_is_parsed(error, expected):
    """Extract unsupported options from reader and backend exceptions."""
    assert _unsupported_option_name(error) == expected


def test_grib_expansion_recurses_into_directory_and_filters_extensions(tmp_path):
    """Find nested GRIB files while ignoring unrelated files."""
    directory = tmp_path / "input"
    nested = directory / "nested"
    nested.mkdir(parents=True)
    (directory / "a.GRIB").touch()
    (nested / "b.grb2").touch()
    (nested / "ignore.txt").touch()
    result = _expand_local_grib_files({"files": str(directory)}, "gfs", "model")
    assert result["files"] == sorted(str(path.resolve()) for path in (directory / "a.GRIB", nested / "b.grb2"))


def test_grib_expansion_leaves_remote_and_non_grib_inputs_unchanged():
    """Leave remote inputs and non-GRIB readers alone."""
    remote = {"files": "s3://bucket/file.grib2"}
    assert _expand_local_grib_files(remote, "gefs", "model") is remote
    local = {"files": "relative/file.nc"}
    assert _expand_local_grib_files(local, "aeronet", "obs") is local


@pytest.mark.parametrize(
    ("message", "expected"),
    [("unexpected keyword argument 'x'", True), ("takes no keyword arguments", True), ("broken", False)],
)
def test_keyword_type_error_detection(message, expected):
    """Recognize signature mismatch errors for VirtualiZarr fallback."""
    assert _is_unsupported_keyword_type_error(TypeError(message)) is expected


def test_promote_forecast_grid_stacks_valid_times():
    """Flatten cycle and lead dimensions onto the canonical time axis."""
    cycles = pd.to_datetime(["2025-08-01", "2025-08-02"])
    leads = [0, 6]
    valid = np.array([[cycles[0], cycles[0] + pd.Timedelta(hours=6)], [cycles[1], cycles[1] + pd.Timedelta(hours=6)]])
    data = xr.Dataset(
        {
            "value": (("cycle", "step"), np.arange(4).reshape(2, 2)),
            "valid": (("cycle", "step"), valid),
            "init": ("cycle", cycles),
            "lead": ("step", leads),
        }
    )
    data.valid.attrs["standard_name"] = "time"
    data.init.attrs["standard_name"] = "forecast_reference_time"
    data.lead.attrs.update(standard_name="forecast_period", units="hours")

    normalized = _promote_time_coordinate(data, "forecast")

    assert normalized.sizes["time"] == 4
    assert normalized["time"].values.tolist() == valid.reshape(-1).tolist()
    assert {"forecast_reference_time", "lead_time", "valid_time"}.issubset(normalized.coords)


def test_time_delta_preserves_timedelta_and_parses_strings():
    """Pass through timedeltas and parse human readable offsets."""
    timedeltas = xr.DataArray([np.timedelta64(3, "h")])
    strings = xr.DataArray(["3 hours"])
    xr.testing.assert_equal(_time_delta(timedeltas), timedeltas)
    assert _time_delta(strings).values[0] == np.timedelta64(3, "h")


def test_grib_expansion_accepts_a_literal_local_file(tmp_path):
    """Resolve a single existing local GRIB path."""
    path = tmp_path / "input.grib"
    path.touch()
    assert _expand_local_grib_files({"files": str(path)}, "ncep_grib", "model")["files"] == [str(path.resolve())]


def test_grib_expansion_falls_back_to_literal_path_with_glob_characters(tmp_path):
    """Retain existing file paths whose names are interpreted as glob syntax."""
    path = tmp_path / "field[1].grib"
    path.touch()
    result = _expand_local_grib_files({"files": str(path)}, "gefs", "model")
    assert result["files"] == [str(path.resolve())]


@pytest.mark.parametrize("failure", [RuntimeError("reader unavailable"), None, object()])
def test_reader_as_xarray_capability_falls_back_to_compatible_default(monkeypatch, failure):
    """Assume compatibility when a reader cannot be resolved or inspected."""
    import mdt.tasks.data as data_module

    if isinstance(failure, BaseException):

        def resolve(_dataset_type):
            raise failure
    else:

        class Reader:
            open_dataset = failure

        def resolve(_dataset_type):
            return Reader if failure is not None else None

    monkeypatch.setattr(data_module, "_resolve_reader_class", resolve)
    assert data_module._reader_accepts_as_xarray("unknown") is True


def test_virtualizarr_capability_assumes_support_when_reader_is_unresolved_or_not_a_class(monkeypatch):
    """Keep the historical VirtualiZarr path for unavailable reader metadata."""
    import mdt.tasks.data as data_module

    monkeypatch.setattr(data_module, "_resolve_reader_class", lambda _name: object())
    assert data_module._reader_supports_virtualizarr("unknown") is True

    def fail(_name):
        raise RuntimeError("no reader")

    monkeypatch.setattr(data_module, "_resolve_reader_class", fail)
    assert data_module._reader_supports_virtualizarr("unknown") is True


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (SimpleNamespace(option="icechunk_url"), True),
        (RuntimeError("VirtualiZarr unsupported"), True),
        (RuntimeError("other error"), False),
    ],
)
def test_virtualizarr_rejection_detection(error, expected):
    """Recognize explicit and textual VirtualiZarr option errors."""
    from mdt.tasks.data import _is_virtualizarr_rejection

    assert _is_virtualizarr_rejection(error) is expected


def test_forecast_time_helpers_cover_passthrough_and_derived_axes(caplog):
    """Leave unidentified dimensions alone and derive reference times as needed."""
    unchanged = xr.Dataset({"value": ("sample", [1, 2])})
    assert _promote_time_coordinate(unchanged, "plain") is unchanged

    no_named_variable = xr.Dataset({"value": ("sample", [1, 2])})
    assert _promote_time_coordinate(no_named_variable, "missing", {"valid_time": "sample"}) is no_named_variable

    numeric = xr.Dataset({"valid": ("sample", [0, 3])})
    with caplog.at_level("WARNING"):
        normalized = _promote_time_coordinate(numeric, "numeric", {"valid_time": "valid"})
    assert "time" not in normalized.coords
    assert "numeric time values" in caplog.text

    dated = xr.Dataset(
        {"value": ("sample", [1, 2])},
        coords={"valid": ("sample", pd.to_datetime(["2025-08-01", "2025-08-01T03:00:00"], format="mixed")), "lead": ("sample", [0, 3])},
    )
    normalized = _promote_time_coordinate(dated, "lead-only", {"valid_time": "valid", "lead_time": "lead"})
    assert normalized.forecast_reference_time.values.tolist() == [np.datetime64("2025-08-01T00:00:00"), np.datetime64("2025-08-01T00:00:00")]


def test_forecast_time_uses_valid_time_alias_and_frees_time_name():
    """Use valid_time when present and replace a nonstandard multidimensional time coordinate."""
    cycles = pd.to_datetime(["2025-08-01"])
    valid_values = np.array([[cycles[0], cycles[0] + pd.Timedelta(hours=3)]])
    alias_data = xr.Dataset(
        {"value": (("cycle", "step"), [[1, 2]]), "valid_time": (("cycle", "step"), valid_values)},
        coords={
            "init": ("cycle", cycles, {"standard_name": "forecast_reference_time"}),
            "lead": ("step", [0, 3], {"standard_name": "forecast_period", "units": "hours"}),
        },
    )
    alias_result = _promote_time_coordinate(alias_data, "alias")
    assert alias_result.sizes["time"] == 2
    np.testing.assert_array_equal(alias_result.valid_time.values, valid_values.reshape(-1))

    time_coord_data = xr.Dataset(
        {"value": (("cycle", "step"), [[1, 2]])},
        coords={
            "time": (("cycle", "step"), valid_values, {"standard_name": "time"}),
            "init": ("cycle", cycles, {"standard_name": "forecast_reference_time"}),
            "lead": ("step", [0, 3], {"standard_name": "forecast_period", "units": "hours"}),
        },
    )
    time_result = _promote_time_coordinate(time_coord_data, "time-alias")
    assert time_result.sizes["time"] == 2
    np.testing.assert_array_equal(time_result.time.values, valid_values.reshape(-1))


def test_explicit_standard_time_axis_is_left_unchanged():
    """Do not normalize a conventional time coordinate into forecast form."""
    data = xr.Dataset({"value": ("time", [1, 2])}, coords={"time": pd.date_range("2025-08-01", periods=2)})
    assert _promote_time_coordinate(data, "ordinary", {"valid_time": "time"}) is data


def test_numeric_valid_time_uses_forecast_reference_and_lead():
    """Treat numeric valid values as offsets when cycle and lead metadata exist."""
    data = xr.Dataset(
        {
            "value": (("cycle", "step"), [[1, 2]]),
            "valid": (("cycle", "step"), [[0, 3]], {"standard_name": "time"}),
        },
        coords={
            "init": ("cycle", pd.to_datetime(["2025-08-01"]), {"standard_name": "forecast_reference_time"}),
            "lead": ("step", [0, 3], {"standard_name": "forecast_period", "units": "hours"}),
        },
    )
    normalized = _promote_time_coordinate(data, "numeric-pair")
    assert pd.to_datetime(normalized.time.values).tolist() == [pd.Timestamp("2025-08-01"), pd.Timestamp("2025-08-01T03:00:00")]


def test_load_data_rejects_empty_xarray_result(mocker):
    """Raise a useful error when a reader returns an empty Dataset."""
    from mdt.tasks.data import load_data

    mocker.patch("mdt.tasks.data._reader_accepts_as_xarray", return_value=False)
    mocker.patch("monetio.load", return_value=xr.Dataset())
    with pytest.raises(ValueError, match="Loaded dataset 'empty' is empty"):
        load_data("empty", "cmaq", {})


def _scalar_forecast_grib_dataset(reference: str, lead_hours: int) -> xr.Dataset:
    """Mimic one grib2io GRIB file: scalar CF forecast coordinates, no time dim."""
    reference_time = np.datetime64(reference)
    lead = np.timedelta64(lead_hours, "h")
    return xr.Dataset(
        {"totAOD550": (("y", "x"), [[0.1]])},
        coords={
            "refDate": ((), reference_time, {"standard_name": "forecast_reference_time"}),
            "leadTime": ((), lead, {"standard_name": "forecast_period"}),
            "validDate": ((), reference_time + lead, {"standard_name": "time"}),
        },
    )


def test_grib_preprocess_promotes_scalar_forecast_coords_to_time_index():
    """Per-file promotion gives each GRIB sample a canonical indexed time axis."""
    ds = _scalar_forecast_grib_dataset("2025-08-01", 3)
    result = _grib_forecast_time_preprocess(ds)

    assert result.sizes["time"] == 1
    assert "time" in result.indexes
    assert result["time"].dims == ("time",)
    assert pd.Timestamp(result["time"].values[0]) == pd.Timestamp("2025-08-01T03:00:00")
    assert result["valid_time"].dims == ("time",)
    assert result["valid_time"].attrs["standard_name"] == "time"
    assert result["forecast_reference_time"].dims == ("time",)
    assert result["lead_time"].dims == ("time",)


def test_grib_preprocess_passthrough_for_indexed_or_unrelated_datasets():
    """Datasets with a time dim or without CF forecast coords are untouched."""
    indexed = xr.Dataset({"value": ("time", [1.0])}, coords={"time": pd.date_range("2025-08-01", periods=1)})
    assert _grib_forecast_time_preprocess(indexed) is indexed

    plain = xr.Dataset({"value": ("sample", [1.0, 2.0])})
    assert _grib_forecast_time_preprocess(plain) is plain


def test_multi_file_grib_concat_keeps_per_sample_valid_times():
    """Concatenating preprocessed GRIB files retains every sample's valid time.

    Regression for ``ValueError: Dimension time already exists``: without the
    preprocess, coords="minimal" collapses the scalar validDate of every file
    to the first file's value and leaves ``time`` without an index.
    """
    files = [
        _grib_forecast_time_preprocess(_scalar_forecast_grib_dataset("2025-08-01", 0)),
        _grib_forecast_time_preprocess(_scalar_forecast_grib_dataset("2025-08-01", 3)),
        _grib_forecast_time_preprocess(_scalar_forecast_grib_dataset("2025-08-02", 0)),
    ]
    with xr.set_options(use_new_combine_kwarg_defaults=True):
        combined = xr.concat(files, dim="time", coords="minimal", compat="override")

    assert "time" in combined.indexes
    expected = pd.DatetimeIndex(["2025-08-01", "2025-08-01 03:00", "2025-08-02"]).values
    np.testing.assert_array_equal(combined["time"].values, expected)

    normalized = _promote_time_coordinate(combined, "multi-file")
    np.testing.assert_array_equal(normalized["time"].values, expected)
    np.testing.assert_array_equal(normalized["valid_time"].values, expected)
    np.testing.assert_array_equal(
        normalized["valid_time"].values,
        normalized["forecast_reference_time"].values + normalized["lead_time"].values,
    )
    assert "time" in normalized.indexes


def test_local_grib_expansion_injects_forecast_preprocess(tmp_path):
    """Multi-file local GRIB loads get the preprocess; single files do not."""
    paths = [tmp_path / "b.grib2", tmp_path / "a.grib2"]
    for path in paths:
        path.touch()
    result = _expand_local_grib_files({"files": str(tmp_path / "*.grib2")}, "gefs", "model")
    assert result["preprocess"] is _grib_forecast_time_preprocess

    single = _expand_local_grib_files({"files": str(paths[0])}, "gefs", "model")
    assert "preprocess" not in single

    def user_preprocess(ds):
        return ds

    custom = _expand_local_grib_files({"files": str(tmp_path / "*.grib2"), "preprocess": user_preprocess}, "gefs", "model")
    assert custom["preprocess"] is user_preprocess


def test_promote_time_scalar_alias_does_not_expand_existing_dim(caplog):
    """A scalar valid-time alias must not expand an already existing time dim."""
    data = xr.Dataset(
        {"value": ("time", [1.0, 2.0])},
        coords={
            "init": ("time", pd.to_datetime(["2025-08-01", "2025-08-01"]), {"standard_name": "forecast_reference_time"}),
            "lead": ("time", [0, 3], {"standard_name": "forecast_period", "units": "hours"}),
            "validDate": ((), np.datetime64("2025-08-01"), {"standard_name": "time"}),
        },
    )
    with caplog.at_level("WARNING"):
        normalized = _promote_time_coordinate(data, "collapsed")
    assert "only a scalar valid time" in caplog.text
    assert normalized.sizes["time"] == 2
    assert "validDate" in normalized.coords


def test_promote_time_single_record_dim_gets_scalar_valid_time():
    """A one-record time axis is labeled from a scalar valid time."""
    data = xr.Dataset(
        {"value": ("time", [1.0])},
        coords={
            "init": ("time", pd.to_datetime(["2025-08-01"]), {"standard_name": "forecast_reference_time"}),
            "lead": ("time", [3], {"standard_name": "forecast_period", "units": "hours"}),
            "validDate": ((), np.datetime64("2025-08-01T03:00:00"), {"standard_name": "time"}),
        },
    )
    normalized = _promote_time_coordinate(data, "single")
    assert pd.Timestamp(normalized["time"].values[0]) == pd.Timestamp("2025-08-01T03:00:00")


def test_promote_time_prefers_canonical_coords_over_aliases():
    """Canonical coordinates win role resolution, avoiding rename collisions."""
    data = xr.Dataset(
        {"value": ("time", [1.0, 2.0])},
        coords={
            "time": pd.date_range("2025-08-01", periods=2, freq="3h"),
            "refDate": ((), np.datetime64("2025-08-01"), {"standard_name": "forecast_reference_time"}),
            "forecast_reference_time": ("time", pd.to_datetime(["2025-08-01", "2025-08-01"])),
        },
    )
    normalized = _promote_time_coordinate(data, "canonical")
    assert normalized["forecast_reference_time"].dims == ("time",)
    np.testing.assert_array_equal(normalized["time"].values, data["time"].values)


def _multi_cycle_forecast_dataset():
    """Two cycles whose lead times repeat on the shared time axis."""
    times = pd.DatetimeIndex(["2025-08-01", "2025-08-01 03:00", "2025-08-02", "2025-08-02 03:00"])
    return xr.Dataset(
        {"value": ("time", [1.0, 2.0, 3.0, 4.0])},
        coords={
            "time": times,
            "forecast_reference_time": ("time", pd.DatetimeIndex(["2025-08-01", "2025-08-01", "2025-08-02", "2025-08-02"]).values),
            "lead_time": ("time", np.array([0, 3, 0, 3], dtype="timedelta64[h]")),
        },
    )


def test_sel_labels_selects_repeated_lead_times_by_value():
    """Non-unique forecast coordinates are selected with value masks, not label indexing."""
    from mdt.tasks.data import _sel_labels

    data = _multi_cycle_forecast_dataset()
    result = _sel_labels(data, {"lead_time": [np.timedelta64(3, "h")]})
    assert result.sizes["time"] == 2
    np.testing.assert_array_equal(result["value"].values, [2.0, 4.0])


def test_sel_labels_slice_on_non_unique_coordinate():
    """Slice bounds work on repeated coordinates where pandas label slicing fails."""
    from mdt.tasks.data import _sel_labels

    data = _multi_cycle_forecast_dataset()
    result = _sel_labels(data, {"lead_time": slice(np.timedelta64(0, "h"), np.timedelta64(3, "h"))})
    assert result.sizes["time"] == 4


def test_sel_labels_parses_string_timedelta_labels():
    """YAML-friendly strings like '3 hours' select timedelta coordinates."""
    from mdt.tasks.data import _sel_labels

    data = _multi_cycle_forecast_dataset()
    result = _sel_labels(data, {"lead_time": ["0 hours", "3 hours"]})
    assert result.sizes["time"] == 4
    result = _sel_labels(data, {"lead_time": slice("0 hours", "3 hours")})
    assert result.sizes["time"] == 4


def test_sel_labels_integer_hours_select_timedelta_coordinate():
    """Plain numbers mean hours, matching the readers' lead_time option."""
    from mdt.tasks.data import _sel_labels

    data = _multi_cycle_forecast_dataset()
    result = _sel_labels(data, {"lead_time": [0, 3]})
    assert result.sizes["time"] == 4
    result = _sel_labels(data, {"lead_time": 3})
    assert result.sizes["time"] == 2
    np.testing.assert_array_equal(result["value"].values, [2.0, 4.0])
    result = _sel_labels(data, {"lead_time": slice(0, 3)})
    assert result.sizes["time"] == 4


def test_sel_labels_unique_index_uses_label_selection():
    """Unique coordinates keep ordinary label-based selection semantics."""
    from mdt.tasks.data import _sel_labels

    data = xr.Dataset(
        {"value": ("time", [1.0, 2.0, 3.0])},
        coords={"time": pd.date_range("2025-08-01", periods=3)},
    )
    result = _sel_labels(data, {"time": slice("2025-08-01", "2025-08-02")})
    assert result.sizes["time"] == 2
    np.testing.assert_array_equal(result["value"].values, [1.0, 2.0])
