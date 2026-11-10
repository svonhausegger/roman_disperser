"""Pointing-table checks in run_batch (scripts/build_dispersed_image.py).

Two rows with the same APT id tuple share an output directory and an RNG
key, so the second silently overwrote the first. run_batch must refuse
such a table before any rendering.

The output directories are created up front, so a table that passes the
check stops at "Nothing to do" and nothing is set up or rendered.
"""

import importlib.util
from pathlib import Path

import pytest
import yaml
from astropy.table import Table

SCRIPTS = Path(__file__).parent.parent / "scripts"


@pytest.fixture(scope="module")
def bdi():
    spec = importlib.util.spec_from_file_location(
        "build_dispersed_image", SCRIPTS / "build_dispersed_image.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(bdi, tmp_path, rows):
    """Write a pointing table from (BANDPASS, EXPOSURE) rows and run_batch it."""
    ptable = Table.read(SCRIPTS / "example_pointings.ecsv", format="ascii.ecsv")
    t = ptable[[0] * len(rows)]                     # copies of row 0
    for i, (band, exposure) in enumerate(rows):
        t["BANDPASS"][i] = band
        t["EXPOSURE"][i] = exposure
        t["RA"][i] += 0.5 * i                       # distinct pointings
    ppath = tmp_path / "pointings.ecsv"
    t.write(ppath, format="ascii.ecsv")

    out = tmp_path / "out"
    for r in t:
        (out / bdi._pointing_dir_name(ppath.stem, r)).mkdir(parents=True,
                                                             exist_ok=True)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"seed": 1, "output_dir": str(out),
                                   "scas": [5]}))
    return bdi.run_batch(str(cfg), str(ppath), verbose=False)


def test_repeated_id_tuple_raises(bdi, tmp_path):
    with pytest.raises(ValueError,
                       match=r"repeated APT id tuples \[\(1, 1, 1, 1, 1, 2\)\]"):
        _run(bdi, tmp_path, [("GRISM", 1), ("GRISM", 2), ("GRISM", 2)])


def test_unique_ids_pass(bdi, tmp_path):
    assert _run(bdi, tmp_path, [("GRISM", 1), ("GRISM", 2)]) is None


def test_repeat_in_other_element_is_ignored(bdi, tmp_path):
    """Only this element's rows are rendered, so only they must be unique."""
    assert _run(bdi, tmp_path, [("GRISM", 1), ("PRISM", 1)]) is None
