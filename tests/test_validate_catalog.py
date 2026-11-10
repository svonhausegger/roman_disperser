"""Catalog checks in scripts/build_dispersed_image.py (validate_catalog).

A NaN galaxy morphology gives a NaN swath in MODEL, and a negative star
sed_index wraps around to another template; both must fail at load time.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from roman_disperser import elements

SCRIPTS = Path(__file__).parent.parent / "scripts"
WL = np.arange(8000.0, 21001.0, 100.0)    # covers the grism band, uniform


@pytest.fixture(scope="module")
def bdi():
    spec = importlib.util.spec_from_file_location(
        "build_dispersed_image", SCRIPTS / "build_dispersed_image.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _store():
    # The few keys validate_catalog reads: star_seds, the galaxy_seds group
    # attrs (sim layout) and one sim partition.
    return {
        "star_seds": np.ones((2, len(WL)), np.float32),
        "galaxy_seds": SimpleNamespace(attrs={}),
        "galaxy_seds/sim_000": np.ones((4, len(WL)), np.float32),
    }


def _meta():
    # two galaxies (one with r_e = 0) and two stars; stars carry NaN
    # morphology, which they never use
    return pd.DataFrame({
        "ra": [10.0, 10.01, 10.02, 10.03],
        "dec": [0.0, 0.01, 0.02, 0.03],
        "type": ["SER", "SER", "PSF", "PSF"],
        "n": [1.0, 4.0, np.nan, np.nan],
        "half_light_radius": [0.3, 0.0, np.nan, np.nan],
        "pa": [10.0, 45.0, np.nan, np.nan],
        "ba": [0.5, 1.0, np.nan, np.nan],
        "sed_index": [0, 3, 0, 1],
        "flux_scale": [1.0, 1.0, 1.0, 1.0],
        "sim": [0, 0, 0, 0],
    })


def _validate(bdi, meta):
    bdi.validate_catalog(meta, _store(), WL, elements.get_element("grism"))


def test_valid_catalog_accepted(bdi):
    """Stars with NaN morphology and a galaxy with r_e = 0 are fine."""
    _validate(bdi, _meta())


@pytest.mark.parametrize("col,value,match", [
    ("n", np.nan, "galaxy column 'n'"),
    ("half_light_radius", np.inf, "galaxy column 'half_light_radius'"),
    ("pa", np.nan, "galaxy column 'pa'"),
    ("ba", np.nan, "galaxy column 'ba'"),
    ("n", 0.0, "out of range"),
    ("n", -1.0, "out of range"),
    ("ba", 0.0, "out of range"),
    ("half_light_radius", -0.1, "out of range"),
])
def test_bad_galaxy_morphology_raises(bdi, col, value, match):
    meta = _meta()
    meta.loc[0, col] = value          # row 0 is a galaxy
    with pytest.raises(ValueError, match=match):
        _validate(bdi, meta)


def test_negative_star_sed_index_raises(bdi):
    """Negative indexing would silently give the star the last template."""
    meta = _meta()
    meta.loc[3, "sed_index"] = -1     # row 3 is a star
    with pytest.raises(ValueError, match="Negative sed_index"):
        _validate(bdi, meta)
