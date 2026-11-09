"""validate_catalog: each hard check still fires (vectorised / threaded version)."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

zarr = pytest.importorskip("zarr")

from roman_disperser import elements

SCRIPTS = Path(__file__).parent.parent / "scripts"
_E = elements.get_element(None)
N_WL = 8
WL = np.linspace(_E.lam_min * 1e4 - 100, _E.lam_max * 1e4 + 100, N_WL)


@pytest.fixture(scope="module")
def bdi():
    spec = importlib.util.spec_from_file_location(
        "build_dispersed_image", SCRIPTS / "build_dispersed_image.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _store(tmp_path, sims=(0, 1), widths=None):
    root = zarr.open_group(str(tmp_path / "s.zarr"), mode="w")
    root.create_array("star_seds", shape=(3, N_WL), dtype="f4")
    root.create_group("galaxy_seds")
    for s in sims:
        w = (widths or {}).get(s, N_WL)
        root.create_array(f"galaxy_seds/sim_{s:03d}", shape=(5, w), dtype="f4")
    return root


def _meta(sims=(0, 1, 1, 0), hlr=(0.2, 0.3, 1.5, 0.1)):
    n = len(sims)
    return pd.DataFrame({
        "ra": np.zeros(n + 1), "dec": np.zeros(n + 1),
        "type": ["SER"] * n + ["PSF"],
        "n": [1.0] * n + [0.0], "half_light_radius": list(hlr) + [0.0],
        "pa": [0.0] * (n + 1), "ba": [0.5] * n + [0.0],
        "sed_index": list(range(n)) + [2], "flux_scale": [1.0] * (n + 1),
        "sim": list(sims) + [0],
    })


def test_valid_catalog_passes_and_warns(bdi, tmp_path, capsys):
    bdi.validate_catalog(_meta(), _store(tmp_path), WL, _E)
    assert "1 galaxies have half_light_radius > 1 arcsec" in capsys.readouterr().out


def test_missing_partition(bdi, tmp_path):
    with pytest.raises(ValueError, match="Missing Zarr array for partition: galaxy_seds/sim_001"):
        bdi.validate_catalog(_meta(), _store(tmp_path, sims=(0,)), WL, _E)


def test_partition_width_mismatch(bdi, tmp_path):
    with pytest.raises(ValueError, match="sim_001 has 7 wavelength bins"):
        bdi.validate_catalog(_meta(), _store(tmp_path, widths={1: 7}), WL, _E)


def test_galaxy_morphology_checks(bdi, tmp_path):
    m = _meta(); m.loc[1, "ba"] = 0.0
    with pytest.raises(ValueError, match="out of range"):
        bdi.validate_catalog(m, _store(tmp_path), WL, _E)
    m = _meta(); m.loc[2, "pa"] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf in galaxy column 'pa'"):
        bdi.validate_catalog(m, _store(tmp_path), WL, _E)


def test_star_sed_index_bound(bdi, tmp_path):
    m = _meta(); m.loc[len(m) - 1, "sed_index"] = 3
    with pytest.raises(ValueError, match="Star sed_index 3 exceeds"):
        bdi.validate_catalog(m, _store(tmp_path), WL, _E)
