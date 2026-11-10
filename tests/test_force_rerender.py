"""``--force`` in scripts/build_dispersed_image.py re-renders existing SCAs.

Before the fix, ``run_batch`` did not pass ``force`` to ``process_pointing``,
which skipped every SCA whose FITS was already on disk, so a forced run
rewrote the meta YAML but kept the old images.

No rendering happens here: the PSF loader, the first step after the
skip-or-render decision, is replaced by a stub that raises ``_Rendered``.
"""

import importlib.util
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from astropy.io import fits
from astropy.table import Table

from roman_disperser import elements

SCRIPTS = Path(__file__).parent.parent / "scripts"


@pytest.fixture(scope="module")
def bdi():
    spec = importlib.util.spec_from_file_location(
        "build_dispersed_image", SCRIPTS / "build_dispersed_image.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Rendered(Exception):
    """Raised by the PSF-loader stub: process_pointing started rendering."""


def _raise_rendered(**kwargs):
    raise _Rendered


@pytest.fixture
def no_render(bdi, monkeypatch):
    monkeypatch.setattr(bdi, "psf_model", types.SimpleNamespace(
        get_or_make_psf_payload=_raise_rendered))


def _pipeline():
    """The fields process_pointing reads, for an empty catalog on SCA 5."""
    return {
        "meta": pd.DataFrame({"ra": np.array([], float),
                              "dec": np.array([], float),
                              "type": np.array([], str)}),
        "element": elements.get_element("grism"),
        "sca_list": [5],
        "sca_data": {5: {}},
        "star_batch_size": 1, "galaxy_batch_size": 1, "galaxy_npix": 1,
        "galaxy_npix_os": 1, "oversample": 1,
        "optical_model_file": "test", "source_cuts": [],
        "wavelengths_um": np.zeros(1), "wavelengths_jax": None,
        "dlam_angstroms": 2.0, "psf_cache_dir": None, "n_catalog_rows": 0,
    }


def _existing_fits(output_dir):
    """An 'old' SCA 5 FITS where process_pointing will look for it."""
    output_dir.mkdir(parents=True)
    path = output_dir / f"grism_{output_dir.name}_detSCA05.fits"
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(np.full((4, 4), 7.0, np.float32), name="MODEL"),
    ]).writeto(path)
    return path


class TestProcessPointing:
    def test_existing_sca_skipped_without_force(self, bdi, no_render, tmp_path):
        out = tmp_path / "pointing"
        path = _existing_fits(out)
        before = path.read_bytes()
        bdi.process_pointing(_pipeline(), 0.0, 0.0, 0.0, out, verbose=False)
        assert path.read_bytes() == before     # kept, never re-rendered

    def test_force_rerenders_existing_sca(self, bdi, no_render, tmp_path):
        out = tmp_path / "pointing"
        path = _existing_fits(out)
        with pytest.raises(_Rendered):
            bdi.process_pointing(_pipeline(), 0.0, 0.0, 0.0, out,
                                 verbose=False, force=True)
        assert not path.exists()               # old FITS removed first


class TestRunBatchForce:
    """run_batch passes force through to process_pointing."""

    def _setup(self, bdi, tmp_path, monkeypatch):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.safe_dump({
            "output_dir": str(tmp_path / "out"), "seed": 1, "scas": [5]}))
        pointings = tmp_path / "pointings.ecsv"
        row = {"PLAN": 1, "PASS": 1, "SEGMENT": 1, "OBSERVATION": 1,
               "VISIT": 1, "EXPOSURE": 1, "RA": 10.0, "DEC": -20.0,
               "PA": 30.0, "EXPOSURE_TIME": 190.22, "MA_TABLE_NUMBER": 1036}
        Table(rows=[row]).write(pointings, format="ascii.ecsv")
        # The pointing was rendered before: its directory exists.
        (tmp_path / "out" / bdi._pointing_dir_name("pointings", row)).mkdir(
            parents=True)

        calls = []
        monkeypatch.setattr(bdi, "setup_pipeline", lambda *a, **k: {
            "timings": {"setup_total": 0.0}})
        monkeypatch.setattr(bdi, "process_pointing",
                            lambda *a, **k: calls.append(k))
        return str(cfg), str(pointings), calls

    def test_force_reaches_process_pointing(self, bdi, tmp_path, monkeypatch):
        cfg, pointings, calls = self._setup(bdi, tmp_path, monkeypatch)
        bdi.run_batch(cfg, pointings, verbose=False, force=True)
        assert len(calls) == 1
        assert calls[0].get("force") is True

    def test_existing_pointing_skipped_without_force(self, bdi, tmp_path,
                                                     monkeypatch):
        cfg, pointings, calls = self._setup(bdi, tmp_path, monkeypatch)
        bdi.run_batch(cfg, pointings, verbose=False)
        assert calls == []
