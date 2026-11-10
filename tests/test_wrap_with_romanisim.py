"""Skip / re-wrap logic of scripts/wrap_with_romanisim.py.

romanisim itself is never run: ``subprocess.run`` is replaced by a fake that
records the call and writes a placeholder L2, so these run in the default
environment.
"""

import importlib.util
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning

SCRIPTS = Path(__file__).parent.parent / "scripts"

OLD_L2 = b"old L2 product"


@pytest.fixture(scope="module")
def wrap():
    spec = importlib.util.spec_from_file_location(
        "wrap_with_romanisim", SCRIPTS / "wrap_with_romanisim.py")
    mod = importlib.util.module_from_spec(spec)
    # Registered first: its dataclass needs to find the module in sys.modules.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop(spec.name, None)


@pytest.fixture
def fake_romanisim(monkeypatch):
    """Replace subprocess.run; record each call and whether the L2 existed."""
    calls = []

    def run(cmd, **kwargs):
        out = Path(cmd[1])
        calls.append({"cmd": cmd, "out_existed": out.exists()})
        out.write_bytes(b"new L2 product")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", run)
    return calls


def _write_fits(path, **drop_or_override):
    hdr = {"WFICENRA": 10.0, "WFICENDEC": -5.0, "WFICENPA": 120.0,
           "DETNUM": 5, "MA_TABLE": 1001, "RNDSEED0": 7, "RNDSEED1": 3,
           "OPTELEM": "GRISM"}
    hdr.update(drop_or_override)
    h = fits.Header()
    with warnings.catch_warnings():
        # WFICENDEC has 9 characters: a HIERARCH card, as in the real products.
        warnings.simplefilter("ignore", VerifyWarning)
        for k, v in hdr.items():
            if v is not None:
                h[k] = v
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(header=h).writeto(path)


def _setup(wrap, tmp_path, l2_age, **hdr):
    """A disperser FITS and, unless l2_age is None, an existing L2.

    l2_age > 0: the L2 is that many seconds older than the FITS (stale);
    l2_age <= 0: the L2 is at least as new as the FITS (fresh).
    """
    in_root, out_root = tmp_path / "output", tmp_path / "output_l2"
    fits_path = in_root / "p0001" / "grism_p0001_detSCA05.fits"
    _write_fits(fits_path, **hdr)
    t_fits = 1_700_000_000
    os.utime(fits_path, (t_fits, t_fits))
    out = wrap.derive_output_path(fits_path, in_root, out_root)
    if l2_age is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(OLD_L2)
        os.utime(out, (t_fits - l2_age, t_fits - l2_age))
    return fits_path, in_root, out_root, out


def _wrap_one(wrap, tmp_path, fits_path, in_root, out_root, dry_run=False):
    return wrap.wrap_one(fits_path, in_root, out_root, tmp_path / "logs",
                         wrap.DEFAULT_DATE, None, None, 2, dry_run)


@pytest.mark.parametrize("l2_age", [0, -60], ids=["same-mtime", "newer"])
def test_fresh_l2_is_skipped(wrap, tmp_path, fake_romanisim, l2_age):
    fits_path, in_root, out_root, out = _setup(wrap, tmp_path, l2_age)
    res = _wrap_one(wrap, tmp_path, fits_path, in_root, out_root)
    assert res.status == "skipped" and res.note == "output exists"
    assert fake_romanisim == []
    assert out.read_bytes() == OLD_L2


def test_missing_l2_is_wrapped(wrap, tmp_path, fake_romanisim):
    fits_path, in_root, out_root, out = _setup(wrap, tmp_path, None)
    res = _wrap_one(wrap, tmp_path, fits_path, in_root, out_root)
    assert res.status == "ok"
    assert len(fake_romanisim) == 1
    assert fake_romanisim[0]["cmd"][:2] == ["romanisim-make-image", str(out)]


def test_stale_l2_is_rewrapped(wrap, tmp_path, fake_romanisim):
    """An L2 older than its FITS is removed just before romanisim runs."""
    fits_path, in_root, out_root, out = _setup(wrap, tmp_path, 60)
    res = _wrap_one(wrap, tmp_path, fits_path, in_root, out_root)
    assert res.status == "ok"
    assert len(fake_romanisim) == 1
    assert fake_romanisim[0]["out_existed"] is False
    assert out.read_bytes() == b"new L2 product"


def test_dry_run_never_deletes_stale_l2(wrap, tmp_path, fake_romanisim):
    fits_path, in_root, out_root, out = _setup(wrap, tmp_path, 60)
    res = _wrap_one(wrap, tmp_path, fits_path, in_root, out_root,
                    dry_run=True)
    assert res.status == "skipped"
    assert res.note.startswith("dry-run: romanisim-make-image")
    assert fake_romanisim == []
    assert out.read_bytes() == OLD_L2


@pytest.mark.parametrize("hdr, note", [
    ({"DETNUM": None}, "header read error"),
    ({"OPTELEM": "IMAGER"}, "unknown OPTELEM"),
], ids=["missing-card", "bad-optelem"])
def test_failed_checks_never_delete_stale_l2(wrap, tmp_path, fake_romanisim,
                                             hdr, note):
    """A header or bandpass error on a stale input keeps the old L2."""
    fits_path, in_root, out_root, out = _setup(wrap, tmp_path, 60, **hdr)
    res = _wrap_one(wrap, tmp_path, fits_path, in_root, out_root)
    assert res.status == "failed"
    assert note in res.note
    assert fake_romanisim == []
    assert out.read_bytes() == OLD_L2
