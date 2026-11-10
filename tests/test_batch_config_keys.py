"""run_batch in scripts/build_dispersed_image.py rejects unknown config keys.

Before the fix, a misspelled key (``psf_cache_directory``) silently fell back
to the default. The checks below return or raise before any setup: either
on the unknown key, or (for accepted configs) on a pointing table with no
rows.
"""

import importlib.util
import re
from pathlib import Path

import pytest
import yaml
from astropy.table import Table

SCRIPTS = Path(__file__).parent.parent / "scripts"
SCRIPT = SCRIPTS / "build_dispersed_image.py"


@pytest.fixture(scope="module")
def bdi():
    spec = importlib.util.spec_from_file_location("build_dispersed_image",
                                                  SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def no_pointings(tmp_path):
    path = tmp_path / "pointings.ecsv"
    Table(names=["PLAN", "RA"], dtype=[int, float]).write(
        path, format="ascii.ecsv")
    return str(path)


@pytest.fixture(autouse=True)
def _keep_env(monkeypatch):
    # run_batch assigns os.environ for cache_dir; restore it afterwards.
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", "unused")


def _write(tmp_path, cfg):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def _base(tmp_path):
    return {"output_dir": str(tmp_path / "out"), "seed": 1}


def test_unknown_keys_raise(bdi, tmp_path, no_pointings):
    cfg = {**_base(tmp_path), "psf_cache": "x", "exptime": 100.0}
    with pytest.raises(ValueError,
                       match=r"Unknown config keys \['exptime', 'psf_cache'\]"):
        bdi.run_batch(_write(tmp_path, cfg), no_pointings, verbose=False)


def test_every_key_the_script_reads_is_accepted(bdi, tmp_path, no_pointings):
    """Keys read via cfg.get("k") / cfg["k"] must all be in the list.

    Guards the list going stale when a new config key is added.
    """
    read = set(re.findall(r'cfg(?:\.get\(|\[)\s*"(\w+)"', SCRIPT.read_text()))
    assert {"seed", "psf_cache_dir", "cache_dir", "source_cuts"} <= read
    cfg = {key: "x" for key in read}
    cfg.update(_base(tmp_path), element="grism",
               cache_dir=str(tmp_path / "cache"))
    assert bdi.run_batch(_write(tmp_path, cfg), no_pointings,
                         verbose=False) is None


def test_deprecated_batch_size_still_accepted(bdi, tmp_path, no_pointings):
    cfg = {**_base(tmp_path), "batch_size": 500}
    with pytest.warns(FutureWarning, match="batch_size"):
        bdi.run_batch(_write(tmp_path, cfg), no_pointings, verbose=False)


@pytest.mark.parametrize("name", ["example_grism_config.yaml",
                                  "example_prism_config.yaml"])
def test_example_configs_accepted(bdi, name, no_pointings):
    bdi.run_batch(str(SCRIPTS / name), no_pointings, verbose=False)
