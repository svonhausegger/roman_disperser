"""The batch config's ``cache_dir`` in scripts/build_dispersed_image.py.

JAX reads ``JAX_COMPILATION_CACHE_DIR`` when it is imported, so before the
fix ``run_batch`` / ``run_warmup`` setting that variable from the config had
no effect: compiled programs kept going to the import-time directory.

Both functions are called with inputs that make them return right after
reading the config (no pointings / no SCAs for this worker), so nothing is
rendered. A small jitted function then has to land in the configured
directory.
"""

import importlib.util
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
import yaml
from astropy.table import Table
from jax.experimental.compilation_cache import compilation_cache as jax_cc

SCRIPTS = Path(__file__).parent.parent / "scripts"

_OPTIONS = ("jax_compilation_cache_dir",
            "jax_persistent_cache_min_compile_time_secs",
            "jax_persistent_cache_min_entry_size_bytes")


@pytest.fixture(scope="module")
def bdi():
    spec = importlib.util.spec_from_file_location(
        "build_dispersed_image", SCRIPTS / "build_dispersed_image.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def jax_cache(monkeypatch):
    """Cache every compile, and put the cache back as it was afterwards."""
    old = {name: getattr(jax.config, name) for name in _OPTIONS}
    # run_* assign os.environ directly; monkeypatch restores the variable.
    if "JAX_COMPILATION_CACHE_DIR" in os.environ:
        monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR",
                           os.environ["JAX_COMPILATION_CACHE_DIR"])
    else:
        monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    yield
    for name, value in old.items():
        jax.config.update(name, value)
    jax_cc.reset_cache()


def _compile_something(tag):
    """Compile a program no earlier test compiled (new constant each tag)."""
    c = float(sum(map(ord, tag)))
    jax.jit(lambda x: jnp.sin(x) * c + jnp.cos(x * c))(
        jnp.arange(3.0)).block_until_ready()


def _config(tmp_path, cache_dir, **extra):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({
        "output_dir": str(tmp_path / "out"), "seed": 1, "scas": [5],
        "cache_dir": str(cache_dir), **extra}))
    return str(cfg)


def test_run_batch_uses_config_cache_dir(bdi, jax_cache, tmp_path):
    cache_dir = tmp_path / "jax-cache"
    pointings = tmp_path / "pointings.ecsv"
    Table(names=["PLAN", "RA"], dtype=[int, float]).write(
        pointings, format="ascii.ecsv")              # no pointings: returns
    bdi.run_batch(_config(tmp_path, cache_dir), str(pointings), verbose=False)

    assert jax.config.jax_compilation_cache_dir == str(cache_dir)
    _compile_something("run_batch")
    assert cache_dir.is_dir() and any(cache_dir.iterdir())


def test_run_warmup_uses_config_cache_dir(bdi, jax_cache, tmp_path):
    cache_dir = tmp_path / "jax-cache"
    # One SCA, two workers: worker 1 gets no SCAs and returns.
    bdi.run_warmup(_config(tmp_path, cache_dir), verbose=False,
                   worker_index=1, num_workers=2)

    assert jax.config.jax_compilation_cache_dir == str(cache_dir)
    _compile_something("run_warmup")
    assert cache_dir.is_dir() and any(cache_dir.iterdir())
