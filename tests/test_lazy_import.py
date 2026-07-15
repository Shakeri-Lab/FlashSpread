"""Subprocess checks for the lightweight root-package import contract."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import textwrap


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_python(source: str, *, no_site: bool = False) -> subprocess.CompletedProcess[str]:
    command = [sys.executable]
    if no_site:
        command.append("-S")
    command.extend(("-c", textwrap.dedent(source)))
    return subprocess.run(
        command,
        cwd=_PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_bare_import_is_metadata_only_without_site_packages():
    result = _run_python(
        f"""
        import sys
        sys.path.insert(0, {_PROJECT_ROOT.as_posix()!r})

        import flashspread

        forbidden = {{"torch", "numpy", "triton", "networkx", "scipy"}}
        loaded = sorted(
            name for name in sys.modules if name.partition(".")[0] in forbidden
        )
        assert not loaded, loaded
        assert flashspread.__version__ == "1.0.0"
        assert flashspread.__author__ == "Heman Shakeri"
        assert flashspread.__email__ == "hs9hd@virginia.edu"
        assert all(
            name == "__version__" or name not in flashspread.__dict__
            for name in flashspread.__all__
        )
        """,
        no_site=True,
    )
    assert result.returncode == 0, result.stderr


def test_bare_core_import_is_also_metadata_only_without_site_packages():
    result = _run_python(
        f"""
        import sys
        sys.path.insert(0, {_PROJECT_ROOT.as_posix()!r})

        import flashspread.core

        forbidden = {{"torch", "numpy", "triton", "networkx", "scipy"}}
        loaded = sorted(
            name for name in sys.modules if name.partition(".")[0] in forbidden
        )
        assert not loaded, loaded
        assert set(name for name in sys.modules if name.startswith("flashspread")) == {{
            "flashspread",
            "flashspread.core",
        }}
        assert "GraphCSR" in dir(flashspread.core)
        assert "GraphCSR" not in flashspread.core.__dict__
        """,
        no_site=True,
    )
    assert result.returncode == 0, result.stderr


def test_bare_engines_import_is_metadata_only_without_site_packages():
    result = _run_python(
        f"""
        import sys
        sys.path.insert(0, {_PROJECT_ROOT.as_posix()!r})

        import flashspread.engines

        forbidden = {{"torch", "numpy", "triton", "networkx", "scipy"}}
        loaded = sorted(
            name for name in sys.modules if name.partition(".")[0] in forbidden
        )
        assert not loaded, loaded
        assert set(name for name in sys.modules if name.startswith("flashspread")) == {{
            "flashspread",
            "flashspread.engines",
        }}
        assert "RenewalEngine" in dir(flashspread.engines)
        assert "RenewalEngine" not in flashspread.engines.__dict__
        """,
        no_site=True,
    )
    assert result.returncode == 0, result.stderr


def test_engine_config_export_and_resolution_do_not_import_torch():
    result = _run_python(
        f"""
        import sys
        from typing import get_type_hints
        sys.path.insert(0, {_PROJECT_ROOT.as_posix()!r})

        from flashspread import EngineConfig

        class CPUDevice:
            type = "cpu"

        class RenewalModel:
            transmission_mode = "constant"

        resolved = EngineConfig().resolve(
            CPUDevice(), markovian=False, model=RenewalModel()
        )
        assert resolved["use_fused"] is False
        assert get_type_hints(EngineConfig.resolve)["device"].__name__ == "_DeviceLike"
        forbidden = {{"torch", "numpy", "triton", "networkx", "scipy"}}
        loaded = sorted(
            name for name in sys.modules if name.partition(".")[0] in forbidden
        )
        assert not loaded, loaded
        """,
        no_site=True,
    )
    assert result.returncode == 0, result.stderr


def test_public_exports_resolve_lazily_and_preserve_from_imports():
    result = _run_python(
        """
        import flashspread

        blessed = set(flashspread.__all__) - {"__version__"}
        assert blessed <= set(dir(flashspread))
        assert all(name not in flashspread.__dict__ for name in blessed)

        star_namespace = {}
        exec("from flashspread import *", star_namespace)
        assert all(name in star_namespace for name in flashspread.__all__)
        assert all(name in flashspread.__dict__ for name in blessed)

        from flashspread import (
            EnsembleEngine,
            FixedDegreeGraph,
            FlashNeighbor,
            FlashNeighborInfectivity,
            MarkovianEngine,
            MarkovianEngineCUDAGraph,
            RandomGeometricGraph,
            ReferenceEnsembleEngine,
            RenewalEngine,
            RenewalEngineCUDAGraph,
            RenewalEngineNonMarkov,
            RenewalEngineNonMarkovCUDAGraph,
            load_graph,
        )

        compatibility_exports = {
            EnsembleEngine,
            FixedDegreeGraph,
            FlashNeighbor,
            FlashNeighborInfectivity,
            MarkovianEngine,
            MarkovianEngineCUDAGraph,
            RandomGeometricGraph,
            ReferenceEnsembleEngine,
            RenewalEngine,
            RenewalEngineCUDAGraph,
            RenewalEngineNonMarkov,
            RenewalEngineNonMarkovCUDAGraph,
            load_graph,
        }
        assert len(compatibility_exports) == 13
        """
    )
    assert result.returncode == 0, result.stderr
