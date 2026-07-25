"""Guards on the repository's own contracts, so documentation cannot drift.

Three things went stale unnoticed and each is cheap to pin:

* the documented lint invocation and the one CI runs must be the same command,
  or contributors and CI enforce different rules;
* the release workflow must stay gated on the test workflow, otherwise a red
  suite is publishable to PyPI (which it was);
* every acceptance preset that exists must appear in the documented evidence
  commands, otherwise features ship measured by nothing -- the hub-heavy
  traversals and the whole compaction path were in exactly that position.
"""

import re
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    path = REPOSITORY / relative
    if not path.exists():
        pytest.skip(f"{relative} is not present in this checkout")
    return path.read_text()


def _ruff_paths(text: str) -> list[str]:
    """Extract the argument list of the first ``ruff check`` invocation."""
    match = re.search(r"ruff check([^\n]*(?:\\\n[^\n]*)*)", text)
    assert match, "no `ruff check` invocation found"
    body = match.group(1).replace("\\\n", " ")
    return sorted(token for token in body.split() if token and not token.startswith("-"))


def test_ci_lints_exactly_what_the_docs_prescribe():
    """CLAUDE.md and CI must run the same ruff command."""
    documented = _ruff_paths(_read("CLAUDE.md"))
    workflow = _ruff_paths(_read(".github/workflows/ci.yml"))
    assert documented == workflow, (
        "CLAUDE.md and .github/workflows/ci.yml disagree on the lint targets.\n"
        f"  docs: {documented}\n"
        f"    ci: {workflow}"
    )


def test_publish_is_gated_on_the_test_workflow():
    """A red suite must not be publishable to PyPI."""
    publish = _read(".github/workflows/publish.yml")
    assert "ci.yml" in publish, (
        "publish.yml must reuse the CI workflow so releases cannot skip tests"
    )
    for job in ("build", "publish"):
        pattern = rf"^  {job}:\n(?:.*\n)*?    needs:.*$"
        assert re.search(pattern, publish, re.MULTILINE), (
            f"the {job!r} job must declare `needs:` on the verify job"
        )


def test_every_acceptance_preset_is_documented_as_evidence():
    """A preset with no documented command is a feature measured by nothing."""
    harness = _read("experiments/benchmark_acceptance.py")
    block = re.search(r"PRESETS = \{(.*?)\n\}", harness, re.DOTALL)
    assert block, "could not locate the PRESETS table"
    presets = set(re.findall(r'"([a-z0-9-]+)":', block.group(1)))
    assert presets, "parsed an empty PRESETS table"

    documented = set(
        re.findall(
            r"benchmark_acceptance\.py walltime --case ([a-z0-9-]+)",
            _read("CLAUDE.md"),
        )
    )
    missing = sorted(presets - documented)
    assert not missing, (
        "these acceptance presets exist but no documented command runs them, so "
        "they are never published as evidence: " + ", ".join(missing)
    )


def test_correctness_gates_in_slurm_wrappers_are_not_swallowed():
    """`pytest ... || echo` turns a failing gate into a passing job."""
    offenders = []
    for script in sorted((REPOSITORY / "slurm").glob("*.sbatch")):
        text = script.read_text()
        for match in re.finditer(r"^.*\|\|\s*echo.*$", text, re.MULTILINE):
            line = match.group(0)
            # nvidia-smi metadata fallbacks and the explicitly best-effort
            # figure regeneration are legitimate; a swallowed test is not.
            if "pytest" in line or "-v -s" in line:
                offenders.append(f"{script.name}: {line.strip()}")
    assert not offenders, (
        "a correctness gate must fail its job rather than log and continue:\n"
        + "\n".join(offenders)
    )
