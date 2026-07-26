"""zisk-zorch — a lean ZisK (pil2-stark eSTARK) prover built on zorch's scheme-agnostic blocks."""

# Single source of truth for the packaged version: pyproject.toml carries no
# literal of its own (it derives this one via `attr =`), and release.yml refuses
# to publish a tag that disagrees with it. Nothing rewrites this string at
# release time — the repo has no dev-release.yml stamping timestamped
# prereleases — so it is also exactly what an install reports.
__version__ = "0.1.0"
