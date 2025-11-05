# Releasing Haxen

This guide explains how maintainers publish new versions of the Haxen control plane, SDKs, and container images.

## Versioning

- Tag releases with semantic versions: `vMAJOR.MINOR.PATCH`.
- The tag drives the automation. Tagging `main` (or dispatching the workflow manually) kicks off Go binary builds, Python package publishing, and Docker image pushes.
- Update component versions before tagging:
  - `sdk/python/pyproject.toml` & `sdk/python/haxen_sdk/__init__.py`
  - Release notes in `CHANGELOG.md`

## Required Secrets

Store these secrets at the repository level:

| Secret | Purpose |
| --- | --- |
| `PYPI_API_TOKEN` | PyPI token used by `twine upload` (username `__token__`). |
| `DOCKER_REGISTRY_USER` / `DOCKER_REGISTRY_PASSWORD` *(optional)* | Use if you push images somewhere other than GHCR. |

GitHub’s built-in `GITHUB_TOKEN` is sufficient for publishing releases and pushing to GHCR (`ghcr.io`).

## Release Workflow (`release.yml`)

Triggers:

- `git push origin vX.Y.Z`
- `workflow_dispatch` (manual run) – toggle PyPI / Docker publishing via inputs.

What happens:

1. Checkout with tags.
2. Install Go, Node.js, Python tooling.
3. Build the control plane UI (`npm install && npm run build`).
4. Run [GoReleaser](https://goreleaser.com) using `.goreleaser.yml` to produce multi-platform binaries and attach them to the GitHub release.
5. Build the Python SDK (`python -m build`) and, if enabled, publish to PyPI with `twine upload`.
6. Build and push the `Dockerfile.control-plane` image (defaults to `ghcr.io/<org>/haxen-control-plane:<tag>`).

Artifacts:

- Release binaries (`haxen-server` for Linux/Darwin/Windows, amd64/arm64).
- Python SDK wheel & sdist on PyPI (and attached to the release for manual runs).
- Multi-architecture Docker image.

## Dry Runs / Pre-Releases

Use `workflow_dispatch` to stage a release without pushing external artifacts:

1. Open the **Actions** tab → **Release** workflow.
2. Click **Run workflow**.
3. Supply a branch/ref and set both “Publish to PyPI” and “Publish Docker image” to `false`.
4. Artifacts appear under the run’s summary for download/testing.

## Testing Release Artifacts

- **Go binaries**: download from the release page or workflow artifacts and run `haxen-server --help`. Cross-platform builds are generated for Linux (amd64/arm64), Darwin (amd64/arm64), and Windows (amd64).
- **Python package**: install locally via `pip install --index-url https://test.pypi.org/simple haxen-sdk` if you push to TestPyPI first, or install from the generated wheel.
- **Docker image**: `docker run --rm ghcr.io/<org>/haxen-control-plane:<tag> --help`.

## Emergency Fixes

1. Cherry-pick the fix onto `main`.
2. Bump component versions if required.
3. Tag `vX.Y.Z+1`.
4. Re-run the Release workflow.

## Related Documentation

- `docs/CONTRIBUTING.md` – expectations for contributors.
- `docs/DEVELOPMENT.md` – local development commands; references this document for publishing.

