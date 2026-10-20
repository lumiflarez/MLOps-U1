# Docker image (U1-R3)

## Image
- Dockerfile: docker/Dockerfile.train (base python:3.10-slim, non-root user app, UID 1000)
- Contents: Python 3.10, requirements.txt, src/, params.yaml, dvc.yaml 
- Registry: ghcr.io/lumiflarez/mlops-u1-train
- Tag: sha-dc34993
- Digest: ghcr.io/lumiflarez/mlops-u1-train@sha256:aa836e43877e95321c4420334775494da935ba87f4cc443bb4766aa219db516d
- Package page: https://github.com/users/lumiflarez/packages/container/package/mlops-u1-train
- Size: 950MB

## Trivy scan (2026-10-20, HIGH and CRITICAL only)
Full report: docs/trivy-train.txt

| Part | HIGH | CRITICAL |
|---|---|---|
| Debian 13.7 base packages | 44 | 0 |
| Python packages | 4 | 0 |
| Fixable (--ignore-unfixed) | 3 | 0 |

Assessment:
- Gate: 0 CRITICAL, so the image may be published.
- Debian HIGH findings are in base OS packages with no fixed version
  released yet (status "affected"). The pipeline does not use these tools, and the container
  runs as a non-root user, which limits privilege-related issues.
- Possible improvements: rebuild regularly to pick up Debian fixes; use a smaller base
  to remove unused OS tools; upgrade Python packages that have fixes.