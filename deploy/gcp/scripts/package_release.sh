#!/usr/bin/env bash
# Package a release (backend source + built frontend) and upload it to the
# release bucket. The VM installs it with `sudo mmm-update` (scripts/update_app.sh
# does that over SSH for you).
#
# Usage:
#   deploy/gcp/scripts/package_release.sh
#
# Env overrides:
#   BUCKET   release bucket        (default: terraform output release_bucket)
#   VERSION  release label         (default: <utc-timestamp>-<git-sha>)
#
# NOTE: packages `git archive HEAD` — committed work only. A dirty tree gets a
# loud warning so you don't silently ship stale code.
set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "${REPO_ROOT}"
TF_DIR="deploy/gcp/terraform"

BUCKET="${BUCKET:-$(terraform -chdir="${TF_DIR}" output -raw release_bucket 2>/dev/null || true)}"
if [ -z "${BUCKET}" ]; then
  echo "ERROR: no bucket. Run 'terraform apply' first or set BUCKET=<name>" >&2
  exit 1
fi

VERSION="${VERSION:-$(date -u +%Y%m%d%H%M%S)-$(git rev-parse --short HEAD)}"

if ! git diff-index --quiet HEAD --; then
  echo "WARNING: working tree is dirty — the release contains HEAD, not your uncommitted changes." >&2
fi

TMP=$(mktemp -d)
trap 'rm -rf "${TMP}"' EXIT

echo "→ packaging backend source (git archive HEAD)"
git archive --format=tar.gz -o "${TMP}/source.tar.gz" HEAD

echo "→ building frontend (npm ci && npm run build, API at /api)"
# VITE_API_URL is baked in at build time; the production default in
# frontend/src/api/client.ts is http://localhost:8000, so it MUST be /api here
# to pair with the VM's nginx proxy (which strips the /api prefix).
(cd frontend && npm ci --silent && VITE_API_URL=/api npm run build)
tar -czf "${TMP}/frontend-dist.tar.gz" -C frontend/dist .

echo "→ uploading release ${VERSION} to gs://${BUCKET}"
gcloud storage cp "${TMP}/source.tar.gz" "gs://${BUCKET}/releases/${VERSION}/source.tar.gz"
gcloud storage cp "${TMP}/frontend-dist.tar.gz" "gs://${BUCKET}/releases/${VERSION}/frontend-dist.tar.gz"
printf '%s' "${VERSION}" | gcloud storage cp - "gs://${BUCKET}/releases/CURRENT"

echo
echo "✓ release ${VERSION} uploaded and marked CURRENT"
echo "  install it on the VM: deploy/gcp/scripts/update_app.sh"
