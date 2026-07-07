#!/usr/bin/env bash
# Package + upload a fresh release, then install it on the VM over IAP SSH.
#
# Usage:
#   deploy/gcp/scripts/update_app.sh              # package, upload, install CURRENT
#   SKIP_PACKAGE=1 deploy/gcp/scripts/update_app.sh            # install CURRENT only
#   SKIP_PACKAGE=1 deploy/gcp/scripts/update_app.sh <version>  # rollback to a version
set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "${REPO_ROOT}"
TF_DIR="deploy/gcp/terraform"
TARGET_VERSION="${1:-}"

if [ "${SKIP_PACKAGE:-0}" != "1" ]; then
  deploy/gcp/scripts/package_release.sh
fi

SSH_CMD=$(terraform -chdir="${TF_DIR}" output -raw ssh_command 2>/dev/null || true)
if [ -z "${SSH_CMD}" ]; then
  echo "ERROR: no terraform outputs — run 'terraform apply' first." >&2
  exit 1
fi

echo "→ installing on the VM (${SSH_CMD})"
# shellcheck disable=SC2086
${SSH_CMD} --command "sudo /usr/local/bin/mmm-update ${TARGET_VERSION}"
