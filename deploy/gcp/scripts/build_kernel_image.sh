#!/usr/bin/env bash
# Build, verify, and push the sandbox kernel image to Artifact Registry.
#
# Usage (from anywhere in the repo, after `terraform apply`):
#   deploy/gcp/scripts/build_kernel_image.sh
#
# Env overrides:
#   REGISTRY        <region>-docker.pkg.dev/<project>/<repo>  (default: terraform output)
#   KERNEL_RUNTIME  podman|docker                             (default: podman, falls back to docker)
#   TAG             image tag                                 (default: latest)
set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "${REPO_ROOT}"
TF_DIR="deploy/gcp/terraform"

REGISTRY="${REGISTRY:-$(terraform -chdir="${TF_DIR}" output -raw artifact_registry 2>/dev/null || true)}"
if [ -z "${REGISTRY}" ]; then
  echo "ERROR: no registry. Run 'terraform apply' first or set REGISTRY=<region>-docker.pkg.dev/<project>/<repo>" >&2
  exit 1
fi

RUNTIME="${KERNEL_RUNTIME:-podman}"
command -v "${RUNTIME}" >/dev/null 2>&1 || RUNTIME=docker
command -v "${RUNTIME}" >/dev/null 2>&1 || { echo "ERROR: need podman or docker" >&2; exit 1; }

TAG="${TAG:-latest}"
IMAGE="${REGISTRY}/mmm-kernel:${TAG}"
REGISTRY_HOST="${REGISTRY%%/*}"

if [ ! -f deploy/kernel/requirements.lock ]; then
  echo "→ regenerating deploy/kernel/requirements.lock (make kernel-lock)"
  make kernel-lock
fi

echo "→ building ${IMAGE} (${RUNTIME})"
"${RUNTIME}" build -t "${IMAGE}" -f deploy/kernel/Containerfile .

echo "→ verifying under the real sandbox flags (read-only, no network, no caps)"
"${RUNTIME}" run --rm --read-only --tmpfs /tmp --network none --cap-drop ALL \
  --user 10001 "${IMAGE}" \
  python -c "import mmm_framework, ipykernel; print('kernel image OK:', mmm_framework.__name__)"

echo "→ pushing to ${REGISTRY_HOST}"
gcloud auth print-access-token \
  | "${RUNTIME}" login --username oauth2accesstoken --password-stdin "${REGISTRY_HOST}"
"${RUNTIME}" push "${IMAGE}"

DIGEST=$(gcloud artifacts docker images describe "${IMAGE}" \
  --format 'value(image_summary.digest)' 2>/dev/null || true)

echo
echo "✓ pushed ${IMAGE}"
if [ -n "${DIGEST}" ]; then
  echo
  echo "Pin the sandbox image by digest (supply-chain pinning) in terraform.tfvars:"
  echo "  kernel_image = \"${REGISTRY}/mmm-kernel@${DIGEST}\""
  echo "then 'terraform apply' and reboot the VM (or run scripts/update_app.sh)."
fi
