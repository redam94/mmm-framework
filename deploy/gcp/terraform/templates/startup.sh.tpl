#!/usr/bin/env bash
# Thin GCE startup script (templated by Terraform — deploy/gcp/terraform/compute.tf).
# Writes the deployment facts to /etc/mmm/deploy.env, then downloads and runs
# the real provisioning script (deploy/gcp/vm/vm_setup.sh, seeded into the
# release bucket by Terraform). Runs on EVERY boot; vm_setup.sh is idempotent.
set -euo pipefail
exec > >(tee -a /var/log/mmm-startup.log) 2>&1
echo "=== mmm startup $(date -u +%FT%TZ) ==="

mkdir -p /etc/mmm

# Terraform-templated deployment facts (no secrets here — secrets are fetched
# from Secret Manager by vm_setup.sh using the VM's service account).
cat > /etc/mmm/deploy.env <<'DEPLOY_ENV'
GCP_PROJECT="${project_id}"
GCP_REGION="${region}"
RELEASE_BUCKET="${release_bucket}"
REGISTRY_HOST="${registry_host}"
KERNEL_IMAGE="${kernel_image}"
AUTH_SECRET_NAME="${auth_secret_name}"
BOOTSTRAP_PW_SECRET_NAME="${bootstrap_pw_secret}"
LLM_KEY_SECRET_NAME="${llm_key_secret}"
SITE_ORIGIN="${site_origin}"
LLM_PROVIDER="${llm_provider}"
LLM_MODEL="${llm_model}"
LLM_LOCATION="${llm_location}"
EMBED_PROVIDER="${embed_provider}"
EMBED_MODEL="${embed_model}"
EMBED_LOCATION="${embed_location}"
BOOTSTRAP_ORG="${bootstrap_org}"
BOOTSTRAP_EMAIL="${bootstrap_email}"
MAX_KERNELS="${max_kernels}"
KERNEL_MEM="${kernel_mem}"
KERNEL_CPUS="${kernel_cpus}"
KERNEL_PIDS="${kernel_pids}"
DATA_DISK_DEVICE="/dev/disk/by-id/google-${data_disk_device}"
DEPLOY_ENV
chmod 0644 /etc/mmm/deploy.env

# Optional extra env for the API process (from var.extra_env).
cat > /etc/mmm/extra.env <<'EXTRA_ENV'
${extra_env_lines}
EXTRA_ENV

# Fetch vm_setup.sh from the release bucket using the metadata-server token
# (python3 ships with Ubuntu; no gcloud needed this early).
TOKEN=$(curl -sf -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["access_token"])')

curl -fsSL -H "Authorization: Bearer $${TOKEN}" \
  "https://storage.googleapis.com/${release_bucket}/bootstrap/vm_setup.sh" \
  -o /usr/local/sbin/mmm-vm-setup
chmod 0755 /usr/local/sbin/mmm-vm-setup

/usr/local/sbin/mmm-vm-setup
echo "=== mmm startup done $(date -u +%FT%TZ) ==="
