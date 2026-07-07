# Project services the deployment depends on. `disable_on_destroy = false` so a
# `terraform destroy` of this stack never turns off APIs that other workloads in
# the project may rely on.

locals {
  required_apis = [
    "compute.googleapis.com",          # VM, disks, load balancer
    "iam.googleapis.com",              # service accounts
    "artifactregistry.googleapis.com", # kernel/API container images
    "secretmanager.googleapis.com",    # JWT signing secret, bootstrap password
    "aiplatform.googleapis.com",       # Vertex AI (agent LLM + KB embeddings via ADC)
    "storage.googleapis.com",          # release artifacts bucket
    "logging.googleapis.com",          # Cloud Logging (ops agent)
    "monitoring.googleapis.com",       # Cloud Monitoring (ops agent)
    "iap.googleapis.com",              # IAP TCP forwarding for SSH (no public SSH)
    "osconfig.googleapis.com",         # VM Manager (patch/inventory, optional)
  ]
}

resource "google_project_service" "required" {
  for_each = toset(local.required_apis)

  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}
