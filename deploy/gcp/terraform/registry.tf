# Artifact Registry repo for the sandbox kernel image (and optionally the API
# image, if you later containerize the API). scripts/build_kernel_image.sh
# builds deploy/kernel/Containerfile and pushes here.

resource "google_artifact_registry_repository" "images" {
  repository_id = var.name_prefix
  location      = var.region
  format        = "DOCKER"
  description   = "MMM Framework images (mmm-kernel sandbox, optionally mmm-api)"

  depends_on = [google_project_service.required]
}

locals {
  registry_host    = "${var.region}-docker.pkg.dev"
  kernel_image_ref = var.kernel_image != "" ? var.kernel_image : "${local.registry_host}/${var.project_id}/${google_artifact_registry_repository.images.repository_id}/mmm-kernel:${var.kernel_image_tag}"
}
