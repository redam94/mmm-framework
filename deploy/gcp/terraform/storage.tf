# Release bucket: scripts/package_release.sh uploads
#   releases/<version>/source.tar.gz        (git archive of the repo)
#   releases/<version>/frontend-dist.tar.gz (built React app)
#   releases/CURRENT                        (text pointer to the active version)
# and Terraform seeds bootstrap/vm_setup.sh, which the VM's thin startup
# script downloads and runs on every boot.

resource "google_storage_bucket" "releases" {
  name                        = "${var.project_id}-${var.name_prefix}-releases"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = false

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      num_newer_versions = 10
    }
    action {
      type = "Delete"
    }
  }

  depends_on = [google_project_service.required]
}

resource "google_storage_bucket_object" "vm_setup" {
  bucket = google_storage_bucket.releases.name
  name   = "bootstrap/vm_setup.sh"
  source = "${path.module}/../vm/vm_setup.sh"
}
