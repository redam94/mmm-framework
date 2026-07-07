# The VM's service account is the ONLY cloud identity in the deployment.
# Kernel containers run with `--network none` and a scrubbed env, so this
# identity never reaches model-authored code (agents/container_kernel.py +
# agents/kernels.py denylist).

resource "google_service_account" "app" {
  account_id   = "${var.name_prefix}-app"
  display_name = "MMM Framework application VM"

  depends_on = [google_project_service.required]
}

# Vertex AI: agent LLM (vertex_anthropic / vertex_gemini) + KB embeddings via ADC.
resource "google_project_iam_member" "vertex_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.app.email}"
}

# Ops agent → Cloud Logging / Monitoring.
resource "google_project_iam_member" "log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.app.email}"
}

resource "google_project_iam_member" "metric_writer" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.app.email}"
}

# Pull the kernel image from this stack's Artifact Registry repo (repo-scoped,
# not project-wide).
resource "google_artifact_registry_repository_iam_member" "kernel_pull" {
  repository = google_artifact_registry_repository.images.name
  location   = var.region
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${google_service_account.app.email}"
}

# Download release artifacts (source + frontend bundles) from the release bucket.
resource "google_storage_bucket_iam_member" "release_reader" {
  bucket = google_storage_bucket.releases.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.app.email}"
}

# Secret access is granted per-secret in secrets.tf — no project-wide
# secretAccessor.
