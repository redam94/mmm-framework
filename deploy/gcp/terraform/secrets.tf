# Secrets live in Secret Manager and are fetched by the VM at boot into
# /etc/mmm/mmm.env (root:mmm, 0640). They are never placed in instance
# metadata, and the kernel env-scrub (agents/kernels.py denylist) keeps them
# out of every sandbox container.

# HS256 signing key for the built-in JWT auth (MMM_AUTH_SECRET). Rotating it
# (taint random_password.auth_secret + re-apply + reboot VM) invalidates every
# outstanding token — all seats must log in again.
resource "random_password" "auth_secret" {
  length  = 64
  special = false
}

resource "google_secret_manager_secret" "auth_secret" {
  secret_id = "${var.name_prefix}-auth-secret"

  replication {
    auto {}
  }

  depends_on = [google_project_service.required]
}

resource "google_secret_manager_secret_version" "auth_secret" {
  secret      = google_secret_manager_secret.auth_secret.id
  secret_data = random_password.auth_secret.result
}

# Initial password for the bootstrap org owner (MMM_AUTH_BOOTSTRAP_PASSWORD).
# Read it with the command in the `bootstrap_password_command` output; rotate
# it after first login via POST /auth/password-reset.
resource "random_password" "bootstrap_password" {
  length           = 24
  special          = true
  override_special = "-_.@"
}

resource "google_secret_manager_secret" "bootstrap_password" {
  secret_id = "${var.name_prefix}-bootstrap-password"

  replication {
    auto {}
  }

  depends_on = [google_project_service.required]
}

resource "google_secret_manager_secret_version" "bootstrap_password" {
  secret      = google_secret_manager_secret.bootstrap_password.id
  secret_data = random_password.bootstrap_password.result
}

# Optional API key for direct (non-Vertex) LLM providers.
resource "google_secret_manager_secret" "llm_api_key" {
  count = var.llm_api_key != "" ? 1 : 0

  secret_id = "${var.name_prefix}-llm-api-key"

  replication {
    auto {}
  }

  depends_on = [google_project_service.required]
}

resource "google_secret_manager_secret_version" "llm_api_key" {
  count = var.llm_api_key != "" ? 1 : 0

  secret      = google_secret_manager_secret.llm_api_key[0].id
  secret_data = var.llm_api_key
}

# Per-secret access for the VM service account.
resource "google_secret_manager_secret_iam_member" "auth_secret_access" {
  secret_id = google_secret_manager_secret.auth_secret.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.app.email}"
}

resource "google_secret_manager_secret_iam_member" "bootstrap_password_access" {
  secret_id = google_secret_manager_secret.bootstrap_password.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.app.email}"
}

resource "google_secret_manager_secret_iam_member" "llm_api_key_access" {
  count = var.llm_api_key != "" ? 1 : 0

  secret_id = google_secret_manager_secret.llm_api_key[0].id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.app.email}"
}
