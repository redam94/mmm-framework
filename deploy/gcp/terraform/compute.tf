# The application VM. Single-VM by design: the platform keeps per-session
# kernel state in the API process and its state in SQLite (WAL), so it scales
# vertically (bigger machine_type / more max_kernels), not horizontally — see
# the README's "Scaling" section. deploy/k8s/ holds the future multi-node
# design; the code has no Kubernetes kernel provisioner yet.

locals {
  data_disk_device = "${var.name_prefix}-data"
  site_origin      = local.lb_enabled ? "https://${var.domain}" : "http://${google_compute_address.direct[0].address}"

  extra_env_lines = join("\n", [for k, v in var.extra_env : "${k}=\"${v}\""])
}

# Static external IP for direct mode only (in LB mode the VM is private).
resource "google_compute_address" "direct" {
  count = local.lb_enabled ? 0 : 1

  name   = "${var.name_prefix}-direct-ip"
  region = var.region
}

resource "google_compute_disk" "data" {
  name = "${var.name_prefix}-data"
  type = var.data_disk_type
  zone = var.zone
  size = var.data_disk_size_gb

  # The data disk holds every seat's models, reports and the auth/session DB.
  lifecycle {
    prevent_destroy = true
  }
}

resource "google_compute_resource_policy" "daily_snapshot" {
  count = var.enable_data_disk_snapshots ? 1 : 0

  name   = "${var.name_prefix}-daily-snapshot"
  region = var.region

  snapshot_schedule_policy {
    schedule {
      daily_schedule {
        days_in_cycle = 1
        start_time    = "07:00" # UTC
      }
    }
    retention_policy {
      max_retention_days    = var.snapshot_retention_days
      on_source_disk_delete = "KEEP_AUTO_SNAPSHOTS"
    }
  }
}

resource "google_compute_disk_resource_policy_attachment" "data_snapshots" {
  count = var.enable_data_disk_snapshots ? 1 : 0

  name = google_compute_resource_policy.daily_snapshot[0].name
  disk = google_compute_disk.data.name
  zone = var.zone
}

resource "google_compute_instance" "app" {
  name         = "${var.name_prefix}-app"
  machine_type = var.machine_type
  zone         = var.zone
  tags         = [local.app_tag]

  allow_stopping_for_update = true

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2404-lts-amd64"
      size  = var.boot_disk_size_gb
      type  = "pd-balanced"
    }
  }

  attached_disk {
    source      = google_compute_disk.data.id
    device_name = local.data_disk_device # → /dev/disk/by-id/google-<name>
  }

  network_interface {
    subnetwork = google_compute_subnetwork.app.id

    # External IP only in direct mode; in LB mode all ingress rides the LB and
    # egress rides Cloud NAT.
    dynamic "access_config" {
      for_each = local.lb_enabled ? [] : [1]
      content {
        nat_ip = google_compute_address.direct[0].address
      }
    }
  }

  service_account {
    email  = google_service_account.app.email
    scopes = ["cloud-platform"] # access governed by IAM roles, not scopes
  }

  shielded_instance_config {
    enable_secure_boot          = true
    enable_vtpm                 = true
    enable_integrity_monitoring = true
  }

  metadata = {
    enable-oslogin         = "TRUE"
    block-project-ssh-keys = "TRUE"
  }

  metadata_startup_script = templatefile("${path.module}/templates/startup.sh.tpl", {
    project_id          = var.project_id
    region              = var.region
    release_bucket      = google_storage_bucket.releases.name
    registry_host       = local.registry_host
    kernel_image        = local.kernel_image_ref
    auth_secret_name    = google_secret_manager_secret.auth_secret.secret_id
    bootstrap_pw_secret = google_secret_manager_secret.bootstrap_password.secret_id
    llm_key_secret      = var.llm_api_key != "" ? google_secret_manager_secret.llm_api_key[0].secret_id : ""
    site_origin         = local.site_origin
    llm_provider        = var.llm_provider
    llm_model           = var.llm_model
    llm_location        = var.llm_location
    embed_provider      = var.embed_provider
    embed_model         = var.embed_model
    embed_location      = var.embed_location
    bootstrap_org       = var.bootstrap_org
    bootstrap_email     = var.bootstrap_email
    max_kernels         = var.max_kernels
    kernel_mem          = var.kernel_mem
    kernel_cpus         = var.kernel_cpus
    kernel_pids         = var.kernel_pids
    data_disk_device    = local.data_disk_device
    extra_env_lines     = local.extra_env_lines
  })

  # The startup script downloads bootstrap/vm_setup.sh and reads the secrets,
  # so those must exist before the VM first boots.
  depends_on = [
    google_storage_bucket_object.vm_setup,
    google_secret_manager_secret_version.auth_secret,
    google_secret_manager_secret_version.bootstrap_password,
    google_secret_manager_secret_iam_member.auth_secret_access,
    google_secret_manager_secret_iam_member.bootstrap_password_access,
    google_compute_router_nat.nat,
  ]
}
