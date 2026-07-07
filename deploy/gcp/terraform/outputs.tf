output "app_url" {
  description = "Where seats log in."
  value       = local.site_origin
}

output "lb_ip" {
  description = "Load balancer IP — point the domain's A record here (LB mode only)."
  value       = local.lb_enabled ? google_compute_global_address.lb[0].address : null
}

output "instance" {
  description = "Application VM."
  value       = "${google_compute_instance.app.name} (${var.zone})"
}

output "ssh_command" {
  description = "SSH via IAP (no public port 22)."
  value       = "gcloud compute ssh ${google_compute_instance.app.name} --zone ${var.zone} --project ${var.project_id} --tunnel-through-iap"
}

output "release_bucket" {
  description = "Bucket scripts/package_release.sh uploads releases to."
  value       = google_storage_bucket.releases.name
}

output "kernel_image" {
  description = "Kernel sandbox image the VM pulls. Pin by digest via var.kernel_image once pushed."
  value       = local.kernel_image_ref
}

output "artifact_registry" {
  description = "Docker repo for scripts/build_kernel_image.sh."
  value       = "${local.registry_host}/${var.project_id}/${google_artifact_registry_repository.images.repository_id}"
}

output "bootstrap_login" {
  description = "First-org owner login (password lives only in Secret Manager)."
  value       = var.bootstrap_email
}

output "bootstrap_password_command" {
  description = "Fetch the bootstrap owner's initial password."
  value       = "gcloud secrets versions access latest --secret ${google_secret_manager_secret.bootstrap_password.secret_id} --project ${var.project_id}"
}

output "next_steps" {
  description = "Deployment order after `terraform apply`."
  value       = <<-EOT
    1. ./scripts/build_kernel_image.sh          (build + verify + push the sandbox image)
    2. ./scripts/package_release.sh             (upload source + frontend build)
    3. ./scripts/update_app.sh                  (install the release on the VM)
    4. ${local.lb_enabled ? "point ${var.domain} at the lb_ip output, wait for the managed cert (up to ~30 min)" : "open ${local.site_origin} from an allowed CIDR"}
    5. log in as ${var.bootstrap_email} (password: see bootstrap_password_command)
  EOT
}
