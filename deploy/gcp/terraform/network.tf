# VPC, subnet, NAT and firewall. The VM has no external IP in load-balancer
# mode, so Cloud NAT provides egress (apt, uv installer, Artifact Registry,
# Vertex AI). Private Google Access is on as well, so Google APIs stay on
# Google's backbone even without NAT.

locals {
  lb_enabled = var.domain != ""
  app_tag    = "${var.name_prefix}-app"
}

resource "google_compute_network" "vpc" {
  name                    = "${var.name_prefix}-vpc"
  auto_create_subnetworks = false

  depends_on = [google_project_service.required]
}

resource "google_compute_subnetwork" "app" {
  name                     = "${var.name_prefix}-app"
  network                  = google_compute_network.vpc.id
  region                   = var.region
  ip_cidr_range            = "10.10.0.0/24"
  private_ip_google_access = true
}

resource "google_compute_router" "router" {
  name    = "${var.name_prefix}-router"
  network = google_compute_network.vpc.id
  region  = var.region
}

resource "google_compute_router_nat" "nat" {
  name                               = "${var.name_prefix}-nat"
  router                             = google_compute_router.router.name
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"

  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}

# Google Front End + health-check ranges → nginx (:80). Used by the HTTPS load
# balancer for both data plane and health checks.
resource "google_compute_firewall" "allow_lb_and_healthchecks" {
  name    = "${var.name_prefix}-allow-lb-hc"
  network = google_compute_network.vpc.id

  allow {
    protocol = "tcp"
    ports    = ["80"]
  }

  source_ranges = ["130.211.0.0/22", "35.191.0.0/16"]
  target_tags   = [local.app_tag]
}

# SSH only through IAP TCP forwarding — no public port 22.
# Connect with: gcloud compute ssh --tunnel-through-iap <vm>
resource "google_compute_firewall" "allow_iap_ssh" {
  name    = "${var.name_prefix}-allow-iap-ssh"
  network = google_compute_network.vpc.id

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["35.235.240.0/20"]
  target_tags   = [local.app_tag]
}

# Direct mode only (domain = ""): plain HTTP from the operator's CIDRs.
resource "google_compute_firewall" "allow_direct_http" {
  count = local.lb_enabled ? 0 : (length(var.allowed_ingress_cidrs) > 0 ? 1 : 0)

  name    = "${var.name_prefix}-allow-direct-http"
  network = google_compute_network.vpc.id

  allow {
    protocol = "tcp"
    ports    = ["80"]
  }

  source_ranges = var.allowed_ingress_cidrs
  target_tags   = [local.app_tag]
}
