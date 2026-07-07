# Global external HTTPS load balancer (created only when var.domain is set):
# managed certificate, HTTP→HTTPS redirect, and a 1-hour backend timeout so
# the /chat SSE stream and long-running fits survive the proxy.

resource "google_compute_global_address" "lb" {
  count = local.lb_enabled ? 1 : 0

  name = "${var.name_prefix}-lb-ip"

  depends_on = [google_project_service.required]
}

resource "google_compute_instance_group" "app" {
  count = local.lb_enabled ? 1 : 0

  name      = "${var.name_prefix}-app-ig"
  zone      = var.zone
  network   = google_compute_network.vpc.id
  instances = [google_compute_instance.app.self_link]

  named_port {
    name = "http"
    port = 80
  }
}

resource "google_compute_health_check" "app" {
  count = local.lb_enabled ? 1 : 0

  name                = "${var.name_prefix}-hc"
  check_interval_sec  = 15
  timeout_sec         = 10
  healthy_threshold   = 2
  unhealthy_threshold = 3

  http_health_check {
    port         = 80
    request_path = "/api/health" # through nginx → proves nginx AND the API are up
  }
}

resource "google_compute_backend_service" "app" {
  count = local.lb_enabled ? 1 : 0

  name                  = "${var.name_prefix}-backend"
  protocol              = "HTTP"
  port_name             = "http"
  load_balancing_scheme = "EXTERNAL_MANAGED"
  timeout_sec           = 3600 # SSE chat streams + long fits
  health_checks         = [google_compute_health_check.app[0].id]

  backend {
    group           = google_compute_instance_group.app[0].id
    balancing_mode  = "UTILIZATION"
    max_utilization = 1.0
  }

  log_config {
    enable      = true
    sample_rate = 1.0
  }
}

resource "google_compute_managed_ssl_certificate" "app" {
  count = local.lb_enabled ? 1 : 0

  # Name derives from the domain: changing the domain replaces the cert.
  name = "${var.name_prefix}-cert-${replace(var.domain, ".", "-")}"

  managed {
    domains = [var.domain]
  }
}

resource "google_compute_url_map" "https" {
  count = local.lb_enabled ? 1 : 0

  name            = "${var.name_prefix}-https"
  default_service = google_compute_backend_service.app[0].id
}

resource "google_compute_target_https_proxy" "app" {
  count = local.lb_enabled ? 1 : 0

  name             = "${var.name_prefix}-https-proxy"
  url_map          = google_compute_url_map.https[0].id
  ssl_certificates = [google_compute_managed_ssl_certificate.app[0].id]
}

resource "google_compute_global_forwarding_rule" "https" {
  count = local.lb_enabled ? 1 : 0

  name                  = "${var.name_prefix}-https-fr"
  load_balancing_scheme = "EXTERNAL_MANAGED"
  ip_address            = google_compute_global_address.lb[0].address
  port_range            = "443"
  target                = google_compute_target_https_proxy.app[0].id
}

# Port 80 → 301 to HTTPS.
resource "google_compute_url_map" "redirect" {
  count = local.lb_enabled ? 1 : 0

  name = "${var.name_prefix}-http-redirect"

  default_url_redirect {
    https_redirect         = true
    redirect_response_code = "MOVED_PERMANENTLY_DEFAULT"
    strip_query            = false
  }
}

resource "google_compute_target_http_proxy" "redirect" {
  count = local.lb_enabled ? 1 : 0

  name    = "${var.name_prefix}-http-proxy"
  url_map = google_compute_url_map.redirect[0].id
}

resource "google_compute_global_forwarding_rule" "http" {
  count = local.lb_enabled ? 1 : 0

  name                  = "${var.name_prefix}-http-fr"
  load_balancing_scheme = "EXTERNAL_MANAGED"
  ip_address            = google_compute_global_address.lb[0].address
  port_range            = "80"
  target                = google_compute_target_http_proxy.redirect[0].id
}

# Optional Cloud DNS A record (when the zone lives in this project).
resource "google_dns_record_set" "app" {
  count = local.lb_enabled && var.dns_managed_zone != "" ? 1 : 0

  managed_zone = var.dns_managed_zone
  name         = "${var.domain}."
  type         = "A"
  ttl          = 300
  rrdatas      = [google_compute_global_address.lb[0].address]
}
