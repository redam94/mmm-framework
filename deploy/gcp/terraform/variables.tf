variable "project_id" {
  description = "GCP project ID to deploy into."
  type        = string
}

variable "region" {
  description = "Region for the VM, Artifact Registry, NAT and release bucket."
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "Zone for the application VM (must be inside var.region)."
  type        = string
  default     = "us-central1-a"
}

variable "name_prefix" {
  description = "Prefix for every resource name (lets several stacks share a project)."
  type        = string
  default     = "mmm"
}

# ── Exposure ───────────────────────────────────────────────────────────────────

variable "domain" {
  description = <<-EOT
    Public domain for the deployment (e.g. augur.example.com). When set, a
    global HTTPS load balancer with a Google-managed certificate is created and
    the VM gets NO external IP. When empty, the stack falls back to "direct
    mode": the VM gets a static external IP serving plain HTTP on port 80,
    reachable only from allowed_ingress_cidrs — for evaluation only, never for
    real client data.
  EOT
  type        = string
  default     = ""
}

variable "dns_managed_zone" {
  description = "Optional Cloud DNS managed-zone NAME in this project. When set together with domain, an A record for the domain is created pointing at the load balancer IP. Leave empty to manage DNS elsewhere."
  type        = string
  default     = ""
}

variable "allowed_ingress_cidrs" {
  description = "CIDRs allowed to reach the VM directly in direct mode (domain = \"\"). Ignored in load-balancer mode."
  type        = list(string)
  default     = []
}

# ── Compute sizing ─────────────────────────────────────────────────────────────

variable "machine_type" {
  description = "VM machine type. Bayesian fits are CPU/RAM hungry: each sandboxed kernel is capped at kernel_mem/kernel_cpus and NUTS runs 4 chains, so size for max_kernels * kernel caps plus ~4 GB for the API + OS."
  type        = string
  default     = "n2-standard-8" # 8 vCPU / 32 GB
}

variable "boot_disk_size_gb" {
  description = "Boot disk size (OS + app releases + venv + container images)."
  type        = number
  default     = 80
}

variable "data_disk_size_gb" {
  description = "Persistent data disk (/data): sessions DB, agent workspace (fitted models, reports, KB), audit log."
  type        = number
  default     = 200
}

variable "data_disk_type" {
  description = "Type of the data disk."
  type        = string
  default     = "pd-balanced"
}

variable "enable_data_disk_snapshots" {
  description = "Attach a daily snapshot schedule to the data disk."
  type        = bool
  default     = true
}

variable "snapshot_retention_days" {
  description = "How many days of daily data-disk snapshots to keep."
  type        = number
  default     = 14
}

# ── Kernel sandbox ─────────────────────────────────────────────────────────────

variable "kernel_image" {
  description = <<-EOT
    Full image reference for the per-session sandbox kernel. Leave empty to use
    the Artifact Registry repo this stack creates, tagged with
    kernel_image_tag. For production, pin by DIGEST once pushed, e.g.
    "us-central1-docker.pkg.dev/PROJECT/mmm/mmm-kernel@sha256:...".
    scripts/build_kernel_image.sh prints the digest after pushing.
  EOT
  type        = string
  default     = ""
}

variable "kernel_image_tag" {
  description = "Tag used when kernel_image is empty."
  type        = string
  default     = "latest"
}

variable "max_kernels" {
  description = "Max live per-session kernels before LRU eviction (MMM_MAX_KERNELS). Each kernel can hold kernel_mem of RAM — keep max_kernels * kernel_mem under the VM's memory."
  type        = number
  default     = 6
}

variable "kernel_mem" {
  description = "Per-kernel memory cap (MMM_KERNEL_MEM)."
  type        = string
  default     = "2g"
}

variable "kernel_cpus" {
  description = "Per-kernel CPU cap (MMM_KERNEL_CPUS). NUTS samples 4 chains in parallel; this is a ceiling, not a reservation."
  type        = string
  default     = "4"
}

variable "kernel_pids" {
  description = "Per-kernel process cap (MMM_KERNEL_PIDS)."
  type        = string
  default     = "512"
}

# ── Agent LLM (Vertex AI by default — uses the VM service account's ADC) ──────

variable "llm_provider" {
  description = "Agent LLM provider: vertex_anthropic / vertex_gemini use the VM service account via ADC (no API key). anthropic / openai / google_genai need llm_api_key."
  type        = string
  default     = "vertex_anthropic"
}

variable "llm_model" {
  description = "Model ID. For Vertex, use the exact Model Garden ID (may carry an @version suffix), e.g. claude-sonnet-4-5@20250929."
  type        = string
  default     = "claude-sonnet-4-5@20250929"
}

variable "llm_location" {
  description = "Vertex region serving the chat model. Anthropic Claude models are served from specific regions (e.g. us-east5)."
  type        = string
  default     = "us-east5"
}

variable "llm_api_key" {
  description = "API key for DIRECT (non-Vertex) providers. Stored in Secret Manager, injected only into the API process (the kernel env-scrub strips it). Leave empty for Vertex."
  type        = string
  default     = ""
  sensitive   = true
}

variable "embed_provider" {
  description = "Knowledge-base embeddings provider. 'vertex' rides the same ADC as the chat model (Anthropic has no embeddings API)."
  type        = string
  default     = "vertex"
}

variable "embed_model" {
  description = "Embeddings model."
  type        = string
  default     = "text-embedding-005"
}

variable "embed_location" {
  description = "Vertex region for the embeddings model."
  type        = string
  default     = "us-central1"
}

# ── Multiseat bootstrap (first org + owner) ───────────────────────────────────

variable "bootstrap_org" {
  description = "Name of the first organization, created on API startup (idempotent)."
  type        = string
  default     = "Augur"
}

variable "bootstrap_email" {
  description = "Email of the first org owner. The generated password lands in Secret Manager (see outputs)."
  type        = string
}

# ── Optional extras ────────────────────────────────────────────────────────────

variable "extra_env" {
  description = "Extra environment variables appended verbatim to the API env file (e.g. { MMM_CONNECTION_SYNC_INTERVAL = \"3600\" }). Do NOT put secrets here — they would land in the instance metadata."
  type        = map(string)
  default     = {}
}
