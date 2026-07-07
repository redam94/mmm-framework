# Terraform + provider pins for the MMM Framework GCP deployment.
#
# State: local by default. For a team deployment, uncomment the GCS backend and
# point it at a bucket you create out-of-band (state buckets should not manage
# themselves):
#
#   terraform {
#     backend "gcs" {
#       bucket = "YOUR_TF_STATE_BUCKET"
#       prefix = "mmm-framework"
#     }
#   }

terraform {
  required_version = ">= 1.7"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 6.8.0, < 8.0.0"
    }
    random = {
      source  = "hashicorp/random"
      version = ">= 3.6.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}
