terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.0"
    }
  }
}

resource "google_logging_project_bucket_config" "this" {
  project        = var.project_id
  location       = var.location
  bucket_id      = var.bucket_id
  retention_days = var.retention_days
}
