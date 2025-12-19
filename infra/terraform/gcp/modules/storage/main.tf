terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.0"
    }
  }
}

resource "google_storage_bucket" "this" {
  name          = var.name
  location      = var.location
  force_destroy = var.force_destroy

  uniform_bucket_level_access = true

  versioning {
    enabled = var.versioning_enabled
  }

  encryption {
    default_kms_key_name = var.kms_key_name
  }

  dynamic "lifecycle_rule" {
    for_each = var.lifecycle_delete_after_days > 0 ? [1] : []
    content {
      condition {
        age = var.lifecycle_delete_after_days
      }
      action {
        type = "Delete"
      }
    }
  }

  labels = var.labels
}
