terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.0"
    }
  }
}

module "vpc" {
  source  = "terraform-google-modules/network/google"
  version = "9.1.0"

  project_id   = var.project_id
  network_name = var.network_name

  subnets = var.subnets
}
