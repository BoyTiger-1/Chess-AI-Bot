terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.0"
    }
  }
}

module "gke" {
  source  = "terraform-google-modules/kubernetes-engine/google"
  version = "34.0.0"

  project_id = var.project_id
  name       = var.name
  region     = var.region

  network    = var.network
  subnetwork = var.subnetwork

  ip_range_pods     = var.ip_range_pods
  ip_range_services = var.ip_range_services

  remove_default_node_pool = true

  node_pools = var.node_pools
}
