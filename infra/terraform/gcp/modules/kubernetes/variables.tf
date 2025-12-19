variable "project_id" {
  type        = string
  description = "GCP project ID"
}

variable "name" {
  type        = string
  description = "GKE cluster name"
}

variable "region" {
  type        = string
  description = "GCP region"
}

variable "network" {
  type        = string
  description = "VPC network name"
}

variable "subnetwork" {
  type        = string
  description = "Subnetwork name"
}

variable "ip_range_pods" {
  type        = string
  description = "Secondary range name for pods"
}

variable "ip_range_services" {
  type        = string
  description = "Secondary range name for services"
}

variable "node_pools" {
  type        = list(any)
  description = "Node pool definitions"
  default = [
    {
      name         = "default"
      machine_type = "e2-standard-2"
      min_count    = 1
      max_count    = 10
      disk_size_gb = 50
    }
  ]
}
