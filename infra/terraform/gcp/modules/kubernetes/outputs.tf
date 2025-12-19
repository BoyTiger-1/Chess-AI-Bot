output "name" {
  value       = module.gke.name
  description = "Cluster name"
}

output "endpoint" {
  value       = module.gke.endpoint
  description = "Cluster endpoint"
}
