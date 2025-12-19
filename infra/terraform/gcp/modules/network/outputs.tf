output "network_name" {
  value       = module.vpc.network_name
  description = "VPC name"
}

output "subnet_self_links" {
  value       = module.vpc.subnets_self_links
  description = "Subnet self links"
}
