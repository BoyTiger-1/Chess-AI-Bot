output "endpoint" {
  value       = module.rds.db_instance_endpoint
  description = "RDS endpoint"
}

output "port" {
  value       = module.rds.db_instance_port
  description = "RDS port"
}
