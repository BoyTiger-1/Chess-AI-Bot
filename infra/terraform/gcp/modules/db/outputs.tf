output "connection_name" {
  value       = google_sql_database_instance.this.connection_name
  description = "Cloud SQL connection name"
}

output "private_ip_address" {
  value       = google_sql_database_instance.this.private_ip_address
  description = "Private IP address"
}
