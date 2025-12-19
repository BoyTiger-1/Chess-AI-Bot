output "host" {
  value       = google_redis_instance.this.host
  description = "Redis host"
}

output "port" {
  value       = google_redis_instance.this.port
  description = "Redis port"
}
