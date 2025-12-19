output "bucket_id" {
  value       = google_logging_project_bucket_config.this.bucket_id
  description = "Log bucket id"
}
