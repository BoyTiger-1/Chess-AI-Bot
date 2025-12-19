variable "project_id" {
  type        = string
  description = "Project ID"
}

variable "location" {
  type        = string
  description = "Bucket location"
  default     = "global"
}

variable "bucket_id" {
  type        = string
  description = "Logging bucket id"
  default     = "business-ai-logs"
}

variable "retention_days" {
  type        = number
  description = "Log retention"
  default     = 30
}
