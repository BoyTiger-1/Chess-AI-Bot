variable "bucket_name" {
  type        = string
  description = "S3 bucket name"
}

variable "versioning_enabled" {
  type        = bool
  description = "Enable versioning"
  default     = true
}

variable "tags" {
  type        = map(string)
  description = "Tags to apply"
  default     = {}
}
