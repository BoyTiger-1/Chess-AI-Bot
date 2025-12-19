variable "log_group_name" {
  type        = string
  description = "CloudWatch log group name"
}

variable "retention_in_days" {
  type        = number
  description = "Log retention in days"
  default     = 30
}

variable "kms_key_id" {
  type        = string
  description = "Optional KMS key for encryption"
  default     = null
}

variable "tags" {
  type        = map(string)
  description = "Tags to apply"
  default     = {}
}
