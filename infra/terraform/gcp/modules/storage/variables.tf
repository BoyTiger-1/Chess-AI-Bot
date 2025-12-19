variable "name" {
  type        = string
  description = "Bucket name"
}

variable "location" {
  type        = string
  description = "Bucket location"
  default     = "US"
}

variable "versioning_enabled" {
  type        = bool
  description = "Enable bucket versioning"
  default     = true
}

variable "kms_key_name" {
  type        = string
  description = "Optional KMS key name"
  default     = null
}

variable "force_destroy" {
  type        = bool
  description = "Force destroy bucket"
  default     = false
}

variable "lifecycle_delete_after_days" {
  type        = number
  description = "Delete objects after N days (0 disables)"
  default     = 0
}

variable "labels" {
  type        = map(string)
  description = "Bucket labels"
  default     = {}
}
