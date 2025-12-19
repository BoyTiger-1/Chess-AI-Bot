variable "name" {
  type        = string
  description = "Cloud SQL instance name"
}

variable "region" {
  type        = string
  description = "Region"
}

variable "database_version" {
  type        = string
  description = "Database version"
  default     = "POSTGRES_16"
}

variable "tier" {
  type        = string
  description = "Instance tier"
  default     = "db-custom-2-8192"
}

variable "availability_type" {
  type        = string
  description = "ZONAL or REGIONAL"
  default     = "REGIONAL"
}

variable "private_network" {
  type        = string
  description = "VPC self link for private IP"
}

variable "db_name" {
  type        = string
  description = "Database name"
}

variable "username" {
  type        = string
  description = "DB username"
}

variable "password" {
  type        = string
  description = "DB password"
  sensitive   = true
}
