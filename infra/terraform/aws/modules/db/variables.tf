variable "identifier" {
  type        = string
  description = "RDS instance identifier"
}

variable "db_name" {
  type        = string
  description = "Database name"
}

variable "username" {
  type        = string
  description = "Master username"
}

variable "password" {
  type        = string
  description = "Master password"
  sensitive   = true
}

variable "subnet_ids" {
  type        = list(string)
  description = "Subnet IDs for the DB subnet group"
}

variable "vpc_security_group_ids" {
  type        = list(string)
  description = "Security group IDs to attach to RDS"
  default     = []
}

variable "instance_class" {
  type        = string
  description = "RDS instance class"
  default     = "db.t3.medium"
}

variable "allocated_storage" {
  type        = number
  description = "Initial storage (GiB)"
  default     = 50
}

variable "max_allocated_storage" {
  type        = number
  description = "Autoscaling max storage (GiB)"
  default     = 200
}

variable "multi_az" {
  type        = bool
  description = "Enable Multi-AZ"
  default     = true
}

variable "backup_retention_days" {
  type        = number
  description = "Backup retention days"
  default     = 7
}

variable "engine_version" {
  type        = string
  description = "Postgres engine version"
  default     = "16.3"
}

variable "family" {
  type        = string
  description = "Parameter group family"
  default     = "postgres16"
}

variable "major_engine_version" {
  type        = string
  description = "Major engine version"
  default     = "16"
}

variable "tags" {
  type        = map(string)
  description = "Tags to apply"
  default     = {}
}
