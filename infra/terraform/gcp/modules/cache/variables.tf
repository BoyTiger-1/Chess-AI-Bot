variable "name" {
  type        = string
  description = "Memorystore instance name"
}

variable "region" {
  type        = string
  description = "Region"
}

variable "tier" {
  type        = string
  description = "BASIC or STANDARD_HA"
  default     = "STANDARD_HA"
}

variable "memory_size_gb" {
  type        = number
  description = "Memory size (GB)"
  default     = 4
}

variable "redis_version" {
  type        = string
  description = "Redis version"
  default     = "REDIS_7_0"
}

variable "authorized_network" {
  type        = string
  description = "Authorized VPC network self link"
}
