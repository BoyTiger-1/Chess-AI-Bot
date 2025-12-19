variable "name" {
  type        = string
  description = "Redis replication group ID"
}

variable "subnet_ids" {
  type        = list(string)
  description = "Subnet IDs for Redis"
}

variable "security_group_ids" {
  type        = list(string)
  description = "Security group IDs for Redis"
  default     = []
}

variable "engine_version" {
  type        = string
  description = "Redis engine version"
  default     = "7.1"
}

variable "node_type" {
  type        = string
  description = "Redis node type"
  default     = "cache.t3.micro"
}

variable "parameter_group_name" {
  type        = string
  description = "Redis parameter group"
  default     = "default.redis7"
}

variable "num_cache_clusters" {
  type        = number
  description = "Number of cache clusters"
  default     = 2
}

variable "automatic_failover_enabled" {
  type        = bool
  description = "Enable automatic failover"
  default     = true
}

variable "multi_az_enabled" {
  type        = bool
  description = "Enable Multi-AZ"
  default     = true
}

variable "tags" {
  type        = map(string)
  description = "Tags to apply"
  default     = {}
}
