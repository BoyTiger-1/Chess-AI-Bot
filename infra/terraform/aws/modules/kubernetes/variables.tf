variable "cluster_name" {
  type        = string
  description = "EKS cluster name"
}

variable "cluster_version" {
  type        = string
  description = "Kubernetes version"
  default     = "1.30"
}

variable "vpc_id" {
  type        = string
  description = "VPC ID"
}

variable "subnet_ids" {
  type        = list(string)
  description = "Subnet IDs for EKS"
}

variable "node_groups" {
  type = map(any)
  description = "EKS managed node group definitions"
  default = {
    default = {
      desired_size = 2
      min_size     = 2
      max_size     = 10

      instance_types = ["t3.medium"]
      capacity_type  = "ON_DEMAND"
    }
  }
}

variable "tags" {
  type        = map(string)
  description = "Tags to apply"
  default     = {}
}
