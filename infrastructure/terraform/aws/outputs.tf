output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "eks_cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "eks_cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "rds_endpoint" {
  description = "RDS endpoint"
  value       = module.database.endpoint
  sensitive   = true
}

output "alb_dns_name" {
  description = "ALB DNS name"
  value       = module.alb.dns_name
}

output "cloudfront_domain" {
  description = "CloudFront distribution domain"
  value       = module.cdn.domain_name
}

output "s3_app_bucket" {
  description = "S3 application bucket name"
  value       = module.storage.app_bucket_name
}

output "secrets_manager_arn" {
  description = "Secrets Manager ARN for database credentials"
  value       = module.secrets.db_secret_arn
  sensitive   = true
}
