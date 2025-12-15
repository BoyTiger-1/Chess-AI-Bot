terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  backend "s3" {
    bucket         = "chess-ai-terraform-state"
    key            = "prod/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-state-lock"
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "Chess-AI-Bot"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# VPC and Networking
module "vpc" {
  source = "../modules/vpc"
  
  vpc_cidr             = var.vpc_cidr
  availability_zones   = var.availability_zones
  environment          = var.environment
  enable_nat_gateway   = true
  enable_vpn_gateway   = false
}

# Security Groups
module "security" {
  source = "../modules/security"
  
  vpc_id      = module.vpc.vpc_id
  environment = var.environment
}

# RDS Database
module "database" {
  source = "../modules/rds"
  
  vpc_id                = module.vpc.vpc_id
  subnet_ids            = module.vpc.private_subnet_ids
  security_group_ids    = [module.security.db_security_group_id]
  
  instance_class        = var.db_instance_class
  allocated_storage     = var.db_allocated_storage
  engine_version        = "15.4"
  database_name         = "chessai"
  master_username       = var.db_master_username
  
  backup_retention_period = 30
  backup_window          = "03:00-04:00"
  maintenance_window     = "mon:04:00-mon:05:00"
  
  multi_az               = var.environment == "prod" ? true : false
  deletion_protection    = var.environment == "prod" ? true : false
  
  environment = var.environment
}

# EKS Cluster
module "eks" {
  source = "../modules/eks"
  
  cluster_name       = "chess-ai-${var.environment}"
  cluster_version    = "1.28"
  vpc_id             = module.vpc.vpc_id
  subnet_ids         = module.vpc.private_subnet_ids
  
  node_groups = {
    general = {
      desired_size   = var.eks_node_desired_size
      min_size       = var.eks_node_min_size
      max_size       = var.eks_node_max_size
      instance_types = ["t3.medium", "t3a.medium"]
      capacity_type  = "SPOT"
    }
    
    compute = {
      desired_size   = 2
      min_size       = 1
      max_size       = 5
      instance_types = ["c5.xlarge", "c5a.xlarge"]
      capacity_type  = "ON_DEMAND"
    }
  }
  
  environment = var.environment
}

# Application Load Balancer
module "alb" {
  source = "../modules/alb"
  
  vpc_id              = module.vpc.vpc_id
  subnet_ids          = module.vpc.public_subnet_ids
  security_group_ids  = [module.security.alb_security_group_id]
  certificate_arn     = var.ssl_certificate_arn
  
  environment = var.environment
}

# CloudFront CDN
module "cdn" {
  source = "../modules/cloudfront"
  
  alb_domain_name = module.alb.dns_name
  environment     = var.environment
}

# S3 Buckets
module "storage" {
  source = "../modules/s3"
  
  environment = var.environment
}

# Secrets Manager
module "secrets" {
  source = "../modules/secrets"
  
  database_credentials = {
    username = var.db_master_username
    password = var.db_master_password
    host     = module.database.endpoint
    port     = module.database.port
    database = "chessai"
  }
  
  environment = var.environment
}

# CloudWatch Monitoring
module "monitoring" {
  source = "../modules/cloudwatch"
  
  eks_cluster_name = module.eks.cluster_name
  alb_arn_suffix   = module.alb.arn_suffix
  rds_instance_id  = module.database.instance_id
  
  alarm_email = var.alarm_email
  environment = var.environment
}

# WAF
module "waf" {
  source = "../modules/waf"
  
  alb_arn     = module.alb.arn
  environment = var.environment
}

# Backup
module "backup" {
  source = "../modules/backup"
  
  rds_instance_arn = module.database.arn
  s3_bucket_arns   = [module.storage.app_bucket_arn]
  
  backup_retention_days = var.environment == "prod" ? 90 : 30
  environment          = var.environment
}
