terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

module "rds" {
  source  = "terraform-aws-modules/rds/aws"
  version = "6.9.0"

  identifier = var.identifier

  engine               = "postgres"
  engine_version       = var.engine_version
  family               = var.family
  major_engine_version = var.major_engine_version

  instance_class = var.instance_class

  allocated_storage     = var.allocated_storage
  max_allocated_storage = var.max_allocated_storage

  db_name  = var.db_name
  username = var.username
  password = var.password

  port                   = 5432
  multi_az               = var.multi_az
  backup_retention_period = var.backup_retention_days

  subnet_ids               = var.subnet_ids
  vpc_security_group_ids   = var.vpc_security_group_ids
  create_db_subnet_group   = true
  publicly_accessible      = false

  storage_encrypted = true

  tags = var.tags
}
