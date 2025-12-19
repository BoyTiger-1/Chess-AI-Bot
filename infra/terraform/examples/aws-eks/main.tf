terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

module "network" {
  source = "../../aws/modules/network"

  name            = var.name
  cidr            = var.vpc_cidr
  azs             = var.azs
  public_subnets  = var.public_subnets
  private_subnets = var.private_subnets

  single_nat_gateway = var.single_nat_gateway

  tags = var.tags
}

module "eks" {
  source = "../../aws/modules/kubernetes"

  cluster_name    = "${var.name}-eks"
  cluster_version = var.kubernetes_version

  vpc_id     = module.network.vpc_id
  subnet_ids = module.network.private_subnet_ids

  tags = var.tags
}
