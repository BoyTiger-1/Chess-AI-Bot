# Infrastructure Documentation

## Overview

This directory contains all infrastructure-as-code (IaC), deployment configurations, and operational scripts for the Chess AI Bot application.

## Directory Structure

```
infrastructure/
├── terraform/          # Terraform IaC for cloud resources
│   ├── aws/           # AWS-specific configurations
│   ├── gcp/           # Google Cloud Platform (future)
│   ├── azure/         # Microsoft Azure (future)
│   └── modules/       # Reusable Terraform modules
├── helm/              # Kubernetes Helm charts
│   └── chess-ai/     # Main application chart
├── kubernetes/        # Raw Kubernetes manifests
└── scripts/          # Deployment and operational scripts
    └── deploy.sh     # Main deployment script
```

## Quick Start

### Prerequisites

1. **Required Tools**:
   ```bash
   # Check versions
   aws --version        # v2.13+
   kubectl version      # v1.28+
   helm version         # v3.12+
   terraform version    # v1.5+
   docker --version     # v24+
   ```

2. **AWS Credentials**:
   ```bash
   aws configure
   # Or use AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables
   ```

3. **Domain and SSL**:
   - Registered domain
   - ACM SSL certificate ARN

### Initial Setup

1. **Clone Repository**:
   ```bash
   git clone https://github.com/yourorg/chess-ai-bot.git
   cd chess-ai-bot/infrastructure
   ```

2. **Create Terraform State Backend**:
   ```bash
   # Run once per AWS account
   cd terraform/aws
   ./scripts/create-backend.sh
   ```

3. **Configure Variables**:
   ```bash
   cd terraform/aws
   cp terraform.tfvars.example terraform.tfvars
   # Edit terraform.tfvars with your values
   ```

### Deploy Infrastructure

```bash
# Initialize Terraform
cd terraform/aws
terraform init

# Plan deployment
terraform plan

# Apply infrastructure
terraform apply

# Save outputs
terraform output > outputs.txt
```

### Deploy Application

```bash
# Return to project root
cd ../..

# Run deployment script
./infrastructure/scripts/deploy.sh prod v1.0.0

# Or manually:
# 1. Build and push Docker image
# 2. Update Helm values
# 3. Deploy with Helm
```

## Terraform Modules

### AWS Infrastructure

#### VPC Module
Creates networking infrastructure:
- VPC with public and private subnets
- Internet Gateway for public access
- NAT Gateways for private subnet internet access
- Route tables and associations
- VPC Flow Logs

**Usage**:
```hcl
module "vpc" {
  source = "../modules/vpc"
  
  vpc_cidr           = "10.0.0.0/16"
  availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]
  environment        = "prod"
}
```

#### Security Module
Creates security groups:
- ALB security group (HTTPS/HTTP)
- Application security group
- Database security group
- EKS cluster security groups

**Usage**:
```hcl
module "security" {
  source = "../modules/security"
  
  vpc_id      = module.vpc.vpc_id
  environment = "prod"
}
```

#### RDS Module
Creates PostgreSQL database:
- Multi-AZ for high availability
- Automated backups (30-day retention)
- Encryption at rest
- Performance Insights enabled

**Usage**:
```hcl
module "database" {
  source = "../modules/rds"
  
  vpc_id             = module.vpc.vpc_id
  subnet_ids         = module.vpc.private_subnet_ids
  security_group_ids = [module.security.db_security_group_id]
  instance_class     = "db.t3.medium"
  environment        = "prod"
}
```

#### EKS Module
Creates Kubernetes cluster:
- Managed control plane
- Multiple node groups (general, compute)
- Auto-scaling enabled
- IRSA (IAM Roles for Service Accounts)

**Usage**:
```hcl
module "eks" {
  source = "../modules/eks"
  
  cluster_name = "chess-ai-prod"
  vpc_id       = module.vpc.vpc_id
  subnet_ids   = module.vpc.private_subnet_ids
  environment  = "prod"
}
```

## Helm Charts

### Chess AI Application Chart

**Features**:
- Horizontal Pod Autoscaling (HPA)
- Pod Disruption Budget (PDB)
- Network Policies
- Security Context
- Liveness/Readiness Probes
- Resource Limits
- Ingress with TLS

**Installation**:
```bash
helm install chess-ai ./helm/chess-ai \
  --namespace chess-ai \
  --create-namespace \
  --values helm/chess-ai/values-prod.yaml
```

**Upgrade**:
```bash
helm upgrade chess-ai ./helm/chess-ai \
  --namespace chess-ai \
  --values helm/chess-ai/values-prod.yaml
```

**Rollback**:
```bash
helm rollback chess-ai --namespace chess-ai
```

### Values Files

- `values.yaml`: Default values
- `values-dev.yaml`: Development environment
- `values-staging.yaml`: Staging environment
- `values-prod.yaml`: Production environment

## Deployment Scripts

### deploy.sh

Main deployment script that handles the entire deployment process:

**Usage**:
```bash
./infrastructure/scripts/deploy.sh [environment] [version]

# Examples:
./infrastructure/scripts/deploy.sh dev latest
./infrastructure/scripts/deploy.sh staging v1.2.0
./infrastructure/scripts/deploy.sh prod v1.2.0
```

**What it does**:
1. Validates environment and prerequisites
2. Configures AWS CLI and kubectl
3. Builds Docker image
4. Pushes image to ECR
5. Runs security scan
6. Creates Kubernetes namespace
7. Creates secrets
8. Deploys with Helm
9. Waits for rollout
10. Runs smoke tests
11. Sends notifications

**Environment Variables**:
```bash
AWS_REGION=us-east-1                    # AWS region
SLACK_WEBHOOK_URL=https://...           # Slack notifications
```

## Environments

### Development
- **Purpose**: Local development and testing
- **Infrastructure**: Minimal (1 node, 2 pods)
- **Domain**: chess-ai-dev.example.com
- **Auto-scaling**: Disabled
- **Monitoring**: Basic

### Staging
- **Purpose**: Pre-production testing
- **Infrastructure**: Similar to production
- **Domain**: chess-ai-staging.example.com
- **Auto-scaling**: Enabled
- **Monitoring**: Full

### Production
- **Purpose**: Live production environment
- **Infrastructure**: Full HA setup
- **Domain**: chess-ai.example.com
- **Auto-scaling**: Enabled
- **Monitoring**: Full with alerting

## Common Operations

### Scale Application

```bash
# Manual scaling
kubectl scale deployment/chess-ai --replicas=10 -n chess-ai

# Adjust HPA
kubectl patch hpa chess-ai -n chess-ai \
  -p '{"spec":{"minReplicas":5,"maxReplicas":30}}'
```

### Update Application

```bash
# Build new version
docker build -t chess-ai:v1.1.0 .

# Deploy new version
./infrastructure/scripts/deploy.sh prod v1.1.0
```

### View Logs

```bash
# Application logs
kubectl logs -f deployment/chess-ai -n chess-ai

# All pods
kubectl logs -f -l app.kubernetes.io/name=chess-ai -n chess-ai

# Specific pod
kubectl logs -f chess-ai-7d9f8c4b5-abc12 -n chess-ai
```

### Execute Commands in Pod

```bash
# Get shell
kubectl exec -it deployment/chess-ai -n chess-ai -- /bin/bash

# Run command
kubectl exec deployment/chess-ai -n chess-ai -- python --version
```

### Check Status

```bash
# Pods
kubectl get pods -n chess-ai

# Deployment
kubectl get deployment chess-ai -n chess-ai

# HPA
kubectl get hpa -n chess-ai

# Ingress
kubectl get ingress -n chess-ai
```

### Restart Pods

```bash
# Rolling restart
kubectl rollout restart deployment/chess-ai -n chess-ai

# Watch rollout
kubectl rollout status deployment/chess-ai -n chess-ai
```

## Monitoring

### Access Grafana

```bash
# Port forward
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80

# Open browser
open http://localhost:3000
```

### Access Prometheus

```bash
# Port forward
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090

# Open browser
open http://localhost:9090
```

### CloudWatch Dashboards

```bash
# Open AWS Console
open https://console.aws.amazon.com/cloudwatch

# Navigate to:
# - Dashboards → Chess-AI-Overview
# - Alarms → All alarms
# - Logs → /aws/eks/chess-ai-prod
```

## Troubleshooting

### Pods Not Starting

```bash
# Check pod status
kubectl describe pod <pod-name> -n chess-ai

# Check events
kubectl get events -n chess-ai --sort-by='.lastTimestamp'

# Check logs
kubectl logs <pod-name> -n chess-ai --previous
```

### Cannot Connect to EKS

```bash
# Update kubeconfig
aws eks update-kubeconfig --name chess-ai-prod --region us-east-1

# Verify
kubectl get nodes
```

### Terraform Errors

```bash
# Refresh state
terraform refresh

# Import existing resources
terraform import aws_instance.example i-1234567890abcdef0

# Force unlock (if state locked)
terraform force-unlock <lock-id>
```

### Image Pull Errors

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Verify image exists
aws ecr describe-images --repository-name chess-ai-bot
```

## Security

### Secrets Management

All secrets are stored in AWS Secrets Manager and synced to Kubernetes:

```bash
# View secrets
kubectl get secrets -n chess-ai

# Describe secret
kubectl describe secret chess-ai-secrets -n chess-ai

# Update secret
kubectl create secret generic chess-ai-secrets \
  --from-literal=db-password=<new-password> \
  --namespace chess-ai \
  --dry-run=client -o yaml | kubectl apply -f -
```

### Certificate Management

Certificates are managed by cert-manager:

```bash
# Check certificate status
kubectl get certificate -n chess-ai

# Describe certificate
kubectl describe certificate chess-ai-tls -n chess-ai

# Force renewal
kubectl delete certificate chess-ai-tls -n chess-ai
# Certificate will be recreated automatically
```

### Security Scanning

```bash
# Scan Docker image
trivy image chess-ai:latest

# Scan Kubernetes manifests
trivy config ./helm/chess-ai/

# Scan Terraform code
tfsec terraform/aws/
```

## Backup and Recovery

### Create Backup

```bash
# RDS snapshot
aws rds create-db-snapshot \
  --db-instance-identifier chess-ai-prod \
  --db-snapshot-identifier chess-ai-manual-$(date +%Y%m%d)

# Helm values backup
helm get values chess-ai -n chess-ai > helm-values-backup.yaml

# Kubernetes resources backup
kubectl get all -n chess-ai -o yaml > k8s-backup.yaml
```

### Restore from Backup

```bash
# Restore RDS from snapshot
aws rds restore-db-instance-from-db-snapshot \
  --db-instance-identifier chess-ai-restored \
  --db-snapshot-identifier chess-ai-manual-20240115

# Restore Helm deployment
helm install chess-ai ./helm/chess-ai \
  --namespace chess-ai \
  --values helm-values-backup.yaml
```

## Cost Management

### View Current Costs

```bash
# EKS cluster
aws eks describe-cluster --name chess-ai-prod

# EC2 instances
aws ec2 describe-instances --filters "Name=tag:Cluster,Values=chess-ai-prod"

# RDS
aws rds describe-db-instances --db-instance-identifier chess-ai-prod
```

### Optimize Costs

1. **Use Spot Instances**: 70% savings on compute
2. **Reserved Instances**: 40% savings on baseline capacity
3. **Right-size Resources**: Monitor and adjust based on usage
4. **S3 Lifecycle Policies**: Archive old data to Glacier
5. **CloudWatch Log Retention**: Reduce retention period

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    tags:
      - 'v*'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      
      - name: Deploy
        run: |
          ./infrastructure/scripts/deploy.sh prod ${{ github.ref_name }}
```

## Additional Resources

- [Deployment Guide](../docs/deployment/deployment-guide.md)
- [Architecture Documentation](../docs/architecture/system-architecture.md)
- [Security Hardening](../docs/security/security-hardening.md)
- [Operational Runbook](../docs/troubleshooting/runbook.md)
- [API Documentation](../docs/api/api-documentation.md)

## Support

- **Documentation**: https://docs.chess-ai.example.com
- **Issues**: https://github.com/yourorg/chess-ai-bot/issues
- **Slack**: #chess-ai-infrastructure
- **Email**: devops@chess-ai.example.com
