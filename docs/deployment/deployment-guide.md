# Deployment Guide

## Prerequisites

### Required Tools
- **AWS CLI**: v2.13+
- **kubectl**: v1.28+
- **Helm**: v3.12+
- **Terraform**: v1.5+
- **Docker**: v24+
- **Git**: v2.40+

### AWS Account Setup
1. AWS account with appropriate permissions
2. IAM user with AdministratorAccess (or specific permissions)
3. AWS CLI configured with credentials
4. S3 bucket for Terraform state
5. DynamoDB table for state locking

### Domain and SSL
1. Registered domain name
2. Route53 hosted zone (or external DNS)
3. ACM SSL certificate (or Let's Encrypt)

## Initial Setup

### 1. Configure AWS CLI

```bash
# Configure AWS credentials
aws configure

# Verify configuration
aws sts get-caller-identity

# Set default region
export AWS_DEFAULT_REGION=us-east-1
```

### 2. Create Terraform State Backend

```bash
# Create S3 bucket for Terraform state
aws s3api create-bucket \
  --bucket chess-ai-terraform-state \
  --region us-east-1

# Enable versioning
aws s3api put-bucket-versioning \
  --bucket chess-ai-terraform-state \
  --versioning-configuration Status=Enabled

# Enable encryption
aws s3api put-bucket-encryption \
  --bucket chess-ai-terraform-state \
  --server-side-encryption-configuration '{
    "Rules": [{
      "ApplyServerSideEncryptionByDefault": {
        "SSEAlgorithm": "AES256"
      }
    }]
  }'

# Create DynamoDB table for state locking
aws dynamodb create-table \
  --table-name terraform-state-lock \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region us-east-1
```

### 3. Request SSL Certificate

```bash
# Request ACM certificate
aws acm request-certificate \
  --domain-name chess-ai.example.com \
  --subject-alternative-names "*.chess-ai.example.com" \
  --validation-method DNS \
  --region us-east-1

# Note the Certificate ARN for later use
```

## Infrastructure Deployment

### 1. Clone Repository

```bash
git clone https://github.com/yourorg/chess-ai-bot.git
cd chess-ai-bot
```

### 2. Configure Terraform Variables

```bash
cd infrastructure/terraform/aws

# Create terraform.tfvars file
cat > terraform.tfvars <<EOF
# Environment
environment = "prod"
aws_region  = "us-east-1"

# Networking
vpc_cidr           = "10.0.0.0/16"
availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]

# Database
db_instance_class    = "db.t3.medium"
db_allocated_storage = 100
db_master_username   = "chessai_admin"

# EKS
eks_node_desired_size = 3
eks_node_min_size     = 2
eks_node_max_size     = 10

# SSL Certificate
ssl_certificate_arn = "arn:aws:acm:us-east-1:123456789012:certificate/xxxxx"

# Monitoring
alarm_email = "devops@yourcompany.com"
EOF

# Create secrets file (DO NOT commit to Git)
cat > terraform.tfvars.secret <<EOF
db_master_password = "$(openssl rand -base64 32)"
EOF
```

### 3. Initialize Terraform

```bash
# Initialize Terraform
terraform init

# Validate configuration
terraform validate

# Plan deployment
terraform plan -var-file=terraform.tfvars.secret

# Review the plan carefully before applying
```

### 4. Deploy Infrastructure

```bash
# Apply Terraform configuration
terraform apply -var-file=terraform.tfvars.secret

# This will create:
# - VPC with public/private subnets
# - EKS cluster and node groups
# - RDS PostgreSQL database
# - ALB and security groups
# - CloudFront distribution
# - S3 buckets
# - Secrets Manager secrets
# - CloudWatch alarms
# - WAF rules
# - Backup plans

# Note the outputs for later use
terraform output > outputs.txt
```

### 5. Configure kubectl

```bash
# Get EKS cluster credentials
aws eks update-kubeconfig \
  --name chess-ai-prod \
  --region us-east-1

# Verify connection
kubectl get nodes

# Expected output: 3 nodes in Ready state
```

## Application Deployment

### 1. Build Docker Image

```bash
# Return to project root
cd ../../..

# Build Docker image
docker build -t chess-ai-bot:1.0.0 .

# Test image locally
docker run -p 8000:8000 chess-ai-bot:1.0.0

# Verify health endpoint
curl http://localhost:8000/health
```

### 2. Push to Container Registry

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  123456789012.dkr.ecr.us-east-1.amazonaws.com

# Create ECR repository
aws ecr create-repository \
  --repository-name chess-ai-bot \
  --region us-east-1

# Tag image
docker tag chess-ai-bot:1.0.0 \
  123456789012.dkr.ecr.us-east-1.amazonaws.com/chess-ai-bot:1.0.0

# Push image
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/chess-ai-bot:1.0.0
```

### 3. Deploy with Helm

```bash
cd infrastructure/helm

# Create namespace
kubectl create namespace chess-ai

# Create secrets
kubectl create secret generic chess-ai-secrets \
  --from-literal=db-password="$(cat ../../terraform/aws/terraform.tfvars.secret | grep db_master_password | cut -d '"' -f 2)" \
  -n chess-ai

# Install cert-manager (for SSL)
helm repo add jetstack https://charts.jetstack.io
helm repo update
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --set installCRDs=true

# Install ingress-nginx
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update
helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --create-namespace \
  --set controller.service.annotations."service\.beta\.kubernetes\.io/aws-load-balancer-type"="nlb"

# Update Helm values
cat > chess-ai/values-prod.yaml <<EOF
image:
  repository: 123456789012.dkr.ecr.us-east-1.amazonaws.com/chess-ai-bot
  tag: "1.0.0"

ingress:
  enabled: true
  hosts:
    - host: chess-ai.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: chess-ai-tls
      hosts:
        - chess-ai.example.com

resources:
  requests:
    cpu: 500m
    memory: 1Gi
  limits:
    cpu: 2000m
    memory: 4Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
EOF

# Install application
helm install chess-ai ./chess-ai \
  -f chess-ai/values-prod.yaml \
  -n chess-ai

# Watch deployment
kubectl get pods -n chess-ai -w
```

### 4. Verify Deployment

```bash
# Check pods
kubectl get pods -n chess-ai

# Check services
kubectl get svc -n chess-ai

# Check ingress
kubectl get ingress -n chess-ai

# Check logs
kubectl logs -f deployment/chess-ai -n chess-ai

# Test health endpoint
kubectl port-forward svc/chess-ai 8000:80 -n chess-ai &
curl http://localhost:8000/health

# Test move endpoint
curl -X POST http://localhost:8000/move \
  -H "Content-Type: application/json" \
  -d '{"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"}'
```

### 5. Configure DNS

```bash
# Get ALB DNS name
ALB_DNS=$(kubectl get ingress chess-ai -n chess-ai -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')

# Create Route53 record (or configure external DNS)
aws route53 change-resource-record-sets \
  --hosted-zone-id Z1234567890ABC \
  --change-batch '{
    "Changes": [{
      "Action": "UPSERT",
      "ResourceRecordSet": {
        "Name": "chess-ai.example.com",
        "Type": "CNAME",
        "TTL": 300,
        "ResourceRecords": [{"Value": "'"$ALB_DNS"'"}]
      }
    }]
  }'
```

## Monitoring Setup

### 1. Deploy Prometheus

```bash
# Add Prometheus Helm repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Prometheus
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set prometheus.prometheusSpec.retention=15d \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=50Gi

# Verify
kubectl get pods -n monitoring
```

### 2. Deploy Grafana

```bash
# Grafana is included in kube-prometheus-stack
# Get Grafana password
kubectl get secret -n monitoring prometheus-grafana \
  -o jsonpath="{.data.admin-password}" | base64 --decode

# Port forward to access Grafana
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80

# Access Grafana at http://localhost:3000
# Username: admin
# Password: (from previous command)
```

### 3. Configure CloudWatch

```bash
# Install CloudWatch agent
kubectl apply -f https://raw.githubusercontent.com/aws-samples/amazon-cloudwatch-container-insights/latest/k8s-deployment-manifest-templates/deployment-mode/daemonset/container-insights-monitoring/quickstart/cwagent-fluentd-quickstart.yaml

# Verify
kubectl get pods -n amazon-cloudwatch
```

## Security Hardening

### 1. Enable Pod Security Standards

```bash
# Label namespace with security policy
kubectl label namespace chess-ai \
  pod-security.kubernetes.io/enforce=restricted \
  pod-security.kubernetes.io/audit=restricted \
  pod-security.kubernetes.io/warn=restricted
```

### 2. Configure Network Policies

```bash
# Apply network policy
kubectl apply -f infrastructure/kubernetes/network-policy.yaml -n chess-ai
```

### 3. Enable Audit Logging

```bash
# Configure EKS audit logging
aws eks update-cluster-config \
  --name chess-ai-prod \
  --logging '{"clusterLogging":[{"types":["api","audit","authenticator","controllerManager","scheduler"],"enabled":true}]}'
```

## Backup and Disaster Recovery

### 1. Configure Automated Backups

```bash
# RDS backups are configured automatically in Terraform
# Verify backup settings
aws rds describe-db-instances \
  --db-instance-identifier chess-ai-prod \
  --query 'DBInstances[0].{BackupRetentionPeriod:BackupRetentionPeriod,PreferredBackupWindow:PreferredBackupWindow}'

# Enable S3 versioning (already done in Terraform)
# Enable S3 cross-region replication for disaster recovery
```

### 2. Test Backup Restore

```bash
# Create manual snapshot
aws rds create-db-snapshot \
  --db-instance-identifier chess-ai-prod \
  --db-snapshot-identifier chess-ai-manual-$(date +%Y%m%d)

# Test restore in different environment
aws rds restore-db-instance-from-db-snapshot \
  --db-instance-identifier chess-ai-test-restore \
  --db-snapshot-identifier chess-ai-manual-$(date +%Y%m%d) \
  --db-instance-class db.t3.small

# Verify and delete test instance
```

## Scaling Configuration

### 1. Configure Cluster Autoscaler

```bash
# Install cluster autoscaler
kubectl apply -f https://raw.githubusercontent.com/kubernetes/autoscaler/master/cluster-autoscaler/cloudprovider/aws/examples/cluster-autoscaler-autodiscover.yaml

# Configure for your cluster
kubectl -n kube-system edit deployment.apps/cluster-autoscaler

# Add cluster name and AWS region
```

### 2. Test Auto-scaling

```bash
# Generate load
kubectl run -it --rm load-generator --image=busybox /bin/sh
while true; do wget -q -O- http://chess-ai.chess-ai.svc.cluster.local/health; done

# Watch HPA
kubectl get hpa -n chess-ai -w

# Watch pods scale
kubectl get pods -n chess-ai -w
```

## Troubleshooting

### Common Issues

#### 1. Pods Not Starting

```bash
# Check pod status
kubectl describe pod <pod-name> -n chess-ai

# Check logs
kubectl logs <pod-name> -n chess-ai

# Common causes:
# - Image pull errors
# - Resource constraints
# - Configuration errors
```

#### 2. Health Checks Failing

```bash
# Test health endpoint
kubectl exec -it <pod-name> -n chess-ai -- curl localhost:8000/health

# Check readiness probe
kubectl describe pod <pod-name> -n chess-ai | grep -A 10 Readiness
```

#### 3. Database Connection Issues

```bash
# Verify RDS endpoint
aws rds describe-db-instances \
  --db-instance-identifier chess-ai-prod \
  --query 'DBInstances[0].Endpoint'

# Test connection from pod
kubectl exec -it <pod-name> -n chess-ai -- /bin/bash
# Inside pod:
# apt-get update && apt-get install -y postgresql-client
# psql -h <rds-endpoint> -U chessai_admin -d chessai
```

## Rollback Procedure

### 1. Helm Rollback

```bash
# List releases
helm list -n chess-ai

# Show history
helm history chess-ai -n chess-ai

# Rollback to previous version
helm rollback chess-ai -n chess-ai

# Rollback to specific revision
helm rollback chess-ai 1 -n chess-ai
```

### 2. Docker Image Rollback

```bash
# Update to previous image
helm upgrade chess-ai ./chess-ai \
  --set image.tag=1.0.0-previous \
  -n chess-ai

# Verify rollback
kubectl get pods -n chess-ai
```

## Maintenance

### Regular Tasks

#### Daily
- Monitor CloudWatch alarms
- Check application logs for errors
- Review Grafana dashboards
- Verify backup completion

#### Weekly
- Review security scan results
- Update dependencies
- Test disaster recovery procedures
- Review cost reports

#### Monthly
- Rotate secrets and credentials
- Update Kubernetes version
- Update node AMIs
- Review and optimize costs
- Conduct security audit

### Update Procedure

```bash
# 1. Update application
docker build -t chess-ai-bot:1.1.0 .
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/chess-ai-bot:1.1.0

# 2. Update Helm chart
helm upgrade chess-ai ./chess-ai \
  --set image.tag=1.1.0 \
  -n chess-ai

# 3. Monitor rollout
kubectl rollout status deployment/chess-ai -n chess-ai

# 4. Verify
curl https://chess-ai.example.com/health
```

## Production Checklist

Before going live, verify:

- [ ] Infrastructure deployed successfully
- [ ] Application pods running (minimum 3)
- [ ] Health checks passing
- [ ] Ingress configured with SSL
- [ ] DNS records pointing to ALB
- [ ] Monitoring and alerting configured
- [ ] Backups tested and verified
- [ ] Security groups properly configured
- [ ] Secrets properly managed
- [ ] Auto-scaling tested
- [ ] Disaster recovery plan documented
- [ ] Runbook created
- [ ] On-call rotation established
- [ ] Documentation updated
- [ ] Performance baseline established
- [ ] Cost alerts configured

## Next Steps

1. Configure custom domain
2. Set up CI/CD pipeline
3. Implement advanced monitoring
4. Configure log aggregation
5. Set up alerting integrations (PagerDuty, Slack)
6. Implement blue-green deployment
7. Configure canary releases
8. Set up performance testing
9. Implement chaos engineering
10. Document incident response procedures
