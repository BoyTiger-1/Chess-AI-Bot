# Security Hardening Guide

## Overview

This document outlines security best practices and hardening procedures for the Chess AI Bot application in production environments.

## Security Principles

1. **Defense in Depth**: Multiple layers of security controls
2. **Least Privilege**: Minimum necessary permissions
3. **Zero Trust**: Verify everything, trust nothing
4. **Security by Design**: Built-in from the start
5. **Continuous Monitoring**: Real-time threat detection

## Network Security

### VPC Configuration

```hcl
# Terraform configuration
vpc_cidr = "10.0.0.0/16"

# Subnet design
- Public subnets:  10.0.0.0/24, 10.0.1.0/24, 10.0.2.0/24
- Private subnets: 10.0.10.0/24, 10.0.11.0/24, 10.0.12.0/24
- Database subnets: 10.0.20.0/24, 10.0.21.0/24, 10.0.22.0/24
```

**Security Controls:**
- ✅ Private subnets for application and database
- ✅ NAT gateway for outbound internet access
- ✅ Internet gateway only for public subnets
- ✅ No direct internet access to private resources
- ✅ VPC Flow Logs enabled for traffic monitoring

### Security Groups

#### ALB Security Group

```hcl
# Ingress rules
- HTTPS (443) from 0.0.0.0/0
- HTTP (80) from 0.0.0.0/0 (redirect to HTTPS)

# Egress rules
- All traffic to application security group
```

#### Application Security Group

```hcl
# Ingress rules
- Port 8000 from ALB security group only

# Egress rules
- Port 5432 to database security group
- Port 443 to 0.0.0.0/0 (for external APIs)
```

#### Database Security Group

```hcl
# Ingress rules
- Port 5432 from application security group only

# Egress rules
- None (database doesn't initiate connections)
```

### Network ACLs

```bash
# Deny known malicious IP ranges
aws ec2 create-network-acl-entry \
  --network-acl-id acl-xxxxx \
  --ingress \
  --rule-number 1 \
  --protocol -1 \
  --cidr-block 203.0.113.0/24 \
  --rule-action deny
```

### VPC Endpoints

```hcl
# Use VPC endpoints to avoid internet gateway
- S3 endpoint
- DynamoDB endpoint
- Secrets Manager endpoint
- ECR endpoint
```

## Application Security

### Container Security

#### Dockerfile Best Practices

```dockerfile
# ✅ Use official base images
FROM python:3.12-slim

# ✅ Run as non-root user
RUN useradd -m -u 1000 appuser
USER appuser

# ✅ Use multi-stage builds
FROM python:3.12-slim as builder
# Build dependencies
FROM python:3.12-slim
COPY --from=builder /app /app

# ✅ Scan for vulnerabilities
# Run: docker scan chess-ai-bot:latest
```

#### Kubernetes Security Context

```yaml
securityContext:
  # Run as non-root
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000
  
  # Drop all capabilities
  capabilities:
    drop:
      - ALL
  
  # No privilege escalation
  allowPrivilegeEscalation: false
  
  # Read-only root filesystem
  readOnlyRootFilesystem: true
  
  # Seccomp profile
  seccompProfile:
    type: RuntimeDefault
```

### Input Validation

```python
# Validate FEN positions
import chess

def validate_fen(fen: str) -> bool:
    """Validate FEN string format and legality."""
    try:
        board = chess.Board(fen)
        return board.is_valid()
    except ValueError:
        return False

# Rate limiting
from flask_limiter import Limiter

limiter = Limiter(
    app,
    default_limits=["100 per hour", "20 per minute"]
)

@app.route('/move', methods=['POST'])
@limiter.limit("10 per minute")
def move():
    data = request.json
    
    # Validate input
    if not data or 'fen' not in data:
        return jsonify({'error': 'Invalid request'}), 400
    
    if not validate_fen(data['fen']):
        return jsonify({'error': 'Invalid FEN'}), 400
    
    # Process request
    ...
```

### HTTPS/TLS Configuration

```nginx
# Nginx Ingress TLS configuration
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384';
ssl_prefer_server_ciphers on;
ssl_session_cache shared:SSL:10m;
ssl_session_timeout 10m;

# HSTS header
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
```

### Security Headers

```python
from flask import Flask
from flask_talisman import Talisman

app = Flask(__name__)

# Configure security headers
Talisman(app,
    force_https=True,
    strict_transport_security=True,
    content_security_policy={
        'default-src': "'self'",
        'script-src': ["'self'", "'unsafe-inline'", "cdn.jsdelivr.net"],
        'style-src': ["'self'", "'unsafe-inline'", "cdn.jsdelivr.net"],
        'img-src': ["'self'", "data:", "https:"],
        'font-src': ["'self'", "cdn.jsdelivr.net"],
    },
    content_security_policy_nonce_in=['script-src'],
    frame_options='DENY',
    referrer_policy='strict-origin-when-cross-origin',
    x_content_type_options=True,
    x_xss_protection=True
)
```

## Data Security

### Encryption at Rest

#### RDS Encryption

```hcl
resource "aws_db_instance" "main" {
  # Enable encryption at rest
  storage_encrypted = true
  kms_key_id       = aws_kms_key.rds.arn
}

resource "aws_kms_key" "rds" {
  description             = "KMS key for RDS encryption"
  deletion_window_in_days = 30
  enable_key_rotation     = true
}
```

#### S3 Encryption

```hcl
resource "aws_s3_bucket" "app" {
  bucket = "chess-ai-app-${var.environment}"
}

resource "aws_s3_bucket_server_side_encryption_configuration" "app" {
  bucket = aws_s3_bucket.app.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.s3.arn
    }
  }
}
```

#### EBS Encryption

```hcl
# Enable EBS encryption by default
resource "aws_ebs_encryption_by_default" "main" {
  enabled = true
}
```

### Encryption in Transit

#### Database Connections

```python
# Require SSL for database connections
import psycopg2

conn = psycopg2.connect(
    host=os.environ['DB_HOST'],
    database=os.environ['DB_NAME'],
    user=os.environ['DB_USER'],
    password=os.environ['DB_PASSWORD'],
    sslmode='require',
    sslrootcert='/path/to/rds-ca-cert.pem'
)
```

#### API Calls

```python
import requests

# Always use HTTPS
response = requests.get(
    'https://api.example.com/data',
    verify=True,  # Verify SSL certificate
    timeout=10
)
```

### Key Management

```hcl
# AWS KMS key for secrets
resource "aws_kms_key" "secrets" {
  description             = "KMS key for secrets encryption"
  deletion_window_in_days = 30
  enable_key_rotation     = true
  
  tags = {
    Name = "chess-ai-secrets-key"
  }
}

# Key policy
resource "aws_kms_key_policy" "secrets" {
  key_id = aws_kms_key.secrets.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM policies"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      },
      {
        Sid    = "Allow EKS to decrypt"
        Effect = "Allow"
        Principal = {
          AWS = aws_iam_role.eks_node.arn
        }
        Action = [
          "kms:Decrypt",
          "kms:DescribeKey"
        ]
        Resource = "*"
      }
    ]
  })
}
```

## Secrets Management

### AWS Secrets Manager

```bash
# Create secret
aws secretsmanager create-secret \
  --name chess-ai/prod/database \
  --secret-string '{
    "username": "chessai_admin",
    "password": "GENERATED_PASSWORD",
    "engine": "postgres",
    "host": "chess-ai-prod.xxxxx.us-east-1.rds.amazonaws.com",
    "port": 5432,
    "dbname": "chessai"
  }' \
  --kms-key-id alias/chess-ai-secrets

# Enable automatic rotation
aws secretsmanager rotate-secret \
  --secret-id chess-ai/prod/database \
  --rotation-lambda-arn arn:aws:lambda:us-east-1:123456789012:function:SecretsManagerRotation \
  --rotation-rules AutomaticallyAfterDays=90
```

### Kubernetes Secrets

```bash
# Create secret from AWS Secrets Manager
kubectl create secret generic database-credentials \
  --from-literal=username=$(aws secretsmanager get-secret-value --secret-id chess-ai/prod/database --query SecretString --output text | jq -r .username) \
  --from-literal=password=$(aws secretsmanager get-secret-value --secret-id chess-ai/prod/database --query SecretString --output text | jq -r .password) \
  -n chess-ai

# Use External Secrets Operator (recommended)
kubectl apply -f https://raw.githubusercontent.com/external-secrets/external-secrets/main/deploy/crds/bundle.yaml

# Create ExternalSecret
cat <<EOF | kubectl apply -f -
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: database-credentials
  namespace: chess-ai
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets-manager
    kind: SecretStore
  target:
    name: database-credentials
  data:
  - secretKey: username
    remoteRef:
      key: chess-ai/prod/database
      property: username
  - secretKey: password
    remoteRef:
      key: chess-ai/prod/database
      property: password
EOF
```

### Environment Variables

```yaml
# Never hardcode secrets
# ❌ BAD
env:
  - name: DB_PASSWORD
    value: "mysecretpassword"

# ✅ GOOD
env:
  - name: DB_PASSWORD
    valueFrom:
      secretKeyRef:
        name: database-credentials
        key: password
```

## Access Control

### IAM Policies

```hcl
# EKS node role policy (least privilege)
resource "aws_iam_role_policy" "eks_node" {
  name = "chess-ai-eks-node-policy"
  role = aws_iam_role.eks_node.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = "arn:aws:secretsmanager:*:*:secret:chess-ai/*"
      },
      {
        Effect = "Allow"
        Action = [
          "kms:Decrypt"
        ]
        Resource = aws_kms_key.secrets.arn
      }
    ]
  })
}
```

### Kubernetes RBAC

```yaml
# Create service account
apiVersion: v1
kind: ServiceAccount
metadata:
  name: chess-ai
  namespace: chess-ai
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::123456789012:role/chess-ai-pod-role

---
# Role with minimal permissions
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: chess-ai
  namespace: chess-ai
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list"]

---
# Bind role to service account
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: chess-ai
  namespace: chess-ai
subjects:
- kind: ServiceAccount
  name: chess-ai
roleRef:
  kind: Role
  name: chess-ai
  apiGroup: rbac.authorization.k8s.io
```

### MFA Enforcement

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DenyAllExceptListedIfNoMFA",
      "Effect": "Deny",
      "NotAction": [
        "iam:CreateVirtualMFADevice",
        "iam:EnableMFADevice",
        "iam:GetUser",
        "iam:ListMFADevices",
        "iam:ListVirtualMFADevices",
        "iam:ResyncMFADevice",
        "sts:GetSessionToken"
      ],
      "Resource": "*",
      "Condition": {
        "BoolIfExists": {
          "aws:MultiFactorAuthPresent": "false"
        }
      }
    }
  ]
}
```

## Web Application Firewall (WAF)

### AWS WAF Configuration

```hcl
resource "aws_wafv2_web_acl" "main" {
  name  = "chess-ai-waf-${var.environment}"
  scope = "REGIONAL"
  
  default_action {
    allow {}
  }
  
  # Rate limiting rule
  rule {
    name     = "RateLimitRule"
    priority = 1
    
    action {
      block {}
    }
    
    statement {
      rate_based_statement {
        limit              = 2000
        aggregate_key_type = "IP"
      }
    }
    
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name               = "RateLimitRule"
      sampled_requests_enabled  = true
    }
  }
  
  # AWS Managed Rules
  rule {
    name     = "AWSManagedRulesCommonRuleSet"
    priority = 2
    
    override_action {
      none {}
    }
    
    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesCommonRuleSet"
        vendor_name = "AWS"
      }
    }
    
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name               = "AWSManagedRulesCommonRuleSet"
      sampled_requests_enabled  = true
    }
  }
  
  # SQL injection protection
  rule {
    name     = "SQLiProtection"
    priority = 3
    
    action {
      block {}
    }
    
    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesSQLiRuleSet"
        vendor_name = "AWS"
      }
    }
    
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name               = "SQLiProtection"
      sampled_requests_enabled  = true
    }
  }
  
  # Geographic blocking (optional)
  rule {
    name     = "GeoBlocking"
    priority = 4
    
    action {
      block {}
    }
    
    statement {
      geo_match_statement {
        country_codes = ["CN", "RU"]  # Example: block China and Russia
      }
    }
    
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name               = "GeoBlocking"
      sampled_requests_enabled  = true
    }
  }
  
  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name               = "chess-ai-waf"
    sampled_requests_enabled  = true
  }
}
```

## Vulnerability Management

### Container Image Scanning

```bash
# Scan with Trivy
trivy image chess-ai-bot:latest

# Scan with AWS ECR
aws ecr start-image-scan \
  --repository-name chess-ai-bot \
  --image-id imageTag=latest

# Get scan results
aws ecr describe-image-scan-findings \
  --repository-name chess-ai-bot \
  --image-id imageTag=latest
```

### Dependency Scanning

```bash
# Python dependencies
pip install safety
safety check --json

# Or use pip-audit
pip install pip-audit
pip-audit

# Automated with GitHub Dependabot
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "daily"
    open-pull-requests-limit: 10
```

### SAST (Static Application Security Testing)

```bash
# Use Bandit for Python
pip install bandit
bandit -r . -f json -o bandit-report.json

# Use SonarQube
sonar-scanner \
  -Dsonar.projectKey=chess-ai-bot \
  -Dsonar.sources=. \
  -Dsonar.host.url=http://sonarqube.example.com \
  -Dsonar.login=YOUR_TOKEN
```

### DAST (Dynamic Application Security Testing)

```bash
# Use OWASP ZAP
docker run -t owasp/zap2docker-stable zap-baseline.py \
  -t https://chess-ai.example.com \
  -r zap-report.html

# Use Nikto
nikto -h https://chess-ai.example.com -o nikto-report.html
```

## Compliance

### GDPR Compliance

**Requirements:**
1. **Data Minimization**: Collect only necessary data
2. **Right to Access**: Users can request their data
3. **Right to Deletion**: Users can delete their data
4. **Data Portability**: Export data in machine-readable format
5. **Consent Management**: Explicit consent for data processing
6. **Data Breach Notification**: Report within 72 hours

**Implementation:**
```python
# Data anonymization
def anonymize_user_data(user_id):
    """Anonymize user data for GDPR compliance."""
    # Replace PII with hashed values
    hashed_id = hashlib.sha256(user_id.encode()).hexdigest()
    
    # Update database
    db.execute(
        "UPDATE users SET email = %s, name = %s WHERE id = %s",
        (hashed_id, 'Anonymous', user_id)
    )

# Data export
def export_user_data(user_id):
    """Export user data in JSON format."""
    data = db.execute(
        "SELECT * FROM users WHERE id = %s",
        (user_id,)
    ).fetchone()
    
    return json.dumps(data, indent=2)
```

### SOC 2 Compliance

**Type II Controls:**
1. **Security**: Access controls, encryption, monitoring
2. **Availability**: Uptime SLA, redundancy, disaster recovery
3. **Processing Integrity**: Data validation, error handling
4. **Confidentiality**: Encryption, access controls
5. **Privacy**: Data handling, consent management

**Evidence Collection:**
- Access logs (CloudTrail, K8s audit logs)
- Change management records (Git, Terraform)
- Monitoring and alerting (CloudWatch, Prometheus)
- Incident response procedures
- Security training records

## Security Monitoring

### CloudTrail

```hcl
resource "aws_cloudtrail" "main" {
  name                          = "chess-ai-trail"
  s3_bucket_name                = aws_s3_bucket.cloudtrail.id
  include_global_service_events = true
  is_multi_region_trail         = true
  enable_log_file_validation    = true
  
  event_selector {
    read_write_type           = "All"
    include_management_events = true
    
    data_resource {
      type   = "AWS::S3::Object"
      values = ["${aws_s3_bucket.app.arn}/"]
    }
  }
}
```

### GuardDuty

```bash
# Enable GuardDuty
aws guardduty create-detector \
  --enable \
  --finding-publishing-frequency FIFTEEN_MINUTES

# Get findings
aws guardduty list-findings \
  --detector-id <detector-id> \
  --finding-criteria '{"Criterion":{"severity":{"Gte":7}}}'
```

### Security Hub

```bash
# Enable Security Hub
aws securityhub enable-security-hub

# Enable standards
aws securityhub batch-enable-standards \
  --standards-subscription-requests \
  StandardsArn=arn:aws:securityhub:::ruleset/cis-aws-foundations-benchmark/v/1.2.0

# Get findings
aws securityhub get-findings \
  --filters '{"SeverityLabel":[{"Value":"CRITICAL","Comparison":"EQUALS"}]}'
```

## Incident Response

### Security Incident Playbook

1. **Detection**
   - Monitor alerts from GuardDuty, Security Hub
   - Review CloudWatch alarms
   - Check WAF metrics

2. **Containment**
   - Isolate affected resources
   - Block malicious IPs in WAF
   - Rotate compromised credentials

3. **Eradication**
   - Remove malware/backdoors
   - Patch vulnerabilities
   - Update security groups

4. **Recovery**
   - Restore from clean backups
   - Verify system integrity
   - Monitor for reinfection

5. **Lessons Learned**
   - Document incident timeline
   - Update runbooks
   - Implement preventive controls

### Emergency Contacts

```yaml
security_contacts:
  - role: Security Lead
    name: John Doe
    email: john@example.com
    phone: +1-555-0100
  
  - role: DevOps Lead
    name: Jane Smith
    email: jane@example.com
    phone: +1-555-0101
  
  - role: On-Call Engineer
    pagerduty: chess-ai-oncall
```

## Security Checklist

### Pre-Deployment
- [ ] All secrets stored in Secrets Manager
- [ ] Database encryption enabled
- [ ] S3 bucket encryption enabled
- [ ] VPC Flow Logs enabled
- [ ] CloudTrail enabled
- [ ] GuardDuty enabled
- [ ] Security Hub enabled
- [ ] WAF rules configured
- [ ] Security groups follow least privilege
- [ ] IAM roles follow least privilege
- [ ] MFA enforced for admins
- [ ] Container images scanned
- [ ] Dependencies up to date
- [ ] HTTPS enforced
- [ ] Security headers configured

### Post-Deployment
- [ ] Penetration testing completed
- [ ] Vulnerability scanning completed
- [ ] Access logs reviewed
- [ ] Monitoring alerts tested
- [ ] Incident response plan documented
- [ ] Backup restore tested
- [ ] Security training completed
- [ ] Compliance audit completed

## Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [AWS Security Best Practices](https://docs.aws.amazon.com/security/)
- [CIS Kubernetes Benchmark](https://www.cisecurity.org/benchmark/kubernetes)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
