# Production Security Checklist

## Pre-Deployment Security Checklist

### Infrastructure Security

#### Network Security
- [ ] VPC configured with public and private subnets
- [ ] NAT gateway configured for private subnet internet access
- [ ] Security groups follow least privilege principle
- [ ] Network ACLs configured for additional security layer
- [ ] VPC Flow Logs enabled for network monitoring
- [ ] No direct internet access to database or application servers
- [ ] Bastion host configured for secure SSH access (if needed)
- [ ] VPN or AWS Systems Manager Session Manager for admin access

#### Firewall and WAF
- [ ] AWS WAF enabled on Application Load Balancer
- [ ] Rate limiting rules configured (100 req/5min per IP)
- [ ] SQL injection protection enabled
- [ ] XSS protection enabled
- [ ] Known bad inputs blocked
- [ ] Geographic restrictions configured (if applicable)
- [ ] Bot detection and mitigation enabled
- [ ] DDoS protection enabled (AWS Shield)

#### Load Balancer Security
- [ ] ALB access logs enabled and stored in S3
- [ ] HTTPS listener configured (no HTTP except redirect)
- [ ] SSL/TLS certificate from ACM or valid CA
- [ ] TLS 1.2+ only (no TLS 1.0/1.1)
- [ ] Strong cipher suites configured
- [ ] HTTP to HTTPS redirect enabled
- [ ] Security headers configured (HSTS, etc.)

### Application Security

#### Container Security
- [ ] Docker image built from official base images
- [ ] Multi-stage build used to minimize image size
- [ ] No secrets in Docker image or layers
- [ ] Image scanned for vulnerabilities (Trivy, ECR scanning)
- [ ] No critical vulnerabilities in production images
- [ ] Container runs as non-root user (UID 1000)
- [ ] Read-only root filesystem where possible
- [ ] Minimal packages installed in container
- [ ] Image signed and verified

#### Kubernetes Security
- [ ] Pod Security Standards enforced (restricted)
- [ ] Security context configured (non-root, no privilege escalation)
- [ ] Resource limits set for all containers
- [ ] Network policies configured
- [ ] RBAC roles follow least privilege
- [ ] Service accounts created per application
- [ ] Secrets stored in Kubernetes Secrets (not ConfigMaps)
- [ ] Image pull secrets configured
- [ ] Pod security policies applied
- [ ] Admission controllers enabled

#### Application Code Security
- [ ] Input validation on all endpoints
- [ ] Output encoding to prevent XSS
- [ ] SQL injection protection (parameterized queries)
- [ ] CSRF protection enabled
- [ ] Rate limiting implemented
- [ ] Authentication required for protected endpoints
- [ ] Authorization checks implemented
- [ ] Secure session management
- [ ] Proper error handling (no sensitive info in errors)
- [ ] Security headers set (CSP, X-Frame-Options, etc.)

### Data Security

#### Encryption at Rest
- [ ] RDS encryption enabled with KMS
- [ ] EBS volumes encrypted
- [ ] S3 buckets encrypted (AES-256 or KMS)
- [ ] Backup snapshots encrypted
- [ ] Secrets Manager secrets encrypted with KMS
- [ ] KMS key rotation enabled

#### Encryption in Transit
- [ ] HTTPS enforced for all external connections
- [ ] TLS 1.2+ required
- [ ] Database connections use SSL/TLS
- [ ] Internal service communication encrypted
- [ ] Certificate management automated (cert-manager)

#### Key Management
- [ ] AWS KMS used for encryption keys
- [ ] Separate KMS keys for different data types
- [ ] Key policies follow least privilege
- [ ] Key rotation enabled (annual)
- [ ] Key usage audited via CloudTrail
- [ ] Key backup and recovery procedures documented

### Access Control

#### IAM Security
- [ ] IAM users follow least privilege principle
- [ ] MFA enabled for all IAM users
- [ ] Root account MFA enabled
- [ ] Root account access keys deleted
- [ ] IAM password policy enforced (complexity, rotation)
- [ ] IAM roles used instead of long-term credentials
- [ ] Cross-account access uses roles (not keys)
- [ ] Service-linked roles used for AWS services
- [ ] IAM Access Analyzer enabled

#### Secrets Management
- [ ] All secrets stored in AWS Secrets Manager
- [ ] No secrets in code, configs, or environment variables
- [ ] Secrets rotation enabled (90 days)
- [ ] Secrets accessed via IAM roles (not keys)
- [ ] Secrets audit logging enabled
- [ ] Backup secrets stored securely offline
- [ ] Emergency access procedures documented

#### Database Access
- [ ] Database master password stored in Secrets Manager
- [ ] Database accessible only from application security group
- [ ] SSL/TLS required for database connections
- [ ] Database audit logging enabled
- [ ] Principle of least privilege for database users
- [ ] No default/weak database passwords
- [ ] Database backup encryption enabled

### Monitoring and Logging

#### Audit Logging
- [ ] CloudTrail enabled in all regions
- [ ] CloudTrail logs encrypted
- [ ] CloudTrail log integrity validation enabled
- [ ] CloudTrail logs sent to S3 bucket with versioning
- [ ] CloudTrail logs analysis via CloudWatch Insights
- [ ] VPC Flow Logs enabled
- [ ] ALB access logs enabled
- [ ] RDS audit logs enabled
- [ ] Kubernetes audit logs enabled

#### Security Monitoring
- [ ] AWS GuardDuty enabled
- [ ] AWS Security Hub enabled
- [ ] AWS Config enabled for compliance monitoring
- [ ] CloudWatch alarms for security events
- [ ] Prometheus alerts for application security events
- [ ] Failed login attempts monitored
- [ ] Unusual API activity monitored
- [ ] Security group changes monitored

#### Incident Response
- [ ] Incident response plan documented
- [ ] Security incident runbook created
- [ ] On-call rotation established
- [ ] PagerDuty integration configured
- [ ] Incident communication plan documented
- [ ] Post-mortem template created
- [ ] Backup and recovery procedures tested

### Compliance

#### GDPR Compliance
- [ ] Data minimization implemented
- [ ] User consent management implemented
- [ ] Right to access implemented
- [ ] Right to deletion implemented
- [ ] Data portability implemented
- [ ] Privacy policy published
- [ ] Data processing agreement signed
- [ ] Breach notification procedure documented

#### SOC 2 Compliance
- [ ] Security controls documented
- [ ] Access control matrix created
- [ ] Change management process documented
- [ ] Incident response process documented
- [ ] Backup and recovery tested
- [ ] Security awareness training completed
- [ ] Vendor security assessments completed

#### CCPA Compliance
- [ ] Data inventory completed
- [ ] Consumer rights procedures implemented
- [ ] Privacy policy updated
- [ ] Data sale opt-out implemented (if applicable)

### Backup and Disaster Recovery

#### Backup Strategy
- [ ] RDS automated backups enabled (30 days retention)
- [ ] RDS manual snapshots taken before major changes
- [ ] S3 versioning enabled
- [ ] S3 cross-region replication enabled (production)
- [ ] Backup testing performed monthly
- [ ] Backup restoration documented
- [ ] Backup encryption enabled

#### Disaster Recovery
- [ ] RTO defined (15 minutes)
- [ ] RPO defined (5 minutes)
- [ ] Disaster recovery plan documented
- [ ] Disaster recovery tested quarterly
- [ ] Multi-AZ deployment configured
- [ ] Cross-region backup configured
- [ ] Failover procedures documented

### Vulnerability Management

#### Scanning
- [ ] Container images scanned on push (ECR)
- [ ] Dependency scanning configured (Dependabot)
- [ ] SAST scanning configured (Bandit, SonarQube)
- [ ] DAST scanning performed (OWASP ZAP)
- [ ] Infrastructure scanning (AWS Config, Security Hub)
- [ ] Vulnerability scan results reviewed weekly
- [ ] Critical vulnerabilities patched within 24 hours

#### Patch Management
- [ ] OS patches applied monthly
- [ ] Application dependencies updated weekly
- [ ] Kubernetes version kept current (n-1)
- [ ] Database patches applied during maintenance windows
- [ ] Emergency patch process documented

### Security Testing

#### Pre-Production Testing
- [ ] Security unit tests written
- [ ] Integration tests include security scenarios
- [ ] Penetration testing performed
- [ ] Security code review completed
- [ ] Threat modeling completed
- [ ] Security acceptance criteria defined

#### Penetration Testing
- [ ] Annual penetration test scheduled
- [ ] Scope defined (all production endpoints)
- [ ] Rules of engagement documented
- [ ] Findings documented and tracked
- [ ] Remediation plan created
- [ ] Retest after remediation

## Post-Deployment Verification

### Immediate Checks (Day 1)
- [ ] All pods running and healthy
- [ ] Health checks passing
- [ ] HTTPS working correctly
- [ ] Certificate valid and trusted
- [ ] WAF rules active
- [ ] Monitoring alerts working
- [ ] Logs flowing to CloudWatch
- [ ] Secrets accessible to application
- [ ] Database connections working
- [ ] Backup jobs running

### Week 1 Checks
- [ ] No critical security alerts
- [ ] GuardDuty findings reviewed
- [ ] Security Hub score reviewed
- [ ] CloudTrail logs analyzed
- [ ] Access logs analyzed for anomalies
- [ ] Performance within SLA
- [ ] No unauthorized access detected

### Month 1 Checks
- [ ] Security scan results reviewed
- [ ] Penetration test scheduled
- [ ] Compliance audit scheduled
- [ ] Security training completed
- [ ] Incident response drill conducted
- [ ] Disaster recovery test performed

## Continuous Security

### Daily
- [ ] Review GuardDuty findings
- [ ] Review CloudWatch security alarms
- [ ] Check for failed authentication attempts
- [ ] Review application error logs

### Weekly
- [ ] Review Security Hub findings
- [ ] Analyze access patterns
- [ ] Review vulnerability scan results
- [ ] Update security patches
- [ ] Review IAM access patterns

### Monthly
- [ ] Security team meeting
- [ ] Review and update security policies
- [ ] Rotate non-automated secrets
- [ ] Test backup restoration
- [ ] Review compliance status
- [ ] Update security documentation

### Quarterly
- [ ] Penetration testing
- [ ] Disaster recovery drill
- [ ] Security awareness training
- [ ] Review and update incident response plan
- [ ] Vendor security review
- [ ] Compliance audit

### Annually
- [ ] Full security audit
- [ ] Threat modeling update
- [ ] Review and update security architecture
- [ ] Review and renew certifications
- [ ] Review and update DRP
- [ ] External security assessment

## Security Contacts

### Internal Team
- Security Lead: john@example.com
- DevOps Lead: jane@example.com
- On-Call Engineer: PagerDuty

### External Resources
- AWS Support: https://console.aws.amazon.com/support
- AWS Security: aws-security@amazon.com
- CERT: https://www.us-cert.gov

## Security Tools

### Required Tools
- [ ] AWS CLI configured
- [ ] kubectl configured with RBAC
- [ ] AWS IAM Authenticator
- [ ] Security scanning tools installed
- [ ] Secrets management tools configured

### Recommended Tools
- [ ] OWASP ZAP for DAST
- [ ] Trivy for container scanning
- [ ] Bandit for Python SAST
- [ ] Safety for Python dependency checking
- [ ] git-secrets to prevent committing secrets

## Security Training

### Required Training
- [ ] OWASP Top 10
- [ ] AWS Security Best Practices
- [ ] Kubernetes Security
- [ ] Secure Coding Practices
- [ ] Incident Response
- [ ] GDPR/CCPA Compliance

### Recommended Training
- [ ] AWS Security Specialty Certification
- [ ] Certified Kubernetes Security Specialist (CKS)
- [ ] CISSP or equivalent
- [ ] Security awareness training

## Emergency Procedures

### Security Incident Response
1. **Detect**: Alert triggered or issue reported
2. **Contain**: Isolate affected resources
3. **Investigate**: Determine scope and impact
4. **Eradicate**: Remove threat and vulnerabilities
5. **Recover**: Restore service securely
6. **Document**: Create post-mortem

### Emergency Contacts
- Critical security incidents: Page security team via PagerDuty
- AWS security issues: Contact AWS Support (Premium)
- Legal/compliance issues: Contact legal team

### Breach Notification
- Internal: Notify within 1 hour
- External (GDPR): Notify within 72 hours
- Affected users: Notify as soon as possible

## Approval

### Sign-off Required From:
- [ ] Security Team Lead
- [ ] DevOps Team Lead
- [ ] Engineering Manager
- [ ] CTO/CISO

### Date: _______________

### Approved By: _______________

---

**Note**: This checklist should be reviewed and updated quarterly or whenever significant changes are made to the infrastructure or application.
