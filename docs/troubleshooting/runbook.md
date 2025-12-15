# Operational Runbook

## Quick Reference

### Emergency Contacts

| Role | Contact | Phone | Email |
|------|---------|-------|-------|
| On-Call Engineer | PagerDuty | +1-555-0100 | oncall@example.com |
| DevOps Lead | Jane Smith | +1-555-0101 | jane@example.com |
| Security Lead | John Doe | +1-555-0102 | john@example.com |
| CTO | Alex Johnson | +1-555-0103 | alex@example.com |

### Critical Links

- **Status Page**: https://status.chess-ai.example.com
- **Grafana Dashboards**: https://grafana.chess-ai.example.com
- **CloudWatch**: https://console.aws.amazon.com/cloudwatch
- **PagerDuty**: https://yourorg.pagerduty.com
- **Slack Channel**: #chess-ai-alerts

## Common Issues and Solutions

### 1. High Error Rate (5xx Errors)

**Symptoms:**
- Increased 500/503 errors in logs
- ALB target health checks failing
- Grafana dashboard shows error spike

**Diagnosis:**

```bash
# Check pod status
kubectl get pods -n chess-ai

# Check pod logs
kubectl logs -f deployment/chess-ai -n chess-ai --tail=100

# Check pod resources
kubectl top pods -n chess-ai

# Check events
kubectl get events -n chess-ai --sort-by='.lastTimestamp'
```

**Common Causes:**

1. **Out of Memory (OOM)**
   ```bash
   # Check memory usage
   kubectl top pods -n chess-ai
   
   # Check for OOM kills
   kubectl describe pod <pod-name> -n chess-ai | grep -i oom
   ```
   
   **Solution:**
   ```bash
   # Increase memory limits
   helm upgrade chess-ai ./helm/chess-ai \
     --set resources.limits.memory=8Gi \
     -n chess-ai
   ```

2. **Move Database Not Loaded**
   ```bash
   # Check readiness probe
   kubectl exec -it <pod-name> -n chess-ai -- curl localhost:8000/health/ready
   ```
   
   **Solution:**
   ```bash
   # Restart pods to reload database
   kubectl rollout restart deployment/chess-ai -n chess-ai
   ```

3. **Database Connection Issues**
   ```bash
   # Test database connectivity
   kubectl exec -it <pod-name> -n chess-ai -- nc -zv <rds-endpoint> 5432
   ```
   
   **Solution:**
   ```bash
   # Verify security groups allow traffic
   aws ec2 describe-security-groups \
     --group-ids <sg-id> \
     --query 'SecurityGroups[0].IpPermissions'
   ```

### 2. High Latency

**Symptoms:**
- P95 latency > 500ms
- CloudWatch ALB TargetResponseTime alarm
- Slow user-reported performance

**Diagnosis:**

```bash
# Check pod CPU usage
kubectl top pods -n chess-ai

# Check ALB metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/ApplicationELB \
  --metric-name TargetResponseTime \
  --dimensions Name=LoadBalancer,Value=<alb-arn> \
  --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 300 \
  --statistics Average

# Check application metrics
curl https://chess-ai.example.com/metrics
```

**Solutions:**

1. **Scale Up Pods**
   ```bash
   # Manual scaling
   kubectl scale deployment/chess-ai --replicas=10 -n chess-ai
   
   # Or adjust HPA
   kubectl patch hpa chess-ai -n chess-ai -p '{"spec":{"minReplicas":5}}'
   ```

2. **Increase CPU Limits**
   ```bash
   helm upgrade chess-ai ./helm/chess-ai \
     --set resources.limits.cpu=4000m \
     -n chess-ai
   ```

3. **Add Node Group**
   ```bash
   # Scale EKS node group
   aws eks update-nodegroup-config \
     --cluster-name chess-ai-prod \
     --nodegroup-name general \
     --scaling-config minSize=3,maxSize=20,desiredSize=6
   ```

### 3. Pod Crashes / CrashLoopBackOff

**Symptoms:**
- Pods in CrashLoopBackOff state
- Constant pod restarts
- Application unavailable

**Diagnosis:**

```bash
# Check pod status
kubectl get pods -n chess-ai

# Check recent logs
kubectl logs <pod-name> -n chess-ai --previous

# Describe pod
kubectl describe pod <pod-name> -n chess-ai

# Check events
kubectl get events -n chess-ai | grep -i error
```

**Common Causes:**

1. **Configuration Error**
   ```bash
   # Check ConfigMap
   kubectl get configmap -n chess-ai
   kubectl describe configmap chess-ai -n chess-ai
   
   # Check Secrets
   kubectl get secrets -n chess-ai
   ```
   
   **Solution:**
   ```bash
   # Fix ConfigMap/Secret
   kubectl edit configmap chess-ai -n chess-ai
   kubectl rollout restart deployment/chess-ai -n chess-ai
   ```

2. **Image Pull Error**
   ```bash
   # Check image pull secrets
   kubectl get pods -n chess-ai -o jsonpath='{.items[*].spec.imagePullSecrets}'
   
   # Verify ECR permissions
   aws ecr get-login-password | docker login --username AWS --password-stdin <ecr-url>
   ```

3. **Application Error**
   ```bash
   # Check application logs
   kubectl logs <pod-name> -n chess-ai | grep -i error
   
   # Test locally
   docker run -p 8000:8000 <image>
   curl localhost:8000/health
   ```

### 4. Database Connection Failures

**Symptoms:**
- Database connection timeouts
- "Could not connect to database" errors
- Readiness probe failures

**Diagnosis:**

```bash
# Check RDS instance status
aws rds describe-db-instances \
  --db-instance-identifier chess-ai-prod \
  --query 'DBInstances[0].DBInstanceStatus'

# Test connectivity from pod
kubectl exec -it <pod-name> -n chess-ai -- /bin/bash
# Inside pod:
nc -zv <rds-endpoint> 5432
```

**Solutions:**

1. **RDS Instance Down**
   ```bash
   # Check RDS events
   aws rds describe-events \
     --source-identifier chess-ai-prod \
     --source-type db-instance
   
   # Restart RDS (if needed)
   aws rds reboot-db-instance \
     --db-instance-identifier chess-ai-prod
   ```

2. **Security Group Issue**
   ```bash
   # Verify security group rules
   aws ec2 describe-security-groups \
     --group-ids <db-sg-id>
   
   # Add rule if missing
   aws ec2 authorize-security-group-ingress \
     --group-id <db-sg-id> \
     --protocol tcp \
     --port 5432 \
     --source-group <app-sg-id>
   ```

3. **Connection Pool Exhausted**
   ```bash
   # Check active connections
   kubectl exec -it <pod-name> -n chess-ai -- psql -h <endpoint> -U admin -d chessai -c \
     "SELECT count(*) FROM pg_stat_activity;"
   
   # Increase max connections (RDS parameter group)
   aws rds modify-db-parameter-group \
     --db-parameter-group-name chess-ai-params \
     --parameters "ParameterName=max_connections,ParameterValue=200,ApplyMethod=pending-reboot"
   ```

### 5. ALB Health Checks Failing

**Symptoms:**
- Targets marked unhealthy in ALB
- Traffic not reaching pods
- 503 errors from ALB

**Diagnosis:**

```bash
# Check target health
aws elbv2 describe-target-health \
  --target-group-arn <tg-arn>

# Check ALB logs in S3
aws s3 cp s3://chess-ai-alb-logs/latest.log - | grep -i error

# Test health endpoint
kubectl port-forward svc/chess-ai 8000:80 -n chess-ai
curl localhost:8000/health/ready
```

**Solutions:**

1. **Readiness Probe Failing**
   ```bash
   # Check readiness configuration
   kubectl get deployment chess-ai -n chess-ai -o yaml | grep -A 10 readinessProbe
   
   # Adjust readiness probe timing
   kubectl patch deployment chess-ai -n chess-ai -p '
     {"spec":{"template":{"spec":{"containers":[{
       "name":"chess-ai",
       "readinessProbe":{"initialDelaySeconds":30,"periodSeconds":10}
     }]}}}}'
   ```

2. **Port Mismatch**
   ```bash
   # Verify service port configuration
   kubectl get svc chess-ai -n chess-ai -o yaml
   
   # Verify target group port
   aws elbv2 describe-target-groups \
     --target-group-arns <tg-arn> \
     --query 'TargetGroups[0].Port'
   ```

### 6. High CPU Usage

**Symptoms:**
- CPU throttling
- Slow response times
- HPA scaling pods aggressively

**Diagnosis:**

```bash
# Check CPU usage
kubectl top pods -n chess-ai

# Check CPU limits
kubectl describe pod <pod-name> -n chess-ai | grep -i limits

# Profile application
kubectl exec -it <pod-name> -n chess-ai -- python -m cProfile -o profile.stats app.py
```

**Solutions:**

1. **Increase CPU Limits**
   ```bash
   helm upgrade chess-ai ./helm/chess-ai \
     --set resources.limits.cpu=4000m \
     --set resources.requests.cpu=1000m \
     -n chess-ai
   ```

2. **Scale Horizontally**
   ```bash
   kubectl scale deployment/chess-ai --replicas=10 -n chess-ai
   ```

3. **Optimize Application**
   ```python
   # Profile and optimize hot paths
   # Use caching for frequently accessed data
   # Optimize database queries
   ```

### 7. SSL/TLS Certificate Issues

**Symptoms:**
- Certificate expired warnings
- HTTPS connections failing
- Browser security errors

**Diagnosis:**

```bash
# Check certificate expiration
openssl s_client -connect chess-ai.example.com:443 -servername chess-ai.example.com < /dev/null 2>/dev/null | \
  openssl x509 -noout -dates

# Check ACM certificate
aws acm describe-certificate \
  --certificate-arn <cert-arn> \
  --query 'Certificate.{Status:Status,NotAfter:NotAfter}'

# Check cert-manager certificate
kubectl get certificate -n chess-ai
kubectl describe certificate chess-ai-tls -n chess-ai
```

**Solutions:**

1. **Renew Certificate**
   ```bash
   # For ACM (auto-renewed, but check validation)
   aws acm resend-validation-email \
     --certificate-arn <cert-arn> \
     --domain chess-ai.example.com
   
   # For cert-manager
   kubectl delete certificate chess-ai-tls -n chess-ai
   # Certificate will be recreated automatically
   ```

2. **Update Certificate ARN**
   ```bash
   # Update Ingress annotation
   kubectl annotate ingress chess-ai -n chess-ai \
     cert-manager.io/cluster-issuer=letsencrypt-prod --overwrite
   ```

### 8. Out of Disk Space

**Symptoms:**
- Pods evicted
- Write operations failing
- Node pressure warnings

**Diagnosis:**

```bash
# Check node disk usage
kubectl get nodes
kubectl describe node <node-name> | grep -i disk

# Check PV usage
kubectl get pv
kubectl df -h /data  # inside pod

# Check log sizes
kubectl exec -it <pod-name> -n chess-ai -- du -sh /var/log/*
```

**Solutions:**

1. **Clean Up Logs**
   ```bash
   # Rotate logs
   kubectl exec -it <pod-name> -n chess-ai -- logrotate /etc/logrotate.conf
   
   # Reduce log retention
   helm upgrade chess-ai ./helm/chess-ai \
     --set logging.retention=3d \
     -n chess-ai
   ```

2. **Increase PV Size**
   ```bash
   # For EBS-backed PV (requires storage class with allowVolumeExpansion: true)
   kubectl edit pvc chess-ai-data -n chess-ai
   # Increase spec.resources.requests.storage
   ```

3. **Add More Nodes**
   ```bash
   aws eks update-nodegroup-config \
     --cluster-name chess-ai-prod \
     --nodegroup-name general \
     --scaling-config desiredSize=6
   ```

## Monitoring Queries

### CloudWatch Insights

```sql
-- Find 5xx errors
fields @timestamp, @message
| filter @message like /5\d{2}/
| sort @timestamp desc
| limit 100

-- Slow requests
fields @timestamp, @message
| filter @message like /took \d{3,} ms/
| sort @timestamp desc
| limit 50

-- Error patterns
fields @timestamp, @message
| filter @message like /ERROR|Exception/
| stats count() by bin(5m)
```

### Prometheus Queries

```promql
# Request rate
sum(rate(http_requests_total[5m]))

# Error rate
sum(rate(http_requests_total{status=~"5.."}[5m]))

# Request latency (p95)
histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))

# Pod CPU usage
sum(rate(container_cpu_usage_seconds_total{namespace="chess-ai"}[5m])) by (pod)

# Pod memory usage
sum(container_memory_working_set_bytes{namespace="chess-ai"}) by (pod)
```

## Emergency Procedures

### Complete Service Outage

1. **Assess Severity**
   ```bash
   # Check all components
   kubectl get pods -n chess-ai
   aws elbv2 describe-target-health --target-group-arn <tg-arn>
   aws rds describe-db-instances --db-instance-identifier chess-ai-prod
   ```

2. **Check Recent Changes**
   ```bash
   # Check recent deployments
   kubectl rollout history deployment/chess-ai -n chess-ai
   
   # Check Terraform changes
   cd infrastructure/terraform/aws
   terraform show
   ```

3. **Rollback if Needed**
   ```bash
   # Rollback application
   helm rollback chess-ai -n chess-ai
   
   # Rollback infrastructure
   cd infrastructure/terraform/aws
   terraform apply -auto-approve -target=<resource>
   ```

4. **Activate Incident Response**
   ```bash
   # Page on-call team
   curl -X POST https://api.pagerduty.com/incidents \
     -H "Authorization: Token token=<token>" \
     -H "Content-Type: application/json" \
     -d '{
       "incident": {
         "type": "incident",
         "title": "Chess AI Complete Outage",
         "service": {"id": "<service-id>", "type": "service_reference"},
         "urgency": "high",
         "body": {"type": "incident_body", "details": "Complete service outage"}
       }
     }'
   
   # Update status page
   # https://status.chess-ai.example.com
   ```

### Data Corruption

1. **Stop Writes**
   ```bash
   # Scale down to prevent further corruption
   kubectl scale deployment/chess-ai --replicas=0 -n chess-ai
   ```

2. **Assess Damage**
   ```bash
   # Check database
   aws rds describe-db-snapshots \
     --db-instance-identifier chess-ai-prod
   
   # Check data integrity
   kubectl exec -it <pod-name> -n chess-ai -- python check_data_integrity.py
   ```

3. **Restore from Backup**
   ```bash
   # Restore RDS from snapshot
   aws rds restore-db-instance-from-db-snapshot \
     --db-instance-identifier chess-ai-restored \
     --db-snapshot-identifier chess-ai-$(date +%Y%m%d)
   
   # Update endpoint
   kubectl set env deployment/chess-ai \
     DB_HOST=chess-ai-restored.<region>.rds.amazonaws.com \
     -n chess-ai
   ```

4. **Verify and Resume**
   ```bash
   # Verify data integrity
   # Test in staging first
   
   # Scale back up
   kubectl scale deployment/chess-ai --replicas=3 -n chess-ai
   ```

### Security Breach

1. **Immediate Actions**
   ```bash
   # Isolate affected resources
   kubectl scale deployment/chess-ai --replicas=0 -n chess-ai
   
   # Block malicious IPs in WAF
   aws wafv2 update-ip-set \
     --name blocked-ips \
     --id <ip-set-id> \
     --scope REGIONAL \
     --addresses 203.0.113.0/24
   
   # Rotate credentials immediately
   aws secretsmanager rotate-secret \
     --secret-id chess-ai/prod/database
   ```

2. **Investigate**
   ```bash
   # Check CloudTrail logs
   aws cloudtrail lookup-events \
     --start-time $(date -u -d '24 hours ago' +%Y-%m-%dT%H:%M:%S) \
     --max-results 100
   
   # Check GuardDuty findings
   aws guardduty list-findings \
     --detector-id <detector-id> \
     --finding-criteria '{"Criterion":{"severity":{"Gte":7}}}'
   
   # Check access logs
   aws s3 sync s3://chess-ai-alb-logs/ ./logs/
   grep -i "suspicious" ./logs/*.log
   ```

3. **Contain and Remediate**
   ```bash
   # Patch vulnerabilities
   # Update dependencies
   # Rebuild and redeploy clean images
   
   # Enable additional monitoring
   aws guardduty create-detector --enable
   ```

4. **Document and Report**
   ```bash
   # Document timeline
   # Identify root cause
   # Implement preventive measures
   # Notify stakeholders
   # File incident report
   ```

## Maintenance Windows

### Scheduled Maintenance Procedure

1. **Pre-Maintenance (1 week before)**
   - Announce maintenance window
   - Update status page
   - Notify users via email
   - Prepare rollback plan

2. **Pre-Maintenance (1 hour before)**
   ```bash
   # Create backup
   aws rds create-db-snapshot \
     --db-instance-identifier chess-ai-prod \
     --db-snapshot-identifier chess-ai-maint-$(date +%Y%m%d-%H%M)
   
   # Verify current state
   kubectl get all -n chess-ai
   helm list -n chess-ai
   ```

3. **During Maintenance**
   ```bash
   # Update status page to "maintenance"
   
   # Perform updates
   helm upgrade chess-ai ./helm/chess-ai -n chess-ai
   
   # Monitor progress
   kubectl rollout status deployment/chess-ai -n chess-ai
   ```

4. **Post-Maintenance**
   ```bash
   # Verify functionality
   curl https://chess-ai.example.com/health
   
   # Run smoke tests
   ./scripts/smoke-tests.sh
   
   # Update status page to "operational"
   
   # Send completion notification
   ```

## Useful Commands Cheat Sheet

```bash
# Kubernetes
kubectl get pods -n chess-ai -o wide
kubectl logs -f deployment/chess-ai -n chess-ai
kubectl exec -it <pod> -n chess-ai -- /bin/bash
kubectl describe pod <pod> -n chess-ai
kubectl rollout restart deployment/chess-ai -n chess-ai
kubectl rollout undo deployment/chess-ai -n chess-ai
kubectl top pods -n chess-ai
kubectl get events -n chess-ai --sort-by='.lastTimestamp'

# Helm
helm list -n chess-ai
helm history chess-ai -n chess-ai
helm rollback chess-ai <revision> -n chess-ai
helm upgrade chess-ai ./helm/chess-ai -n chess-ai

# AWS CLI
aws eks update-kubeconfig --name chess-ai-prod --region us-east-1
aws rds describe-db-instances --db-instance-identifier chess-ai-prod
aws elbv2 describe-target-health --target-group-arn <arn>
aws cloudwatch get-metric-statistics --namespace AWS/ApplicationELB

# Docker
docker ps
docker logs <container-id>
docker exec -it <container-id> /bin/bash
docker stats

# System
top
htop
free -h
df -h
netstat -tulpn
ss -tulpn
journalctl -u kubelet -f
```

## Escalation Path

1. **Level 1**: On-call engineer (PagerDuty)
2. **Level 2**: DevOps lead
3. **Level 3**: Engineering manager
4. **Level 4**: CTO

**Escalation Triggers:**
- Service down > 15 minutes
- Data corruption detected
- Security breach confirmed
- Unable to resolve within 30 minutes
