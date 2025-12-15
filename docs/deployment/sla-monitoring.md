# SLA Definitions and Monitoring

## Service Level Agreements (SLAs)

### Production Environment

#### Availability SLA

**Target**: 99.9% uptime (Three Nines)

- **Monthly Downtime Budget**: 43.2 minutes
- **Annual Downtime Budget**: 8.76 hours
- **Measurement Period**: Calendar month
- **Exclusions**: Scheduled maintenance (announced 7 days in advance)

**Calculation**:
```
Availability % = (Total Time - Downtime) / Total Time × 100
```

**Monitoring**:
- Synthetic monitoring from multiple geographic locations
- Health check endpoint: `/health`
- Check frequency: Every 60 seconds
- Threshold: 3 consecutive failures = downtime

#### Performance SLA

**API Response Time**:
- **P50 (Median)**: < 100ms
- **P95**: < 500ms
- **P99**: < 1000ms
- **P99.9**: < 2000ms

**Measurement**:
- Application load balancer metrics
- Application-level timing
- End-to-end synthetic transactions

#### Error Rate SLA

**Target**: < 0.1% error rate

- **4xx Errors**: < 1% (client errors)
- **5xx Errors**: < 0.1% (server errors)
- **Timeout Errors**: < 0.05%

**Calculation**:
```
Error Rate % = (Failed Requests / Total Requests) × 100
```

### Staging Environment

- **Availability**: 95% (best effort)
- **Response Time**: < 1000ms (P95)
- **Error Rate**: < 1%

### Development Environment

- **Availability**: 90% (best effort)
- **No formal SLA commitments**

## Service Level Objectives (SLOs)

### Operational SLOs

| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| Availability | 99.9% | < 99.5% | < 99% |
| P95 Latency | < 500ms | > 1s | > 2s |
| P99 Latency | < 1s | > 2s | > 5s |
| Error Rate | < 0.1% | > 0.5% | > 1% |
| API Success Rate | > 99.9% | < 99.5% | < 99% |
| Database Connection Time | < 10ms | > 50ms | > 100ms |
| Health Check Response | < 50ms | > 200ms | > 500ms |

### Infrastructure SLOs

| Component | Availability | Recovery Time |
|-----------|--------------|---------------|
| EKS Cluster | 99.95% | < 5 minutes |
| RDS Database | 99.95% | < 5 minutes |
| Application Load Balancer | 99.99% | < 1 minute |
| CloudFront CDN | 99.99% | N/A (AWS managed) |
| S3 Storage | 99.99% | N/A (AWS managed) |

### Recovery Objectives

**RTO (Recovery Time Objective)**:
- **Tier 1 - Critical**: 15 minutes
- **Tier 2 - Important**: 1 hour
- **Tier 3 - Normal**: 4 hours

**RPO (Recovery Point Objective)**:
- **Database**: 5 minutes (point-in-time recovery)
- **File Storage**: 1 hour (S3 versioning)
- **Configuration**: 0 minutes (Git version control)

## Monitoring Strategy

### Health Checks

#### Liveness Probe
```yaml
livenessProbe:
  httpGet:
    path: /health/live
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3
```

**Purpose**: Detect if the application is running
**Action**: Restart pod if failing

#### Readiness Probe
```yaml
readinessProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 3
```

**Purpose**: Detect if the application can serve traffic
**Action**: Remove from load balancer if failing

### Synthetic Monitoring

**External Monitoring** (from multiple regions):

```bash
# Health check
curl -f https://chess-ai.example.com/health

# Move endpoint check
curl -X POST https://chess-ai.example.com/move \
  -H "Content-Type: application/json" \
  -d '{"fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"}'
```

**Monitoring Locations**:
- US East (Virginia)
- US West (Oregon)
- EU West (Ireland)
- Asia Pacific (Singapore)

**Check Frequency**: Every 60 seconds
**Alert Threshold**: 3 consecutive failures

### Real User Monitoring (RUM)

**Metrics Collected**:
- Page load time
- Time to first byte (TTFB)
- Time to interactive (TTI)
- API response times
- Error rates
- Browser/device information

**Implementation**:
```javascript
// Client-side performance monitoring
window.addEventListener('load', () => {
  const perfData = window.performance.timing;
  const pageLoadTime = perfData.loadEventEnd - perfData.navigationStart;
  
  // Send to analytics
  sendMetric('page_load_time', pageLoadTime);
});
```

### Application Performance Monitoring (APM)

**Metrics**:
- Request rate
- Error rate
- Response time distribution
- Database query performance
- External API call performance
- Memory usage
- CPU usage
- Garbage collection metrics

**Tools**:
- Prometheus + Grafana
- AWS CloudWatch
- Custom `/metrics` endpoint

### Infrastructure Monitoring

**Kubernetes Metrics**:
- Pod health and status
- Resource utilization (CPU, memory)
- Node health
- Persistent volume usage
- Network traffic

**AWS Metrics**:
- ALB metrics (request count, latency, error rate)
- RDS metrics (connections, CPU, IOPS, storage)
- EKS control plane metrics
- CloudFront metrics (requests, data transfer)

### Log Aggregation

**Centralized Logging**:
- Application logs → CloudWatch Logs
- Container logs → CloudWatch Container Insights
- ALB access logs → S3
- VPC Flow Logs → CloudWatch Logs
- CloudTrail audit logs → S3

**Log Retention**:
- Application logs: 30 days (dev), 90 days (prod)
- Access logs: 90 days
- Audit logs: 1 year (compliance requirement)

### Alerting

**Alert Channels**:
1. **PagerDuty**: Critical alerts (24/7 on-call)
2. **Slack**: Warning and informational alerts
3. **Email**: Daily/weekly summaries
4. **SMS**: Critical production incidents only

**Alert Severity Levels**:

| Severity | Response Time | Notification Method |
|----------|---------------|---------------------|
| Critical | < 5 minutes | PagerDuty + Slack + SMS |
| High | < 30 minutes | PagerDuty + Slack |
| Medium | < 2 hours | Slack + Email |
| Low | Next business day | Email |

**Alert Rules**:

```yaml
# Critical Alerts (P1)
- Service completely unavailable (all pods down)
- Error rate > 5%
- Database unavailable
- Data corruption detected
- Security breach detected

# High Alerts (P2)
- Availability < 99%
- Error rate > 1%
- P95 latency > 2s
- Less than 2 pods available
- Node failure

# Medium Alerts (P3)
- Error rate > 0.5%
- P95 latency > 1s
- High CPU/memory usage (> 80%)
- Certificate expiring in < 30 days
- Backup failure

# Low Alerts (P4)
- Error rate > 0.1%
- P95 latency > 500ms
- High resource usage (> 70%)
- Certificate expiring in < 60 days
```

## SLA Reporting

### Daily Reports

**Automated Report** (generated at 9 AM UTC):
- Yesterday's availability
- Error rate summary
- P95/P99 latency
- Incident count and duration
- Top errors

**Distribution**: Slack #chess-ai-metrics

### Weekly Reports

**Comprehensive Report** (generated every Monday):
- 7-day availability trend
- Performance metrics (with graphs)
- Error analysis
- Incident post-mortems
- Capacity planning metrics
- Cost analysis

**Distribution**: Email to engineering team

### Monthly Reports

**Executive Summary**:
- SLA compliance status
- Monthly uptime percentage
- Incident summary
- Key metrics trends
- Capacity planning recommendations
- Cost optimization opportunities

**Distribution**: Email to leadership team

### SLA Dashboard

**Real-time Dashboard** (Grafana):

```
https://grafana.chess-ai.example.com/d/sla-overview
```

**Panels**:
1. Current availability (today, week, month)
2. Error rate (last 24h, 7d, 30d)
3. Latency distribution (P50, P95, P99)
4. Incident timeline
5. SLA budget remaining (monthly)
6. Request rate
7. Active users
8. Resource utilization

### SLA Tracking

**Monthly SLA Calculation**:

```python
# Calculate monthly availability
total_minutes = days_in_month * 24 * 60
downtime_minutes = sum(incident_durations)
availability = (total_minutes - downtime_minutes) / total_minutes * 100

# Check SLA compliance
sla_target = 99.9
is_compliant = availability >= sla_target

# Calculate SLA credits (if applicable)
if not is_compliant:
    credit_percentage = calculate_credit(availability)
```

**SLA Credit Tiers** (if offering paid service):

| Availability | Credit |
|--------------|--------|
| < 99.9% but ≥ 99.0% | 10% |
| < 99.0% but ≥ 95.0% | 25% |
| < 95.0% | 50% |

## Incident Management

### Severity Definitions

**Severity 1 (Critical)**:
- Complete service outage
- Data loss or corruption
- Security breach
- Impact: All users affected

**Severity 2 (High)**:
- Partial service outage
- Severe performance degradation
- Major feature unavailable
- Impact: Many users affected

**Severity 3 (Medium)**:
- Minor feature unavailable
- Performance degradation
- Intermittent errors
- Impact: Some users affected

**Severity 4 (Low)**:
- Minor bug
- Cosmetic issue
- Documentation error
- Impact: Few users affected

### Incident Response Process

1. **Detection** (T+0)
   - Alert triggered
   - On-call engineer paged

2. **Acknowledgment** (T+5 min)
   - Engineer acknowledges incident
   - Initial assessment

3. **Investigation** (T+10 min)
   - Identify root cause
   - Determine impact

4. **Communication** (T+15 min)
   - Update status page
   - Notify stakeholders

5. **Mitigation** (T+15-30 min)
   - Implement fix or workaround
   - Monitor for improvement

6. **Resolution** (varies)
   - Confirm service restored
   - Update status page

7. **Post-Mortem** (within 48h)
   - Root cause analysis
   - Action items
   - Process improvements

### Status Page

**Public Status Page**: https://status.chess-ai.example.com

**Components Monitored**:
- API Service
- Web Application
- Database
- CDN
- Authentication (future)

**Incident Updates**:
- Investigating (initial)
- Identified (root cause known)
- Monitoring (fix deployed)
- Resolved (fully resolved)

**Maintenance Windows**:
- Scheduled at least 7 days in advance
- Typically Tuesday 2-4 AM UTC
- Announced on status page and via email

## Capacity Planning

### Growth Targets

**Year 1**:
- Users: 10,000 → 100,000
- Requests: 1M/day → 10M/day
- Data: 100GB → 1TB

**Year 2**:
- Users: 100,000 → 1,000,000
- Requests: 10M/day → 100M/day
- Data: 1TB → 10TB

### Resource Scaling Thresholds

**Application Pods**:
- Current: 3-20 pods
- Scale up at: 70% CPU or 80% memory
- Scale down at: 30% CPU and 40% memory

**Database**:
- Current: db.t3.medium
- Upgrade at: 80% CPU or 80% storage
- Read replicas: Add at 1000+ concurrent connections

**Storage**:
- Current: 100GB RDS, 50GB S3
- Expand at: 80% utilization
- Archive to Glacier: After 90 days

### Cost Optimization

**Current Costs** (estimated monthly):
- EKS: $200
- EC2 (nodes): $500
- RDS: $150
- ALB: $50
- S3: $20
- CloudWatch: $50
- Data Transfer: $100
- **Total**: ~$1,070/month

**Optimization Strategies**:
- Use Spot instances for non-critical workloads (70% savings)
- Reserved instances for baseline capacity (40% savings)
- S3 lifecycle policies (90% savings on archived data)
- CloudWatch log retention tuning (50% savings)
- Rightsize instances based on utilization

## Continuous Improvement

### Monthly Review

- Analyze SLA compliance
- Review incident trends
- Identify performance bottlenecks
- Update capacity forecasts
- Review and update alert thresholds

### Quarterly Review

- Comprehensive performance analysis
- Infrastructure optimization opportunities
- SLA target review and adjustment
- Tool evaluation (APM, monitoring)
- Cost optimization initiatives

### Annual Review

- SLA performance report
- Infrastructure roadmap
- Disaster recovery testing
- Compliance audit
- Budget planning

## Compliance and Audit

### Audit Requirements

**SOC 2 Type II**:
- Availability monitoring logs (12 months)
- Incident reports and post-mortems
- Change management records
- Access logs
- Backup and recovery testing results

**GDPR**:
- Data processing agreements
- Privacy impact assessments
- Breach notification procedures

### Documentation

All monitoring data retained for:
- Metrics: 15 days (Prometheus), 90 days (CloudWatch)
- Logs: 30-90 days (depending on type)
- Audit logs: 365 days
- Incident reports: Indefinitely

## Contact Information

**SLA Questions**:
- Email: sla@chess-ai.example.com
- Slack: #sla-discussions

**Incident Reports**:
- PagerDuty: https://yourorg.pagerduty.com
- Slack: #incidents

**Status Page**:
- https://status.chess-ai.example.com
- Twitter: @ChessAIStatus
