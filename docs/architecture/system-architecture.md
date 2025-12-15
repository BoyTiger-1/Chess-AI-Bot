# System Architecture

## Overview

Chess AI Bot is a production-grade, cloud-native application that provides intelligent chess move suggestions based on historical game data analysis. The system is designed to be highly available, scalable, and secure.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Internet Users                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                    ┌───────▼────────┐
                    │   CloudFront   │  ← CDN for static assets
                    │      (CDN)     │  ← SSL/TLS termination
                    └───────┬────────┘  ← DDoS protection
                            │
                    ┌───────▼────────┐
                    │      WAF       │  ← Web Application Firewall
                    │   (AWS WAF)    │  ← Rate limiting
                    └───────┬────────┘  ← SQL injection protection
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼─────┐     ┌───────▼─────┐     ┌─────▼──────┐
│    ALB      │     │    ALB      │     │    ALB     │
│  (AZ-1)     │     │  (AZ-2)     │     │  (AZ-3)    │
└───────┬─────┘     └───────┬─────┘     └─────┬──────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                    ┌───────▼────────┐
                    │  EKS Cluster   │
                    │  (Kubernetes)  │
                    └───────┬────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼─────┐     ┌───────▼─────┐     ┌─────▼──────┐
│  Pod Group  │     │  Pod Group  │     │  Pod Group │
│   (AZ-1)    │     │   (AZ-2)    │     │   (AZ-3)   │
│             │     │             │     │            │
│ ┌─────────┐ │     │ ┌─────────┐ │     │ ┌────────┐ │
│ │Flask App│ │     │ │Flask App│ │     │ │Flask   │ │
│ │+ Gunicorn│     │ │+ Gunicorn│     │ │App     │ │
│ └─────────┘ │     │ └─────────┘ │     │ └────────┘ │
└─────────────┘     └─────────────┘     └────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                    ┌───────▼────────┐
                    │   RDS Multi-AZ │
                    │  (PostgreSQL)  │
                    │                │
                    │  ┌──────────┐  │
                    │  │ Primary  │  │
                    │  └────┬─────┘  │
                    │       │        │
                    │  ┌────▼─────┐  │
                    │  │ Standby  │  │
                    │  └──────────┘  │
                    └────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Supporting Services                           │
├─────────────────────────────────────────────────────────────────┤
│  S3 Buckets    │  CloudWatch    │  Secrets Manager │  Backup   │
│  - Static      │  - Logs        │  - DB creds      │  - RDS    │
│  - Data        │  - Metrics     │  - API keys      │  - S3     │
│  - Backups     │  - Alarms      │  - Certificates  │           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Monitoring Stack                              │
├─────────────────────────────────────────────────────────────────┤
│  Prometheus    │    Grafana     │  AlertManager  │  CloudWatch │
│  - Metrics     │  - Dashboards  │  - Alerts      │  - APM      │
│  - Scraping    │  - Visualization│ - PagerDuty   │  - Tracing  │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Frontend Layer

#### CloudFront (CDN)
- **Purpose**: Content delivery, SSL/TLS termination, caching
- **Configuration**:
  - Origin: Application Load Balancer
  - Cache behavior: Static assets cached for 1 day
  - Compression: Gzip/Brotli enabled
  - SSL/TLS: TLS 1.2+ only
  - Geographic restrictions: Configurable

#### AWS WAF
- **Purpose**: Web application firewall, security rules
- **Protection**:
  - Rate limiting (100 requests/5min per IP)
  - SQL injection protection
  - XSS protection
  - Known bad inputs blocking
  - Geographic blocking (optional)
  - Bot detection and mitigation

### 2. Load Balancing Layer

#### Application Load Balancer (ALB)
- **Purpose**: Distribute traffic across multiple availability zones
- **Configuration**:
  - Protocol: HTTP/HTTPS
  - Health checks: `/health/ready` endpoint
  - Idle timeout: 60 seconds
  - Connection draining: 300 seconds
  - Sticky sessions: Disabled (stateless app)
  - Target groups: EKS node groups

### 3. Application Layer

#### EKS Cluster (Kubernetes)
- **Purpose**: Container orchestration
- **Configuration**:
  - Version: 1.28+
  - Node groups: 3 (one per AZ)
  - Auto-scaling: 3-20 nodes
  - Instance types: t3.medium (general), c5.xlarge (compute)
  - Spot instances: Enabled for cost optimization

#### Application Pods
- **Container**: Flask + Gunicorn
- **Replicas**: 3-20 (auto-scaled)
- **Resources**:
  - CPU: 500m request, 2000m limit
  - Memory: 1Gi request, 4Gi limit
- **Probes**:
  - Liveness: `/health/live`
  - Readiness: `/health/ready`
  - Startup: 30s timeout
- **Security**:
  - Non-root user (UID 1000)
  - Read-only root filesystem
  - No privileged escalation
  - Security context constraints

### 4. Data Layer

#### RDS PostgreSQL (Optional - for future use)
- **Purpose**: Persistent data storage, user data, analytics
- **Configuration**:
  - Engine: PostgreSQL 15.4
  - Instance class: db.t3.medium (prod), db.t3.small (dev)
  - Multi-AZ: Enabled (prod)
  - Storage: 100GB GP3, auto-scaling to 500GB
  - Backup retention: 30 days (prod), 7 days (dev)
  - Encryption: AES-256 at rest
  - SSL/TLS: Required for connections
  - Automated backups: Daily at 3:00 AM UTC
  - Maintenance window: Monday 4:00-5:00 AM UTC

#### S3 Buckets
- **Purpose**: Static assets, data files, backups
- **Buckets**:
  1. `chess-ai-static-{env}`: Static assets (chess pieces, CSS, JS)
  2. `chess-ai-data-{env}`: Game data, move databases
  3. `chess-ai-backups-{env}`: Database backups, snapshots
- **Configuration**:
  - Versioning: Enabled
  - Encryption: AES-256 (SSE-S3)
  - Lifecycle policies: 90-day archive to Glacier
  - Replication: Cross-region (prod only)
  - Access logs: Enabled

### 5. Security Layer

#### Secrets Manager
- **Purpose**: Secure credential storage
- **Secrets**:
  - Database credentials
  - API keys and tokens
  - SSL/TLS certificates
  - Encryption keys
- **Configuration**:
  - Automatic rotation: 90 days
  - Encryption: AWS KMS
  - Access logging: CloudTrail

#### IAM Roles and Policies
- **Principle**: Least privilege access
- **Roles**:
  - EKS node role: EC2, ECR, CloudWatch
  - Pod service account role: Secrets Manager, S3
  - RDS role: Monitoring, backups
- **Policies**:
  - Read-only for most services
  - Write access only where required
  - MFA required for production changes

### 6. Monitoring and Observability

#### Prometheus
- **Purpose**: Metrics collection and storage
- **Metrics**:
  - Application metrics: `/metrics` endpoint
  - Container metrics: cAdvisor
  - Node metrics: Node Exporter
  - Kubernetes metrics: kube-state-metrics
- **Retention**: 15 days

#### Grafana
- **Purpose**: Visualization and dashboards
- **Dashboards**:
  - Application performance
  - Infrastructure health
  - Business metrics
  - Cost analysis

#### CloudWatch
- **Purpose**: AWS service monitoring, logs aggregation
- **Configuration**:
  - Log groups: Application, EKS, RDS, ALB
  - Retention: 30 days (dev), 90 days (prod)
  - Alarms: CPU, memory, disk, latency, errors
  - SNS notifications: Email, Slack, PagerDuty

## Data Flow

### 1. User Request Flow

```
User → CloudFront → WAF → ALB → Kubernetes Service → Pod → Response
```

1. User makes HTTPS request to `chess-ai.example.com`
2. CloudFront receives request, checks cache
3. If cache miss, forwards to ALB through WAF
4. WAF inspects request for threats
5. ALB performs health check and routes to healthy pod
6. Kubernetes service selects available pod
7. Pod processes request (Flask/Gunicorn)
8. Response flows back through the chain

### 2. Move Suggestion Flow

```
Client → POST /move → Load FEN → Query move_db → Calculate best move → Return move
```

1. Client sends POST request with FEN position
2. Application parses FEN and validates
3. Queries in-memory move database
4. Calculates win ratios from historical data
5. Selects best move(s) using statistical analysis
6. Returns move in UCI notation

### 3. Health Check Flow

```
ALB → GET /health/ready → Check move_db → Return 200/503
```

1. ALB sends health check every 10 seconds
2. Application checks if move database is loaded
3. Returns 200 if healthy, 503 if not ready
4. ALB marks pod as healthy/unhealthy

## Scaling Strategy

### Horizontal Scaling
- **Trigger**: CPU > 70% or Memory > 80%
- **Action**: Add pods (up to max 20)
- **Scale-down**: Gradual, 50% per minute
- **Scale-up**: Aggressive, 100% per 30 seconds

### Vertical Scaling
- **Node groups**: Multiple instance types
- **Spot instances**: For non-critical workloads
- **Reserved instances**: For baseline capacity

### Database Scaling
- **Read replicas**: For read-heavy workloads
- **Connection pooling**: PgBouncer
- **Caching**: Redis/Memcached (future)

## High Availability

### Multi-AZ Deployment
- **Application**: Pods distributed across 3 AZs
- **Database**: RDS Multi-AZ with automatic failover
- **Load balancer**: ALB in all AZs

### Disaster Recovery
- **RTO**: 15 minutes (recovery time objective)
- **RPO**: 5 minutes (recovery point objective)
- **Backup strategy**:
  - Automated daily RDS snapshots
  - Point-in-time recovery (PITR)
  - Cross-region replication (prod)
  - S3 versioning and lifecycle policies

### Fault Tolerance
- **Pod disruption budget**: Minimum 2 pods available
- **Health checks**: Automatic pod restart on failure
- **Circuit breakers**: Prevent cascading failures
- **Graceful degradation**: Fallback to random moves

## Security Architecture

### Network Security
- **VPC**: Isolated network, private subnets
- **Security groups**: Least privilege firewall rules
- **NACLs**: Additional network layer security
- **VPC Flow Logs**: Network traffic monitoring

### Application Security
- **Authentication**: JWT tokens (future)
- **Authorization**: RBAC (future)
- **Input validation**: Schema validation
- **Output encoding**: XSS prevention
- **HTTPS only**: TLS 1.2+ enforced
- **CORS**: Restricted origins

### Data Security
- **Encryption at rest**: AES-256
- **Encryption in transit**: TLS 1.3
- **Key management**: AWS KMS
- **Secrets rotation**: Automated 90-day rotation
- **Data anonymization**: PII handling (future)

### Compliance
- **GDPR**: Data privacy, right to deletion
- **CCPA**: California privacy compliance
- **SOC 2**: Security controls documentation
- **PCI DSS**: Not applicable (no payments)

## Performance Optimization

### Caching Strategy
- **CDN caching**: Static assets (1 day TTL)
- **Application caching**: Move database in-memory
- **Browser caching**: Cache-Control headers
- **Database caching**: Query result caching (future)

### Database Optimization
- **Indexing**: Strategic indexes on frequently queried columns
- **Connection pooling**: Reduce connection overhead
- **Query optimization**: EXPLAIN ANALYZE for slow queries
- **Partitioning**: Table partitioning for large datasets (future)

### Application Optimization
- **Gunicorn**: Multiple workers and threads
- **Async processing**: Background jobs (future)
- **Code optimization**: Profiling and optimization
- **Resource limits**: Prevent resource exhaustion

## Cost Optimization

### Infrastructure
- **Spot instances**: 70% cost savings for compute
- **Auto-scaling**: Scale down during off-peak
- **Reserved instances**: 40% savings for baseline
- **S3 lifecycle policies**: Archive to Glacier

### Monitoring
- **CloudWatch**: Optimize log retention
- **Prometheus**: Efficient metric storage
- **Cost allocation tags**: Track spending by component
- **Budget alerts**: Prevent cost overruns

## Technology Stack

### Application
- **Language**: Python 3.12
- **Framework**: Flask 3.0+
- **WSGI Server**: Gunicorn
- **Chess Engine**: python-chess
- **Data Processing**: pandas

### Infrastructure
- **Cloud Provider**: AWS (primary), GCP/Azure (alternative)
- **Container Runtime**: Docker
- **Orchestration**: Kubernetes (EKS)
- **IaC**: Terraform
- **Package Manager**: Helm

### Monitoring
- **Metrics**: Prometheus, CloudWatch
- **Visualization**: Grafana
- **Logging**: CloudWatch Logs, ELK Stack
- **Tracing**: AWS X-Ray (future)
- **Alerting**: AlertManager, SNS

### CI/CD
- **Source Control**: Git (GitHub/GitLab)
- **CI/CD**: GitHub Actions, GitLab CI
- **Container Registry**: ECR, DockerHub
- **Deployment**: Helm, kubectl

## Future Enhancements

1. **Caching Layer**: Redis/Memcached for hot data
2. **Message Queue**: SQS/RabbitMQ for async processing
3. **ML Pipeline**: Model training and deployment
4. **API Gateway**: Kong/API Gateway for advanced routing
5. **Service Mesh**: Istio for advanced traffic management
6. **Multi-region**: Global deployment for low latency
7. **GraphQL API**: Advanced querying capabilities
8. **WebSocket**: Real-time move streaming
9. **Mobile Apps**: iOS/Android native apps
10. **AI Analytics**: Advanced analytics and insights
