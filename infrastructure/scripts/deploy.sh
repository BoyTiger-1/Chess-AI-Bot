#!/bin/bash
set -euo pipefail

# Chess AI Bot Deployment Script
# Usage: ./deploy.sh [environment] [version]

ENVIRONMENT=${1:-dev}
VERSION=${2:-latest}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Validate environment
validate_environment() {
    if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|prod)$ ]]; then
        log_error "Invalid environment: $ENVIRONMENT. Must be dev, staging, or prod."
        exit 1
    fi
    
    log_info "Deploying to environment: $ENVIRONMENT"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local tools=("kubectl" "helm" "docker" "aws")
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is not installed"
            exit 1
        fi
    done
    
    log_info "All prerequisites met"
}

# Configure AWS CLI
configure_aws() {
    log_info "Configuring AWS CLI..."
    
    export AWS_REGION=${AWS_REGION:-us-east-1}
    export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    
    log_info "AWS Account: $AWS_ACCOUNT_ID"
    log_info "AWS Region: $AWS_REGION"
}

# Configure kubectl
configure_kubectl() {
    log_info "Configuring kubectl..."
    
    aws eks update-kubeconfig \
        --name "chess-ai-${ENVIRONMENT}" \
        --region "$AWS_REGION"
    
    # Verify connection
    if ! kubectl get nodes &> /dev/null; then
        log_error "Failed to connect to Kubernetes cluster"
        exit 1
    fi
    
    log_info "Connected to EKS cluster: chess-ai-${ENVIRONMENT}"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    
    cd "$PROJECT_ROOT"
    
    local image_tag="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/chess-ai-bot:${VERSION}"
    
    docker build \
        --platform linux/amd64 \
        --build-arg VERSION="$VERSION" \
        --tag "$image_tag" \
        --tag "chess-ai-bot:$VERSION" \
        .
    
    log_info "Image built successfully: $image_tag"
    echo "$image_tag" > /tmp/chess-ai-image-tag
}

# Push Docker image to ECR
push_image() {
    log_info "Pushing Docker image to ECR..."
    
    # Login to ECR
    aws ecr get-login-password --region "$AWS_REGION" | \
        docker login --username AWS --password-stdin \
        "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
    
    # Create repository if it doesn't exist
    aws ecr describe-repositories \
        --repository-names chess-ai-bot \
        --region "$AWS_REGION" &> /dev/null || \
    aws ecr create-repository \
        --repository-name chess-ai-bot \
        --region "$AWS_REGION" \
        --image-scanning-configuration scanOnPush=true
    
    # Push image
    local image_tag=$(cat /tmp/chess-ai-image-tag)
    docker push "$image_tag"
    
    log_info "Image pushed successfully"
}

# Run security scan
scan_image() {
    log_info "Running security scan..."
    
    local image_tag=$(cat /tmp/chess-ai-image-tag)
    
    # Wait for ECR scan to complete
    log_info "Waiting for ECR scan to complete..."
    sleep 30
    
    # Get scan results
    local findings=$(aws ecr describe-image-scan-findings \
        --repository-name chess-ai-bot \
        --image-id imageTag="$VERSION" \
        --region "$AWS_REGION" \
        --query 'imageScanFindings.findingSeverityCounts' \
        --output json)
    
    log_info "Scan results: $findings"
    
    # Check for critical vulnerabilities
    local critical=$(echo "$findings" | jq -r '.CRITICAL // 0')
    if [[ "$critical" -gt 0 ]]; then
        log_error "Found $critical critical vulnerabilities"
        if [[ "$ENVIRONMENT" == "prod" ]]; then
            log_error "Cannot deploy to production with critical vulnerabilities"
            exit 1
        else
            log_warn "Proceeding with deployment despite vulnerabilities (non-prod environment)"
        fi
    fi
}

# Create namespace if not exists
create_namespace() {
    log_info "Creating namespace if not exists..."
    
    kubectl create namespace chess-ai --dry-run=client -o yaml | kubectl apply -f -
    kubectl label namespace chess-ai environment="$ENVIRONMENT" --overwrite
    
    log_info "Namespace ready"
}

# Create secrets
create_secrets() {
    log_info "Creating secrets..."
    
    # Get database password from Secrets Manager
    local db_password=$(aws secretsmanager get-secret-value \
        --secret-id "chess-ai/${ENVIRONMENT}/database" \
        --query SecretString \
        --output text | jq -r .password)
    
    # Create Kubernetes secret
    kubectl create secret generic chess-ai-secrets \
        --from-literal=db-password="$db_password" \
        --namespace chess-ai \
        --dry-run=client -o yaml | kubectl apply -f -
    
    log_info "Secrets created"
}

# Deploy with Helm
deploy_helm() {
    log_info "Deploying application with Helm..."
    
    cd "$PROJECT_ROOT/infrastructure/helm"
    
    local image_tag=$(cat /tmp/chess-ai-image-tag)
    
    # Prepare values file
    cat > "chess-ai/values-${ENVIRONMENT}.yaml" <<EOF
image:
  repository: ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/chess-ai-bot
  tag: "$VERSION"

replicaCount: $([ "$ENVIRONMENT" = "prod" ] && echo "3" || echo "2")

resources:
  requests:
    cpu: $([ "$ENVIRONMENT" = "prod" ] && echo "500m" || echo "250m")
    memory: $([ "$ENVIRONMENT" = "prod" ] && echo "1Gi" || echo "512Mi")
  limits:
    cpu: $([ "$ENVIRONMENT" = "prod" ] && echo "2000m" || echo "1000m")
    memory: $([ "$ENVIRONMENT" = "prod" ] && echo "4Gi" || echo "2Gi")

autoscaling:
  enabled: $([ "$ENVIRONMENT" = "prod" ] && echo "true" || echo "false")
  minReplicas: $([ "$ENVIRONMENT" = "prod" ] && echo "3" || echo "2")
  maxReplicas: $([ "$ENVIRONMENT" = "prod" ] && echo "20" || echo "5")

ingress:
  enabled: true
  hosts:
    - host: chess-ai-${ENVIRONMENT}.example.com
      paths:
        - path: /
          pathType: Prefix

env:
  - name: ENVIRONMENT
    value: "$ENVIRONMENT"
  - name: APP_VERSION
    value: "$VERSION"
EOF
    
    # Deploy/upgrade with Helm
    helm upgrade --install chess-ai ./chess-ai \
        --namespace chess-ai \
        --values "chess-ai/values-${ENVIRONMENT}.yaml" \
        --wait \
        --timeout 10m \
        --atomic
    
    log_info "Helm deployment successful"
}

# Wait for rollout
wait_for_rollout() {
    log_info "Waiting for rollout to complete..."
    
    kubectl rollout status deployment/chess-ai \
        --namespace chess-ai \
        --timeout=10m
    
    log_info "Rollout completed successfully"
}

# Smoke tests
run_smoke_tests() {
    log_info "Running smoke tests..."
    
    # Get service endpoint
    local endpoint
    if [[ "$ENVIRONMENT" == "dev" ]]; then
        # Port forward for dev
        kubectl port-forward svc/chess-ai 8000:80 -n chess-ai &
        local port_forward_pid=$!
        sleep 5
        endpoint="http://localhost:8000"
    else
        endpoint="https://chess-ai-${ENVIRONMENT}.example.com"
    fi
    
    # Test health endpoint
    local health_status=$(curl -s "$endpoint/health" | jq -r .status)
    if [[ "$health_status" != "healthy" ]]; then
        log_error "Health check failed"
        [[ -n "${port_forward_pid:-}" ]] && kill "$port_forward_pid"
        exit 1
    fi
    log_info "✓ Health check passed"
    
    # Test readiness endpoint
    local ready_status=$(curl -s "$endpoint/health/ready" | jq -r .status)
    if [[ "$ready_status" != "ready" ]]; then
        log_error "Readiness check failed"
        [[ -n "${port_forward_pid:-}" ]] && kill "$port_forward_pid"
        exit 1
    fi
    log_info "✓ Readiness check passed"
    
    # Test move endpoint
    local move=$(curl -s -X POST "$endpoint/move" \
        -H "Content-Type: application/json" \
        -d '{"fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"}' | \
        jq -r .move)
    
    if [[ -z "$move" || "$move" == "null" ]]; then
        log_error "Move endpoint test failed"
        [[ -n "${port_forward_pid:-}" ]] && kill "$port_forward_pid"
        exit 1
    fi
    log_info "✓ Move endpoint test passed (move: $move)"
    
    # Clean up port forward
    [[ -n "${port_forward_pid:-}" ]] && kill "$port_forward_pid"
    
    log_info "All smoke tests passed"
}

# Update status page
update_status_page() {
    log_info "Updating status page..."
    
    # This would integrate with your status page provider
    # Example for Atlassian Statuspage:
    # curl -X POST "https://api.statuspage.io/v1/pages/PAGE_ID/incidents" \
    #   -H "Authorization: OAuth YOUR_API_KEY" \
    #   -d "incident[name]=Deployment to $ENVIRONMENT" \
    #   -d "incident[status]=resolved"
    
    log_info "Status page updated"
}

# Send notification
send_notification() {
    log_info "Sending deployment notification..."
    
    local message="✅ Chess AI Bot deployed to $ENVIRONMENT (version: $VERSION)"
    
    # Slack notification (if webhook URL is set)
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        curl -X POST "$SLACK_WEBHOOK_URL" \
            -H "Content-Type: application/json" \
            -d "{\"text\": \"$message\"}"
    fi
    
    log_info "Notification sent"
}

# Cleanup
cleanup() {
    log_info "Cleaning up..."
    
    rm -f /tmp/chess-ai-image-tag
    
    log_info "Cleanup completed"
}

# Main deployment flow
main() {
    log_info "Starting deployment..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Version: $VERSION"
    
    validate_environment
    check_prerequisites
    configure_aws
    configure_kubectl
    build_image
    push_image
    scan_image
    create_namespace
    create_secrets
    deploy_helm
    wait_for_rollout
    run_smoke_tests
    update_status_page
    send_notification
    cleanup
    
    log_info "🎉 Deployment completed successfully!"
    log_info "Application URL: https://chess-ai-${ENVIRONMENT}.example.com"
}

# Trap errors and cleanup
trap cleanup EXIT

# Run main
main
