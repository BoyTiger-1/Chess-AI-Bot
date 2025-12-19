# Terraform infrastructure (Business AI Assistant)

This directory provides **infrastructure-as-code scaffolding** for deploying the Business AI Assistant on AWS, GCP, or Azure.

The intent of these modules is to:

- Provide a sane, secure-by-default starting point
- Keep cloud-provider concerns isolated in provider-specific module wrappers
- Encourage a consistent pattern across environments (dev/staging/production)

> Note: These modules are intentionally lightweight wrappers around well-maintained community modules.
> You should pin module versions and review IAM permissions before production use.

## Layout

```
infra/terraform/
  aws/
    modules/
      network/          # VPC + subnets + routing
      kubernetes/       # EKS
      db/               # RDS (PostgreSQL)
      cache/            # ElastiCache (Redis)
      storage/          # S3 buckets (+ optional CloudFront)
      observability/    # CloudWatch logs/alarms (scaffold)
  gcp/
    modules/...
  azure/
    modules/...
  examples/
    aws-eks/            # Example composition of AWS modules
```

## State

For production, use a remote backend (e.g. S3 + DynamoDB lock, GCS, or Azure Storage) and separate state per environment.

## Example

See `infra/terraform/examples/aws-eks/`.
