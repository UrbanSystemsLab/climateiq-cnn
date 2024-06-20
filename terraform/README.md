# GCP Data Pipeline Deployment with Terraform

## Initial Setup
- [Install Terraform](https://developer.hashicorp.com/terraform/install)
- [Install the gcloud CLI](https://cloud.google.com/sdk/docs/install)

From the root of this repository:
```bash
cd terraform/dev
terraform init
```

From the root of this repository:
```bash
cd terraform/prod
terraform init
```

You will need read & write permissions to the `climateiq-state` and
`test-climateiq-state` GCS buckets, as this is where
[Terraform state](https://developer.hashicorp.com/terraform/language/state)
is stored.

## Deployment
To deploy to staging, from the root of this repository:
```bash
cd terraform/dev
terraform apply
```

To deploy to production, perform the same steps but from the `prod` directory
```bash
cd terraform/prod
terraform apply
```
