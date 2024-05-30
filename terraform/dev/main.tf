terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "5.23.0"
    }
  }
}

provider "google" {
  project = "climateiq-test"
}

module "data_pipeline" {
  source        = "../modules/data_pipeline"
  bucket_prefix = "test-"
}

resource "google_storage_bucket" "tf_state" {
  name     = "test-climateiq-state"
  location = "us-west1"

  versioning {
    enabled = true
  }
  lifecycle {
    prevent_destroy = true
  }
}

terraform {
  backend "gcs" {
    bucket = "test-climateiq-state"
    prefix = "terraform/state"
  }
}
