data "google_project" "project" {
}

# Place the source code for the cloud function into a GCS bucket.
data "archive_file" "source" {
  type        = "zip"
  output_path = "${path.module}/files/cloud_function_source.zip"

  # Add main.py to the root of the zip file.
  source {
    content  = file("{path.module}/../../../usl_pipeline/cloud_functions/main.py")
    filename = "main.py"
  }
  # Add requirements.txt to the root of the zip file.
  source {
    content  = file("{path.module}/../../../usl_pipeline/cloud_functions/requirements.txt")
    filename = "requirements.txt"
  }
  # Add all the contents of usl_lib to a usl_lib directory inside the zip file.
  dynamic "source" {
    for_each = fileset("{path.module}/../../../usl_pipeline/usl_lib/usl_lib/", "**/*.py")
    content {
      content  = file("{path.module}/../../../usl_pipeline/usl_lib/usl_lib/${source.value}")
      filename = "usl_lib/${source.value}"
    }
  }
}

resource "google_storage_bucket" "source" {
  name     = "${var.bucket_prefix}climateiq-cloud-functions"
  location = var.bucket_region
}

resource "google_storage_bucket_object" "source" {
  name   = basename(data.archive_file.source.output_path)
  bucket = google_storage_bucket.source.name
  source = data.archive_file.source.output_path
}

# Create a firestore database to use as our metastore.
# We only need one database, so we name it (default) as recommended:
# https://firebase.google.com/docs/firestore/manage-databases#the_default_database
resource "google_firestore_database" "database" {
  name        = "(default)"
  location_id = var.bucket_region
  type        = "FIRESTORE_NATIVE"
}

# To use GCS CloudEvent triggers, the GCS service account requires the Pub/Sub
# Publisher(roles/pubsub.publisher) IAM role in the specified project.
# (See https://cloud.google.com/eventarc/docs/run/quickstart-storage#before-you-begin)
data "google_storage_project_service_account" "gcs" {
}

resource "google_project_iam_member" "gcs_pubsub_publishing" {
  project = data.google_project.project.project_id
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:${data.google_storage_project_service_account.gcs.email_address}"
}
