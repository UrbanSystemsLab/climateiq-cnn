data "google_project" "project" {
}

# Place the source code for the cloud function into a GCS bucket.
data "archive_file" "source" {
  type        = "zip"
  output_path = "${path.module}/files/generate_feature_matrix_source.zip"
  source_dir  = "${path.module}/../cloud_functions/generate_feature_matrix/"
}

resource "google_storage_bucket" "source" {
  name     = "climateiq-cloud-functions"
  location = "us-west1"
}

resource "google_storage_bucket_object" "source" {
  name   = basename(data.archive_file.source.output_path)
  bucket = google_storage_bucket.source.name
  source = data.archive_file.source.output_path
}

# Create buckets for storing raw file chunks and processed feature matrix chunks.
resource "google_storage_bucket" "chunks" {
  name     = "climateiq-map-chunks"
  location = "us-west1"
}

resource "google_storage_bucket" "features" {
  name     = "climateiq-map-feature-chunks"
  location = "us-west1"
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

# Create a service account used by the function and Eventarc trigger
resource "google_service_account" "account" {
  account_id   = "gcf-sa"
  display_name = "generate-feature-matrix cloud function service account"
}

resource "google_project_iam_member" "invoking" {
  project    = data.google_project.project.project_id
  role       = "roles/run.invoker"
  member     = "serviceAccount:${google_service_account.account.email}"
  depends_on = [google_project_iam_member.gcs_pubsub_publishing]
}

resource "google_project_iam_member" "event_receiving" {
  project    = data.google_project.project.project_id
  role       = "roles/eventarc.eventReceiver"
  member     = "serviceAccount:${google_service_account.account.email}"
  depends_on = [google_project_iam_member.invoking]
}

resource "google_project_iam_member" "artifactregistry_reader" {
  project    = data.google_project.project.project_id
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${google_service_account.account.email}"
  depends_on = [google_project_iam_member.event_receiving]
}

# Give read access to the chunks and write access to the features buckets.
resource "google_storage_bucket_iam_member" "chunks_reader" {
  bucket = google_storage_bucket.chunks.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.account.email}"
}

resource "google_storage_bucket_iam_member" "features_writer" {
  bucket = google_storage_bucket.features.name
  role   = "roles/storage.objectUser"
  member = "serviceAccount:${google_service_account.account.email}"
}

# Create a function triggered by writes to the chunks bucket.
resource "google_cloudfunctions2_function" "function" {
  depends_on = [
    google_project_iam_member.gcs_pubsub_publishing,
  ]

  name        = "generate-feature-matrix"
  description = "Create a feature matrix from uploaded archives of geo files."
  location    = "us-west1"
  # location    = google_storage_bucket.chunks.location # The trigger must be in the same location as the bucket

  build_config {
    runtime     = "python311"
    entry_point = "build_feature_matrix"
    source {
      storage_source {
        bucket = google_storage_bucket.source.name
        object = google_storage_bucket_object.source.name
      }
    }
  }

  service_config {
    available_memory      = "2Gi"
    timeout_seconds       = 60
    service_account_email = google_service_account.account.email
  }

  event_trigger {
    # trigger_region        = google_storage_bucket.chunks.location # The trigger must be in the same location as the bucket
    trigger_region        = "us-west1"
    event_type            = "google.cloud.storage.object.v1.finalized"
    retry_policy          = "RETRY_POLICY_DO_NOT_RETRY"
    service_account_email = google_service_account.account.email
    event_filters {
      attribute = "bucket"
      value     = google_storage_bucket.chunks.name
    }
  }

  lifecycle {
    replace_triggered_by = [
      google_storage_bucket_object.source
    ]
  }
}
