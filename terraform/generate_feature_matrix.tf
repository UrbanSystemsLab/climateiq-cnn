data "google_project" "project" {
}

# Place the source code for the cloud function into a GCS bucket.
data "archive_file" "source" {
  type        = "zip"
  source_dir  = "${path.module}/files/cloud_function_source"
  output_path = "${path.module}/files/cloud_function_source.zip"
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

resource "google_storage_bucket" "chunks" {
  name     = "climateiq-study-area-chunks"
  location = "us-west1"
}

resource "google_storage_bucket" "features" {
  name     = "climateiq-study-area-feature-chunks"
  location = "us-west1"
}

# Create a firestore database to use as our metastore.
# We only need one database, so we name it (default) as recommended:
# https://firebase.google.com/docs/firestore/manage-databases#the_default_database
resource "google_firestore_database" "database" {
  name        = "(default)"
  location_id = "us-west1"
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

# Create a service account used by the function and Eventarc trigger
resource "google_service_account" "generate_feature_matrix" {
  account_id   = "gcf-sa"
  display_name = "generate-feature-matrix cloud function service account"
}

# Grant permissions needed to trigger and run cloud functions.
resource "google_project_iam_member" "invoking" {
  project    = data.google_project.project.project_id
  role       = "roles/run.invoker"
  member     = "serviceAccount:${google_service_account.generate_feature_matrix.email}"
  depends_on = [google_project_iam_member.gcs_pubsub_publishing]
}

resource "google_project_iam_member" "event_receiving" {
  project    = data.google_project.project.project_id
  role       = "roles/eventarc.eventReceiver"
  member     = "serviceAccount:${google_service_account.generate_feature_matrix.email}"
  depends_on = [google_project_iam_member.invoking]
}

resource "google_project_iam_member" "artifactregistry_reader" {
  project    = data.google_project.project.project_id
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${google_service_account.generate_feature_matrix.email}"
  depends_on = [google_project_iam_member.event_receiving]
}

# Give read access to the chunks and write access to the features buckets.
resource "google_storage_bucket_iam_member" "chunks_reader" {
  bucket = google_storage_bucket.chunks.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.generate_feature_matrix.email}"
}

resource "google_storage_bucket_iam_member" "features_writer" {
  bucket = google_storage_bucket.features.name
  role   = "roles/storage.objectUser"
  member = "serviceAccount:${google_service_account.generate_feature_matrix.email}"
}

# Give write access to error reporter.
resource "google_project_iam_member" "error_writer" {
  project = data.google_project.project.project_id
  role    = "roles/errorreporting.writer"
  member  = "serviceAccount:${google_service_account.generate_feature_matrix.email}"
}

# Give write access to firestore.
resource "google_project_iam_member" "firestore_writer" {
  project = data.google_project.project.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${google_service_account.generate_feature_matrix.email}"
}

# Create a function triggered by writes to the chunks bucket.
resource "google_cloudfunctions2_function" "chunk_writes" {
  depends_on = [
    google_project_iam_member.gcs_pubsub_publishing,
  ]

  name        = "generate-feature-matrix"
  description = "Create a feature matrix from uploaded archives of geo files."
  location    = lower(google_storage_bucket.chunks.location) # The trigger must be in the same location as the bucket

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
    available_memory      = "256M"
    timeout_seconds       = 60
    service_account_email = google_service_account.generate_feature_matrix.email
  }

  event_trigger {
    trigger_region        = lower(google_storage_bucket.chunks.location) # The trigger must be in the same location as the bucket
    event_type            = "google.cloud.storage.object.v1.finalized"
    retry_policy          = "RETRY_POLICY_RETRY"
    service_account_email = google_service_account.generate_feature_matrix.email
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
