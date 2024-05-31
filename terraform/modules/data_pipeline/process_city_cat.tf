# Bucket for raw CityCAT output.
resource "google_storage_bucket" "raw_labels" {
  name     = "${var.bucket_prefix}climateiq-flood-simulation-output"
  location = var.bucket_region
}

# Bucket for processed labels.
resource "google_storage_bucket" "processed_labels" {
  name     = "${var.bucket_prefix}climateiq-study-area-label-chunks"
  location = var.bucket_region
}

# Create a service account used by the function and Eventarc trigger
resource "google_service_account" "labels" {
  account_id   = "gcf-label-processing-sa"
  display_name = "Used By Label Processing Cloud Function"
}

# Grant permissions needed for the cloud function to run and receive event triggers.
resource "google_project_iam_member" "labels_invoking" {
  project    = data.google_project.project.project_id
  role       = "roles/run.invoker"
  member     = "serviceAccount:${google_service_account.labels.email}"
  depends_on = [google_project_iam_member.gcs_pubsub_publishing]
}

resource "google_project_iam_member" "labels_event_receiving" {
  project    = data.google_project.project.project_id
  role       = "roles/eventarc.eventReceiver"
  member     = "serviceAccount:${google_service_account.labels.email}"
  depends_on = [google_project_iam_member.invoking]
}

resource "google_project_iam_member" "labels_artifactregistry_reader" {
  project    = data.google_project.project.project_id
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${google_service_account.labels.email}"
  depends_on = [google_project_iam_member.event_receiving]
}

# Give read & write access to the raw citycat bucket.
resource "google_storage_bucket_iam_member" "labels_reader" {
  bucket = google_storage_bucket.raw_labels.name
  role   = "roles/storage.objectUser"
  member = "serviceAccount:${google_service_account.labels.email}"
}

# Give write access to the processed labels bucket.
resource "google_storage_bucket_iam_member" "processed_labels_writer" {
  bucket = google_storage_bucket.processed_labels.name
  role   = "roles/storage.objectUser"
  member = "serviceAccount:${google_service_account.labels.email}"
}

# Give write access to error reporter.
resource "google_project_iam_member" "labels_error_writer" {
  project = data.google_project.project.project_id
  role    = "roles/errorreporting.writer"
  member  = "serviceAccount:${google_service_account.labels.email}"
}

# Give write access to firestore.
resource "google_project_iam_member" "labels_firestore_writer" {
  project = data.google_project.project.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${google_service_account.labels.email}"
}

# Create a function triggered by writes to the citycat config bucket.
resource "google_cloudfunctions2_function" "labels_processor" {
  depends_on = [
    google_project_iam_member.gcs_pubsub_publishing,
  ]

  name        = "write-processed-citycat-labels"
  description = "Processes CityCAT outputs."
  location    = lower(google_storage_bucket.raw_labels.location) # The trigger must be in the same location as the bucket

  build_config {
    runtime     = "python311"
    entry_point = "process_citycat_outputs"
    source {
      storage_source {
        bucket = google_storage_bucket.source.name
        object = google_storage_bucket_object.source.name
      }
    }
  }

  service_config {
    available_memory      = "4Gi"
    timeout_seconds       = 540
    service_account_email = google_service_account.labels.email
    environment_variables = {
      BUCKET_PREFIX = var.bucket_prefix
    }
  }

  event_trigger {
    trigger_region        = lower(google_storage_bucket.raw_labels.location) # The trigger must be in the same location as the bucket
    event_type            = "google.cloud.storage.object.v1.finalized"
    retry_policy          = "RETRY_POLICY_RETRY"
    service_account_email = google_service_account.labels.email
    event_filters {
      attribute = "bucket"
      value     = google_storage_bucket.raw_labels.name
    }
  }

  lifecycle {
    replace_triggered_by = [
      google_storage_bucket_object.source
    ]
  }
}
