resource "google_storage_bucket" "chunks" {
  name     = "${var.bucket_prefix}climateiq-study-area-chunks"
  location = var.bucket_region
}

resource "google_storage_bucket" "features" {
  name     = "${var.bucket_prefix}climateiq-study-area-feature-chunks"
  location = var.bucket_region
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
    available_memory      = "4Gi"
    timeout_seconds       = 60
    service_account_email = google_service_account.generate_feature_matrix.email
    environment_variables = {
      BUCKET_PREFIX = var.bucket_prefix
    }
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
