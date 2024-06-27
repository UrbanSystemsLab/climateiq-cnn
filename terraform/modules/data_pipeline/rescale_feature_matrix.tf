resource "google_storage_bucket" "features" {
  name     = "${var.bucket_prefix}climateiq-study-area-feature-chunks"
  location = var.bucket_region
}

# Create a service account used by the function and Eventarc trigger
resource "google_service_account" "rescale_feature_matrix" {
  account_id   = "gcf-sa"
  display_name = "rescale-feature-matrix cloud function service account"
}

# Grant permissions needed to trigger and run cloud functions.
resource "google_project_iam_member" "invoking" {
  project    = data.google_project.project.project_id
  role       = "roles/run.invoker"
  member     = "serviceAccount:${google_service_account.rescale_feature_matrix.email}"
  depends_on = [google_project_iam_member.gcs_pubsub_publishing]
}

resource "google_project_iam_member" "event_receiving" {
  project    = data.google_project.project.project_id
  role       = "roles/eventarc.eventReceiver"
  member     = "serviceAccount:${google_service_account.rescale_feature_matrix.email}"
  depends_on = [google_project_iam_member.invoking]
}

resource "google_project_iam_member" "artifactregistry_reader" {
  project    = data.google_project.project.project_id
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${google_service_account.rescale_feature_matrix.email}"
  depends_on = [google_project_iam_member.event_receiving]
}

# Give read and write access to the features buckets.
resource "google_storage_bucket_iam_member" "features_reader" {
  bucket = google_storage_bucket.features.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.rescale_feature_matrix.email}"
}

resource "google_storage_bucket_iam_member" "features_writer" {
  bucket = google_storage_bucket.features.name
  role   = "roles/storage.objectUser"
  member = "serviceAccount:${google_service_account.rescale_feature_matrix.email}"
}

# Give write access to error reporter.
resource "google_project_iam_member" "error_writer" {
  project = data.google_project.project.project_id
  role    = "roles/errorreporting.writer"
  member  = "serviceAccount:${google_service_account.rescale_feature_matrix.email}"
}

# Give write access to firestore.
resource "google_project_iam_member" "firestore_writer" {
  project = data.google_project.project.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${google_service_account.rescale_feature_matrix.email}"
}

# Create a function triggered by writes to the features bucket.
resource "google_cloudfunctions2_function" "features_writes" {
  depends_on = [
    google_project_iam_member.gcs_pubsub_publishing,
  ]

  name        = "rescale-feature-matrix"
  description = "Create a scaled feature matrix from uploaded unscaled feature matrix files."
  location    = lower(google_storage_bucket.features.location) # The trigger must be in the same location as the bucket

  build_config {
    runtime     = "python311"
    entry_point = "rescale_feature_matrices"
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
    service_account_email = google_service_account.rescale_feature_matrix.email
    environment_variables = {
      BUCKET_PREFIX = var.bucket_prefix
    }
  }

  event_trigger {
    trigger_region        = lower(google_storage_bucket.features.location) # The trigger must be in the same location as the bucket
    event_type            = "google.cloud.storage.object.v1.finalized"
    retry_policy          = "RETRY_POLICY_RETRY"
    service_account_email = google_service_account.rescale_feature_matrix.email
    event_filters {
      attribute = "bucket"
      value     = google_storage_bucket.features.name
    }
  }

  lifecycle {
    replace_triggered_by = [
      google_storage_bucket_object.source
    ]
  }
}
