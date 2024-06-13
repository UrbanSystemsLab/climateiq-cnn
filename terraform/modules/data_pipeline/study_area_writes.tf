# Create a bucket for the original, un-processed files describing a study area geography.
resource "google_storage_bucket" "study_areas" {
  name     = "${var.bucket_prefix}climateiq-study-areas"
  location = var.bucket_region
}

# Create a service account used by the function and Eventarc trigger
resource "google_service_account" "study_area_writer" {
  account_id   = "gcf-study-area-writer-sa"
  display_name = "write-study-area-metadata function service account"
}

# Grant permissions needed to trigger and run cloud functions.
resource "google_project_iam_member" "study_area_writer_invoking" {
  project    = data.google_project.project.project_id
  role       = "roles/run.invoker"
  member     = "serviceAccount:${google_service_account.study_area_writer.email}"
  depends_on = [google_project_iam_member.gcs_pubsub_publishing]
}

resource "google_project_iam_member" "study_area_writer_event_receiving" {
  project    = data.google_project.project.project_id
  role       = "roles/eventarc.eventReceiver"
  member     = "serviceAccount:${google_service_account.study_area_writer.email}"
  depends_on = [google_project_iam_member.invoking]
}

resource "google_project_iam_member" "study_area_writer_artifactregistry_reader" {
  project    = data.google_project.project.project_id
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${google_service_account.study_area_writer.email}"
  depends_on = [google_project_iam_member.event_receiving]
}

# Give write access to error reporter.
resource "google_project_iam_member" "study_area_writer_error_writer" {
  project = data.google_project.project.project_id
  role    = "roles/errorreporting.writer"
  member  = "serviceAccount:${google_service_account.study_area_writer.email}"
}

# Give write access to firestore.
resource "google_project_iam_member" "study_area_writer_firestore_writer" {
  project = data.google_project.project.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${google_service_account.study_area_writer.email}"
}

# Grant read access to the study area bucket.
resource "google_storage_bucket_iam_member" "study_area_reader" {
  bucket = google_storage_bucket.study_areas.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.study_area_writer.email}"
}


# Create a function triggered by writes to the study-areas bucket.
resource "google_cloudfunctions2_function" "study_area_write" {
  depends_on = [
    google_project_iam_member.gcs_pubsub_publishing,
  ]

  name        = "write-study-area-metadata"
  description = "Writes study area metadata from raw geo files."
  location    = lower(google_storage_bucket.study_areas.location) # The trigger must be in the same location as the bucket

  build_config {
    runtime     = "python311"
    entry_point = "write_study_area_metadata"
    source {
      storage_source {
        bucket = google_storage_bucket.source.name
        object = google_storage_bucket_object.source.name
      }
    }
  }

  service_config {
    available_memory      = "512Mi"
    timeout_seconds       = 60
    service_account_email = google_service_account.study_area_writer.email
    environment_variables = {
      BUCKET_PREFIX = var.bucket_prefix
    }
  }

  event_trigger {
    trigger_region        = lower(google_storage_bucket.study_areas.location) # The trigger must be in the same location as the bucket
    event_type            = "google.cloud.storage.object.v1.finalized"
    retry_policy          = "RETRY_POLICY_RETRY"
    service_account_email = google_service_account.study_area_writer.email
    event_filters {
      attribute = "bucket"
      value     = google_storage_bucket.study_areas.name
    }
  }

  lifecycle {
    replace_triggered_by = [
      google_storage_bucket_object.source
    ]
  }
}
