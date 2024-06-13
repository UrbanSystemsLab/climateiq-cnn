# Create buckets for storing raw files, file chunks and processed feature matrix chunks.
resource "google_storage_bucket" "city_cat_config" {
  name     = "${var.bucket_prefix}climateiq-flood-simulation-config"
  location = var.bucket_region
}

# Create a service account used by the function and Eventarc trigger
resource "google_service_account" "city_cat_config" {
  account_id   = "gcf-city-cat-sa"
  display_name = "record-citycat-config cloud function service account"
}

# Grant permissions needed to trigger and run cloud functions.
resource "google_project_iam_member" "city_cat_config_invoking" {
  project    = data.google_project.project.project_id
  role       = "roles/run.invoker"
  member     = "serviceAccount:${google_service_account.city_cat_config.email}"
  depends_on = [google_project_iam_member.gcs_pubsub_publishing]
}

resource "google_project_iam_member" "city_cat_config_event_receiving" {
  project    = data.google_project.project.project_id
  role       = "roles/eventarc.eventReceiver"
  member     = "serviceAccount:${google_service_account.city_cat_config.email}"
  depends_on = [google_project_iam_member.invoking]
}

resource "google_project_iam_member" "city_cat_config_artifactregistry_reader" {
  project    = data.google_project.project.project_id
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${google_service_account.city_cat_config.email}"
  depends_on = [google_project_iam_member.event_receiving]
}

# Give read access to the citycat config bucket.
resource "google_storage_bucket_iam_member" "city_cat_config_reader" {
  bucket = google_storage_bucket.city_cat_config.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.city_cat_config.email}"
}

resource "google_storage_bucket_iam_member" "city_cat_features_writer" {
  bucket = google_storage_bucket.features.name
  role   = "roles/storage.objectUser"
  member = "serviceAccount:${google_service_account.city_cat_config.email}"
}

# Give write access to error reporter.
resource "google_project_iam_member" "city_cat_config_error_writer" {
  project = data.google_project.project.project_id
  role    = "roles/errorreporting.writer"
  member  = "serviceAccount:${google_service_account.city_cat_config.email}"
}

# Give write access to firestore.
resource "google_project_iam_member" "city_cat_config_firestore_writer" {
  project = data.google_project.project.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${google_service_account.city_cat_config.email}"
}

# Create a function triggered by writes to the citycat config bucket.
resource "google_cloudfunctions2_function" "write_citycat_config" {
  depends_on = [
    google_project_iam_member.gcs_pubsub_publishing,
  ]

  name        = "write-citycat-config-data"
  description = "Writes CityCAT config artifacts."
  location    = lower(google_storage_bucket.city_cat_config.location) # The trigger must be in the same location as the bucket

  build_config {
    runtime     = "python311"
    entry_point = "write_flood_scenario_metadata_and_features"
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
    service_account_email = google_service_account.city_cat_config.email
    environment_variables = {
      BUCKET_PREFIX = var.bucket_prefix
    }
  }

  event_trigger {
    trigger_region        = lower(google_storage_bucket.city_cat_config.location) # The trigger must be in the same location as the bucket
    event_type            = "google.cloud.storage.object.v1.finalized"
    retry_policy          = "RETRY_POLICY_RETRY"
    service_account_email = google_service_account.city_cat_config.email
    event_filters {
      attribute = "bucket"
      value     = google_storage_bucket.city_cat_config.name
    }
  }

  lifecycle {
    replace_triggered_by = [
      google_storage_bucket_object.source
    ]
  }
}


# Create a function triggered by deletes to the citycat config bucket.
resource "google_cloudfunctions2_function" "delete_citycat_config" {
  depends_on = [
    google_project_iam_member.gcs_pubsub_publishing,
  ]

  name        = "delete-citycat-config-metadata"
  description = "Deletes CityCAT config metadata from the metastore."
  location    = lower(google_storage_bucket.city_cat_config.location) # The trigger must be in the same location as the bucket

  build_config {
    runtime     = "python311"
    entry_point = "delete_flood_scenario_metadata"
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
    service_account_email = google_service_account.city_cat_config.email
    environment_variables = {
      BUCKET_PREFIX = var.bucket_prefix
    }
  }

  event_trigger {
    trigger_region        = lower(google_storage_bucket.city_cat_config.location) # The trigger must be in the same location as the bucket
    event_type            = "google.cloud.storage.object.v1.deleted"
    retry_policy          = "RETRY_POLICY_RETRY"
    service_account_email = google_service_account.city_cat_config.email
    event_filters {
      attribute = "bucket"
      value     = google_storage_bucket.city_cat_config.name
    }
  }

  lifecycle {
    replace_triggered_by = [
      google_storage_bucket_object.source
    ]
  }
}
