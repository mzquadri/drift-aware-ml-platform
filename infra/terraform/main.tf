// The service on Cloud Run, with a bucket for artefacts and a registry for
// images. Small on purpose: the point is that the deployment is described in
// code and reproducible, not that it is large.

terraform {
  required_version = ">= 1.6"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

locals {
  name = "drift-aware-ml-platform"
  labels = {
    app        = local.name
    managed-by = "terraform"
  }
}

resource "google_artifact_registry_repository" "images" {
  location      = var.region
  repository_id = local.name
  format        = "DOCKER"
  description   = "Container images for the demand service"
  labels        = local.labels
}

resource "google_storage_bucket" "artifacts" {
  name          = "${var.project_id}-${local.name}-artifacts"
  location      = var.region
  force_destroy = false
  labels        = local.labels

  // Model bundles and drift reports are reproducible, but keeping versions
  // means a bad promotion can be traced rather than guessed at.
  versioning {
    enabled = true
  }

  uniform_bucket_level_access = true

  lifecycle_rule {
    condition {
      age                = 90
      with_state         = "ARCHIVED"
      num_newer_versions = 5
    }
    action {
      type = "Delete"
    }
  }
}

// A dedicated identity rather than the default compute account, which has far
// more than a prediction service needs.
resource "google_service_account" "service" {
  account_id   = "${local.name}-sa"
  display_name = "Demand service runtime identity"
}

resource "google_storage_bucket_iam_member" "artifacts_rw" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.service.email}"
}

resource "google_cloud_run_v2_service" "service" {
  name     = local.name
  location = var.region
  labels   = local.labels

  template {
    service_account = google_service_account.service.email

    scaling {
      // Scales to zero when idle, which is what keeps a portfolio deployment
      // inside the free tier.
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }

    containers {
      image = var.image

      resources {
        limits = {
          cpu    = var.cpu
          memory = var.memory
        }
      }

      env {
        name  = "MLFLOW_TRACKING_URI"
        value = var.mlflow_tracking_uri
      }

      env {
        name  = "MODEL_NAME"
        value = var.model_name
      }

      env {
        name  = "MODEL_STAGE"
        value = "Production"
      }

      env {
        name  = "PREDICTION_LOG"
        value = "/tmp/predictions.parquet"
      }

      ports {
        container_port = 8000
      }

      // Liveness restarts a wedged process. Readiness gates traffic, and is the
      // one that knows whether a model is actually loaded.
      liveness_probe {
        http_get {
          path = "/health"
        }
        initial_delay_seconds = 20
        period_seconds        = 30
        failure_threshold     = 3
      }

      startup_probe {
        http_get {
          path = "/ready"
        }
        initial_delay_seconds = 10
        period_seconds        = 5
        failure_threshold     = 30
      }
    }
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }
}

resource "google_cloud_run_v2_service_iam_member" "public" {
  count    = var.allow_public ? 1 : 0
  name     = google_cloud_run_v2_service.service.name
  location = google_cloud_run_v2_service.service.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}
