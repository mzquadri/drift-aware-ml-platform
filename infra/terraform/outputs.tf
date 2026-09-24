output "service_url" {
  description = "Base URL of the deployed prediction service"
  value       = google_cloud_run_v2_service.service.uri
}

output "artifacts_bucket" {
  description = "Bucket holding model bundles and drift reports"
  value       = google_storage_bucket.artifacts.name
}

output "image_repository" {
  description = "Artifact Registry path to push service images to"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}"
}

output "service_account" {
  description = "Runtime identity of the service"
  value       = google_service_account.service.email
}
