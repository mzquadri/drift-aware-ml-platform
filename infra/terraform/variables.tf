variable "project_id" {
  description = "Google Cloud project that will own these resources"
  type        = string
}

variable "region" {
  description = "Region for Cloud Run, the bucket and the image registry"
  type        = string
  default     = "europe-west3" # Frankfurt, so data stays in the EU
}

variable "image" {
  description = "Fully qualified container image for the prediction service"
  type        = string
}

variable "model_name" {
  description = "Registered model name the service loads from MLflow"
  type        = string
  default     = "bike-demand"
}

variable "mlflow_tracking_uri" {
  description = "MLflow tracking server the service reads the champion from"
  type        = string
}

variable "min_instances" {
  description = "Set to 0 so an idle deployment costs nothing"
  type        = number
  default     = 0
}

variable "max_instances" {
  description = "Upper bound on concurrent instances, so a traffic spike cannot run up a bill"
  type        = number
  default     = 3
}

variable "cpu" {
  type    = string
  default = "1"
}

variable "memory" {
  description = "A tree ensemble plus its preprocessing sits comfortably in this"
  type        = string
  default     = "1Gi"
}

variable "allow_public" {
  description = "Whether unauthenticated callers may invoke the service"
  type        = bool
  default     = false
}
