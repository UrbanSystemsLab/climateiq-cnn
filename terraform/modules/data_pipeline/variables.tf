variable "bucket_prefix" {
  description = "Prefix to be appended to all GCS buckets."
  type        = string
}

variable "bucket_region" {
  description = "Region in which to create all GCS buckets."
  type        = string
  default     = "us-central1"
}
