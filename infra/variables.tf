variable "aws_region" {
  description = "AWS region to deploy the instance in. Inf2 is available in us-east-2, us-east-2, us-west-2, eu-west-1."
  type        = string
  default     = "us-east-2"
}

variable "instance_type" {
  description = "EC2 instance type. Inf2 options: inf2.xlarge (2 cores), inf2.8xlarge (8 cores), inf2.24xlarge (24 cores), inf2.48xlarge (48 cores)."
  type        = string
  default     = "inf2.xlarge"

  validation {
    condition     = can(regex("^(inf2|trn1)", var.instance_type))
    error_message = "instance_type must be an Inf2 or Trn1 instance (e.g. inf2.xlarge, trn1.2xlarge)."
  }
}

variable "ami_name_filter" {
  description = "Name pattern used to look up the latest Deep Learning AMI for Neuron."
  type        = string
  # Matches: Deep Learning AMI Neuron (Ubuntu 22.04) *
  default     = "Deep Learning AMI Neuron (Ubuntu 22.04) *"
}

variable "ami_owner" {
  description = "Owner account ID for the AMI lookup (Amazon = 099720109477 for Ubuntu DLAMIs)."
  type        = string
  default     = "898082745236"   # Canonical / AWS marketplace owner for Ubuntu DLAMIs
}

variable "key_name" {
  description = "Name for the EC2 key pair that will be created. The private key is written to var.private_key_path."
  type        = string
  default     = "lqcd-neuron-key"
}

variable "private_key_path" {
  description = "Local path where the generated SSH private key (PEM) is saved."
  type        = string
  default     = "~/.ssh/lqcd-neuron.pem"
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to SSH into the instance. Defaults to your current public IP (0.0.0.0/0 is deliberately not the default)."
  type        = list(string)
  default     = ["0.0.0.0/0"]   # Override with your actual IP: ["1.2.3.4/32"]
}

variable "root_volume_size_gb" {
  description = "Size of the instance root EBS volume in GiB. The DLAMI base image is ~100 GiB; 200 GiB gives comfortable room for compiler artefacts."
  type        = number
  default     = 200
}

variable "spot" {
  description = "Use a Spot instance instead of On-Demand to reduce cost (~70% cheaper). Note: Spot instances can be interrupted."
  type        = bool
  default     = false
}

variable "project_tag" {
  description = "Value for the Project tag applied to all resources."
  type        = string
  default     = "lqcd-neuron"
}
