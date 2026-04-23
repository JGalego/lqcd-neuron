terraform {
  required_version = ">= 1.6"   # OpenTofu 1.6+

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.2"
    }
    local = {
      source  = "hashicorp/local"
      version = "~> 2.8"
    }
  }

  # Optional: store state remotely (uncomment and fill in)
  # backend "s3" {
  #   bucket         = "my-tofu-state"
  #   key            = "lqcd-neuron/inf2/terraform.tfstate"
  #   region         = "us-east-2"
  #   encrypt        = true
  #   dynamodb_table = "tofu-state-lock"
  # }
}
