locals {
  # `aws_instance.inf2` is a list (count = 0 or 1).  `one(...)` returns
  # the single element if present, else null — so all downstream outputs
  # cleanly produce null when var.skip_persistent_instance = true.
  inf2 = one(aws_instance.inf2)
}

output "instance_id" {
  description = "EC2 instance ID (null when skip_persistent_instance = true)."
  value       = try(local.inf2.id, null)
}

output "public_ip" {
  description = "Public IP address of the Inf2 instance."
  value       = try(local.inf2.public_ip, null)
}

output "public_dns" {
  description = "Public DNS hostname."
  value       = try(local.inf2.public_dns, null)
}

output "instance_type" {
  description = "EC2 instance type that was launched."
  value       = try(local.inf2.instance_type, var.instance_type)
}

output "ami_id" {
  description = "AMI ID used for the instance."
  value       = data.aws_ami.neuron_dlami.id
}

output "ami_name" {
  description = "Friendly name of the DLAMI that was resolved."
  value       = data.aws_ami.neuron_dlami.name
}

output "private_key_path" {
  description = "Local path of the generated SSH private key."
  value       = pathexpand(var.private_key_path)
  sensitive   = true
}

output "ssh_command" {
  description = "Ready-to-paste SSH command (null when skip_persistent_instance = true)."
  value       = local.inf2 == null ? null : "ssh -i ${pathexpand(var.private_key_path)} ubuntu@${local.inf2.public_dns}"
}

output "setup_command" {
  description = "One-liner to bootstrap the instance after SSH-ing in."
  value       = local.inf2 == null ? null : "ssh -i ${pathexpand(var.private_key_path)} ubuntu@${local.inf2.public_dns} 'bash -s' < scripts/setup_inf2.sh"
}

output "ssm_command" {
  description = "AWS SSM Session Manager connect command (no open ports needed)."
  value       = local.inf2 == null ? null : "aws ssm start-session --target ${local.inf2.id} --region ${var.aws_region}"
}

# ---------------------------------------------------------------------------
# Benchmark-job outputs
# ---------------------------------------------------------------------------
output "aws_region" {
  description = "AWS region (echoed for scripts that need it)."
  value       = var.aws_region
}

output "bench_sns_topic_arn" {
  description = "SNS topic ARN that the bench job publishes results to."
  value       = aws_sns_topic.bench_results.arn
}

output "bench_s3_bucket" {
  description = "S3 bucket where benchmark logs are archived."
  value       = aws_s3_bucket.bench_results.bucket
}

output "bench_launch_template_id" {
  description = "Launch-template ID for ephemeral one-shot bench instances."
  value       = aws_launch_template.bench_ephemeral.id
}

output "bench_job_command" {
  description = "Local command to trigger a benchmark run."
  value       = var.skip_persistent_instance ? "make bench-job MODE=ephemeral NEURON=1" : "make bench-job NEURON=1"
}

output "persistent_instance" {
  description = "Whether the long-running instance was provisioned."
  value       = !var.skip_persistent_instance
}
