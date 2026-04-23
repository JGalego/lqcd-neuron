output "instance_id" {
  description = "EC2 instance ID."
  value       = aws_instance.inf2.id
}

output "public_ip" {
  description = "Public IP address of the Inf2 instance."
  value       = aws_instance.inf2.public_ip
}

output "public_dns" {
  description = "Public DNS hostname."
  value       = aws_instance.inf2.public_dns
}

output "instance_type" {
  description = "EC2 instance type that was launched."
  value       = aws_instance.inf2.instance_type
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
  description = "Ready-to-paste SSH command."
  value       = "ssh -i ${pathexpand(var.private_key_path)} ubuntu@${aws_instance.inf2.public_dns}"
}

output "setup_command" {
  description = "One-liner to bootstrap the instance after SSH-ing in."
  value       = "ssh -i ${pathexpand(var.private_key_path)} ubuntu@${aws_instance.inf2.public_dns} 'bash -s' < scripts/setup_inf2.sh"
}

output "ssm_command" {
  description = "AWS SSM Session Manager connect command (no open ports needed)."
  value       = "aws ssm start-session --target ${aws_instance.inf2.id} --region ${var.aws_region}"
}
