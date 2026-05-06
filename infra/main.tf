provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_tag
      ManagedBy   = "opentofu"
      Environment = "experiment"
    }
  }
}

# ---------------------------------------------------------------------------
# AMI lookup — latest Deep Learning AMI Neuron (Ubuntu 22.04)
# ---------------------------------------------------------------------------
data "aws_ami" "neuron_dlami" {
  most_recent = true
  owners      = [var.ami_owner]

  filter {
    name   = "name"
    values = [var.ami_name_filter]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }

  filter {
    name   = "state"
    values = ["available"]
  }
}

# ---------------------------------------------------------------------------
# Networking — minimal public VPC
# ---------------------------------------------------------------------------
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = { Name = "${var.project_tag}-vpc" }
}

resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = "${var.project_tag}-igw" }
}

resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "${var.aws_region}a"
  map_public_ip_on_launch = true

  tags = { Name = "${var.project_tag}-public" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }

  tags = { Name = "${var.project_tag}-rt" }
}

resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

# ---------------------------------------------------------------------------
# Security group — SSH only (restrict var.allowed_cidr_blocks in production)
# ---------------------------------------------------------------------------
resource "aws_security_group" "ssh" {
  name        = "${var.project_tag}-ssh"
  description = "SSH access for lqcd-neuron development"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  egress {
    description = "All outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.project_tag}-sg" }
}

# ---------------------------------------------------------------------------
# SSH key pair — generated locally, public half uploaded to AWS
# ---------------------------------------------------------------------------
resource "tls_private_key" "key" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

resource "aws_key_pair" "key" {
  key_name   = var.key_name
  public_key = tls_private_key.key.public_key_openssh
}

resource "local_sensitive_file" "private_key" {
  content         = tls_private_key.key.private_key_pem
  filename        = pathexpand(var.private_key_path)
  file_permission = "0600"
}

# ---------------------------------------------------------------------------
# IAM instance profile — allows SSM Session Manager as an alternative to SSH
# ---------------------------------------------------------------------------
data "aws_iam_policy_document" "ec2_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "instance" {
  name               = "${var.project_tag}-instance-role"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.instance.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "instance" {
  name = "${var.project_tag}-instance-profile"
  role = aws_iam_role.instance.name
}

# ---------------------------------------------------------------------------
# EC2 instance (On-Demand or Spot)
# ---------------------------------------------------------------------------
locals {
  user_data = <<-EOF
    #!/bin/bash
    set -euo pipefail
    export INSTANCE_TYPE="${var.instance_type}"
    # Signal cloud-init that our setup script should run on next SSH login
    touch /var/run/lqcd-neuron-bootstrap-needed
  EOF
}

resource "aws_instance" "inf2" {
  count = var.skip_persistent_instance ? 0 : 1

  ami                    = data.aws_ami.neuron_dlami.id
  instance_type          = var.instance_type
  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.ssh.id]
  key_name               = aws_key_pair.key.key_name
  iam_instance_profile   = aws_iam_instance_profile.instance.name

  user_data = local.user_data

  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.root_volume_size_gb
    delete_on_termination = true
    encrypted             = true
  }

  # Spot configuration (only applied when var.spot = true)
  dynamic "instance_market_options" {
    for_each = var.spot ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        instance_interruption_behavior = "terminate"
      }
    }
  }

  tags = {
    Name             = "${var.project_tag}-inf2"
    BenchSnsTopicArn = aws_sns_topic.bench_results.arn
    BenchS3Bucket    = aws_s3_bucket.bench_results.bucket
  }

  # Ensure the key is available before the instance starts
  depends_on = [aws_key_pair.key]
}

# ===========================================================================
# Benchmark-job plumbing
#
# A benchmark run is fired off via SSM RunCommand (persistent mode) or via a
# one-shot Inf2 launched from a launch template (ephemeral mode).  Either
# way, the on-instance script:
#   - uploads the full log to s3://${aws_s3_bucket.bench_results}/runs/.../
#   - publishes a summary + presigned download URL to the SNS topic, which
#     fans out to the email address in var.notification_email.
# ===========================================================================

# ---------------------------------------------------------------------------
# SNS topic for benchmark-result emails
# ---------------------------------------------------------------------------
resource "aws_sns_topic" "bench_results" {
  name = "${var.project_tag}-bench-results"
  tags = { Name = "${var.project_tag}-bench-results" }
}

resource "aws_sns_topic_subscription" "email" {
  count     = var.notification_email == "" ? 0 : 1
  topic_arn = aws_sns_topic.bench_results.arn
  protocol  = "email"
  endpoint  = var.notification_email
  # NOTE: AWS sends a one-click confirmation link to this address. The
  # subscription stays in `PendingConfirmation` until the recipient clicks
  # it — Terraform cannot wait on that, so the first apply will report the
  # subscription as pending and that is expected.
}

# ---------------------------------------------------------------------------
# S3 bucket for benchmark artefacts (logs, CSVs, plots in future)
# ---------------------------------------------------------------------------
resource "random_id" "bench_bucket" {
  byte_length = 4
}

resource "aws_s3_bucket" "bench_results" {
  bucket        = "${var.project_tag}-bench-${random_id.bench_bucket.hex}"
  force_destroy = true
  tags          = { Name = "${var.project_tag}-bench-results" }
}

resource "aws_s3_bucket_public_access_block" "bench_results" {
  bucket                  = aws_s3_bucket.bench_results.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "bench_results" {
  bucket = aws_s3_bucket.bench_results.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_versioning" "bench_results" {
  bucket = aws_s3_bucket.bench_results.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "bench_results" {
  bucket = aws_s3_bucket.bench_results.id

  rule {
    id     = "expire-old-runs"
    status = "Enabled"

    filter { prefix = "runs/" }

    expiration {
      days = var.bench_log_retention_days
    }
    noncurrent_version_expiration {
      noncurrent_days = 7
    }
    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }
  }
}

# ---------------------------------------------------------------------------
# IAM policy: allow the instance role to publish to SNS and write to S3
# ---------------------------------------------------------------------------
data "aws_iam_policy_document" "bench_publish" {
  statement {
    sid       = "PublishBenchResults"
    actions   = ["sns:Publish"]
    resources = [aws_sns_topic.bench_results.arn]
  }
  statement {
    sid       = "UploadBenchArtefacts"
    actions   = ["s3:PutObject", "s3:PutObjectAcl", "s3:AbortMultipartUpload"]
    resources = ["${aws_s3_bucket.bench_results.arn}/*"]
  }
  statement {
    sid       = "ListBenchBucket"
    actions   = ["s3:ListBucket", "s3:GetBucketLocation"]
    resources = [aws_s3_bucket.bench_results.arn]
  }
  statement {
    sid       = "DescribeSelf"
    actions   = ["ec2:DescribeTags", "ec2:DescribeInstances"]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "bench_publish" {
  name   = "${var.project_tag}-bench-publish"
  policy = data.aws_iam_policy_document.bench_publish.json
}

resource "aws_iam_role_policy_attachment" "bench_publish" {
  role       = aws_iam_role.instance.name
  policy_arn = aws_iam_policy.bench_publish.arn
}

# ---------------------------------------------------------------------------
# Launch template — used by `make bench-job MODE=ephemeral` to launch a
# one-shot Inf2 that runs the benchmark, emails the results, then
# terminates itself (instance_initiated_shutdown_behavior=terminate).
#
# The user-data is rendered at trigger time by scripts/trigger_bench_job.sh
# so that the bench flags can vary per run; what we template here are only
# the pieces that are constant across runs (region, bucket, topic, branch).
# ---------------------------------------------------------------------------
resource "aws_launch_template" "bench_ephemeral" {
  name          = "${var.project_tag}-bench-ephemeral"
  image_id      = data.aws_ami.neuron_dlami.id
  instance_type = var.instance_type
  key_name      = aws_key_pair.key.key_name

  iam_instance_profile {
    name = aws_iam_instance_profile.instance.name
  }

  network_interfaces {
    associate_public_ip_address = true
    security_groups             = [aws_security_group.ssh.id]
    subnet_id                   = aws_subnet.public.id
  }

  block_device_mappings {
    device_name = "/dev/sda1"
    ebs {
      volume_type           = "gp3"
      volume_size           = var.root_volume_size_gb
      delete_on_termination = true
      encrypted             = true
    }
  }

  instance_initiated_shutdown_behavior = "terminate"

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name             = "${var.project_tag}-bench-ephemeral"
      BenchSnsTopicArn = aws_sns_topic.bench_results.arn
      BenchS3Bucket    = aws_s3_bucket.bench_results.bucket
      Lifecycle        = "ephemeral"
    }
  }
}
