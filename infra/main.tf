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

  tags = { Name = "${var.project_tag}-inf2" }

  # Ensure the key is available before the instance starts
  depends_on = [aws_key_pair.key]
}
