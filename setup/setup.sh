#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Dynamically detect the package manager
if command -v yum > /dev/null; then
  PACKAGE_MANAGER="yum"
  INSTALL_CMD="sudo yum install -y"
  UPDATE_CMD="sudo yum update -y"
elif command -v apt-get > /dev/null; then
  PACKAGE_MANAGER="apt-get"
  INSTALL_CMD="sudo apt-get install -y"
  UPDATE_CMD="sudo apt-get update -y"
else
  echo "Neither yum nor apt-get is available on this system."
  exit 1
fi

echo "Using $PACKAGE_MANAGER as the package manager."

# Update and install necessary system dependencies
$UPDATE_CMD
$INSTALL_CMD cairo-devel  # for Amazon Linux (yum) or libcairo2-dev (apt-get)
$INSTALL_CMD python3-pip

# Upgrade pip3 and install Python dependencies
echo "Upgrading pip3..."
pip3 install --upgrade pip

echo "Installing Python dependencies..."
pip3 install -r requirements.txt
