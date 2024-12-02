#!/bin/bash

# Variables
VM_PASSWORD="sfc"  # Replace with your VM password
REQUIREMENTS_FILE="requirements.txt"

# Ensure the script is executable (self-permissioning)
if [ ! -x "$0" ]; then
    echo "Adding executable permissions to the script."
    chmod +x "$0"
fi

# Check for requirements.txt
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "ERROR: requirements.txt not found in the current directory."
    exit 1
fi

# Install Python and pip
echo "$VM_PASSWORD" | sudo -S apt update
echo "$VM_PASSWORD" | sudo -S apt install -y python3 python3-pip

# Install virtualenv
echo "$VM_PASSWORD" | sudo -S pip3 install virtualenv

# Create a virtual environment
virtualenv venv

# Activate the virtual environment
source venv/bin/activate

# Install libraries from requirements.txt
pip install -r requirements.txt

# Confirm installation
echo "Python and required libraries have been successfully installed in the virtual environment."
