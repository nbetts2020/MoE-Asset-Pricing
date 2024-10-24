#!/bin/bash
# Install system dependencies
apt-get update
apt-get install -y libcairo2-dev
# Now install Python dependencies
pip install -r requirements.txt
