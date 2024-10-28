#!/bin/bash
set -e

export DEBIAN_FRONTEND=noninteractive

sudo apt-get update

sudo apt-get install -y \
    libcairo2-dev \
    build-essential \
    python3-dev \
    python3-pip \
    python3-venv

if ! command -v pip3 &> /dev/null
then
    echo "pip3 could not be found, installing pip..."
    sudo apt-get install -y python3-pip
fi

pip3 install --upgrade pip setuptools wheel

pip3 install peewee psycopg2-binary rmm-cu12

if [ -f requirements.txt ]; then
    pip3 install -r requirements.txt
else
    echo "requirements.txt not found. Skipping this step."
fi

pip3 install flash_attn

echo "Setup and upload completed successfully."
