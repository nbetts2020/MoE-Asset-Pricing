#!/bin/bash
set -e

apt-get update
apt-get install -y libcairo2-dev build-essential python3-dev

pip install --upgrade pip setuptools wheel

pip install peewee psycopg2 rmm-cu12

pip install -r requirements.txt

pip install flash_attn
