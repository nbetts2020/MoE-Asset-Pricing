# utils/termination_listener.py

import os
import time
import requests
from dotenv import load_dotenv
import boto3

# Load environment variables from .env
load_dotenv()

BUCKET_NAME = os.getenv('BUCKET_NAME')
CHECKPOINT_DIR = os.getenv('CHECKPOINT_DIR')

s3_client = boto3.client('s3')

def upload_checkpoint():
    # Find the latest checkpoint
    checkpoints = sorted(
        [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pt')],
        key=lambda x: os.path.getmtime(os.path.join(CHECKPOINT_DIR, x)),
        reverse=True
    )
    if checkpoints:
        latest_checkpoint = checkpoints[0]
        local_path = os.path.join(CHECKPOINT_DIR, latest_checkpoint)
        s3_path = f'checkpoints/{latest_checkpoint}'
        try:
            s3_client.upload_file(local_path, BUCKET_NAME, s3_path)
            print(f"Uploaded {latest_checkpoint} to s3://{BUCKET_NAME}/{s3_path}")
        except Exception as e:
            print(f"Failed to upload checkpoint: {e}")
    else:
        print("No checkpoints found to upload.")

def listen_for_termination():
    termination_url = "http://169.254.169.254/latest/meta-data/spot/termination-time"
    while True:
        try:
            response = requests.get(termination_url, timeout=1)
            if response.status_code == 200:
                print("Termination notice received. Uploading checkpoint...")
                upload_checkpoint()
                break
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)  # poll every 2 seconds

if __name__ == "__main__":
    listen_for_termination()
