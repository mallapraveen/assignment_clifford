import sys
import gimme_aws_creds.main
import gimme_aws_creds.ui
import os
import boto3
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())


def get_boto_client(service):
    client = boto3.client(
        service_name=service,
        region_name="us-east-1",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
    )

    return client
