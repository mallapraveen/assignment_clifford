import boto3, os
from langchain_aws import ChatBedrock
from constants import get_boto_client

bedrock_runtime = get_boto_client("bedrock-runtime")
print(bedrock_runtime)


def get_aws_chat_sonnet(temperature, streaming) -> ChatBedrock:
    model_kwargs = {
        "temperature": temperature,
    }

    return ChatBedrock(
        client=bedrock_runtime,
        model_kwargs=model_kwargs,
        model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        streaming=streaming,
    )


def get_aws_chat_opus(temperature, streaming) -> ChatBedrock:
    model_kwargs = {
        "temperature": temperature,
    }

    return ChatBedrock(
        client=bedrock_runtime,
        model_kwargs=model_kwargs,
        model_id="us.anthropic.claude-3-opus-20240229-v1:0",
        streaming=streaming,
    )


def get_aws_chat_haiku(temperature, streaming) -> ChatBedrock:
    model_kwargs = {
        "temperature": temperature,
    }

    return ChatBedrock(
        client=bedrock_runtime,
        model_kwargs=model_kwargs,
        model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",
        streaming=streaming,
    )
