"""
llm.py - LLM client for AWS Bedrock.

This module provides a simplified interface to Claude models hosted on AWS Bedrock.
It wraps LangChain's ChatBedrock client to return plain strings instead of message objects.

CONFIGURATION:
==============
Set these environment variables to customize the LLM:

    BEDROCK_MODEL_ID     - The Bedrock model identifier
                           Default: "anthropic.claude-3-haiku-20240307-v1:0"
                           Examples:
                             - "anthropic.claude-3-haiku-20240307-v1:0"  (fast, cheap)
                             - "anthropic.claude-3-sonnet-20240229-v1:0" (balanced)
                             - "anthropic.claude-3-opus-20240229-v1:0"   (most capable)

    AWS_DEFAULT_REGION   - AWS region where Bedrock is available
                           Default: "us-east-1"

AWS CREDENTIALS:
================
Bedrock requires valid AWS credentials. These are read automatically from:
  - Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
  - ~/.aws/credentials file
  - IAM role (if running on AWS infrastructure)

Make sure your credentials have bedrock:InvokeModel permission.
"""

import os
from langchain_aws import ChatBedrock  # LangChain's AWS Bedrock integration


class BedrockLLM:
    """
    Wrapper around ChatBedrock that returns plain string responses.

    LangChain's ChatBedrock returns AIMessage objects, but most of our code
    just needs the text content. This wrapper simplifies the interface.

    Usage:
        llm = BedrockLLM()
        response = llm.invoke("What is 2 + 2?")
        print(response)  # "4" (plain string)
    """

    def __init__(self):
        """
        Initialize the Bedrock client.

        Reads model ID and region from environment variables,
        falling back to defaults if not set.
        """
        self._llm = ChatBedrock(
            # Model ID determines which Claude model to use
            # See AWS Bedrock docs for available model IDs
            model_id=os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"),
            # AWS region - Bedrock availability varies by region
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        )

    def invoke(self, prompt: str) -> str:
        """
        Send a prompt to the LLM and return the response text.

        Args:
            prompt: The text prompt to send to the model.

        Returns:
            The model's response as a plain string.
        """
        # ChatBedrock.invoke() returns an AIMessage object
        response = self._llm.invoke(prompt)
        # Extract just the text content from the message
        return response.content


def get_llm() -> BedrockLLM:
    """
    Factory function to create a BedrockLLM instance.

    Use this instead of directly instantiating BedrockLLM to allow
    for future enhancements like caching, connection pooling, or
    swapping implementations.

    Returns:
        A configured BedrockLLM instance ready to use.
    """
    return BedrockLLM()
