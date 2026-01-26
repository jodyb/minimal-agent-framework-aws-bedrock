from __future__ import annotations

import os
from langchain_aws import ChatBedrock

class BedrockLLM:
    """Wrapper around ChatBedrock that returns string responses."""

    def __init__(self):
        self._llm = ChatBedrock(
            model_id=os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"),
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        )

    def invoke(self, prompt: str) -> str:
        response = self._llm.invoke(prompt)
        return response.content

def get_llm() -> BedrockLLM:
    return BedrockLLM()
