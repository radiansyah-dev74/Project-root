# app/ai/llm_client.py

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)


def call_llm(prompt: str) -> str:
    """
    Call LLM via OpenRouter.
    Returns raw text output.
    """
    response = client.chat.completions.create(
        model="google/gemma-2-9b-it",
        messages=[
            {"role": "system", "content": "You are an AI evaluator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content
