from openai import OpenAI
from research_system.config.settings import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
    DEEPSEEK_BASE_URL,
)

_client = None


def get_llm() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            timeout=60.0,
        )
    return _client


def llm_call(
    system_prompt: str, user_prompt: str, temperature: float = 0.3
) -> str:
    client = get_llm()
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content
