from openai import OpenAI
from typing import Optional, List, Dict

def call_llm(
        url:str,
        api_key:str,
        model_name:str,
        prompt: str,
        system_prompt: Optional[str] =None,
        temperature: float = 0.7
) -> str:
    """Call the LLM API to get a response"""

    client = OpenAI(api_key=api_key, base_url=url)

    # Building a message list
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({
            "role":"assistant",
            "content":system_prompt
        })
    
    messages.append({
        "role": "user",
        "content": prompt
    })
    
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature= temperature,
        stream=False
    )

    return response.choices[0].message.content