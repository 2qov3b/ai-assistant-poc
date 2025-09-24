from openai import OpenAI
from typing import Optional, List, Dict, Any

def call_llm_docs(
        docs:List[Any],
        query:str,
        url:str,
        api_key:str,
        model_name:str,
)->str:
    #Deepseek client
    llm = ChatDeepSeek(model=model_name, api_key = api_key, base_url = url)
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    #Submit the matched documents and user questions to DeepSeek for polishing
    response = chain.run(input_documents = docs, question = query)
    return response

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