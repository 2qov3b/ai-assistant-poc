from openai import OpenAI
from typing import Optional, List, Dict, Any
from langchain.chains.question_answering import load_qa_chain
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from function import check_order_status
import json

class IntentClassification(BaseModel):
    """Model of intent recognition classification results"""
    intent_type: str = Field(description="意図タイプ: knowledge|order|other")
    confidence: float = Field(description="意図分類の信頼性 0.0-1.0")

def classify_intent(
        url: str,
        api_key:str,
        model_name:str,
        message: str,
        role: str
)-> dict:
    
    #Setting the output parser
    #When the LLM parses the user's request (message), it generates Json output
    parser = JsonOutputParser(pydantic_object=IntentClassification)
    #Create LLM
    llm = ChatDeepSeek(model=model_name, api_key=api_key, base_url=url)
    #Prompt template
    prompt = PromptTemplate(
            template="""{role}として以下のユーザーからの質問の意図を分類してください：
            分類基準：
            1. 製品に関する質問（knowledge）：製品の使用方法やポリシーの条件などに関する、ドキュメントの取得が必要な質問
            2. 注文に関する質問（order）：注文状況や配送情報などに関する質問
            3. その他の質問（other）：手動によるカスタマーサービスが必要な複雑な質問 \n
            {format_instructions} \n
            ユーザーからの質問: {message} \n
            """,
            input_variables=["role", "message"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
    #Create a chain call Prompt → LLM → Paser Output
    chain = prompt | llm | parser
    #Execute call
    result = chain.invoke({"role":role, "message":message})

    return {
        "intent_type": result["intent_type"],
        "confidence": result["confidence"]
    }


def handle_order_query(
    url: str,
    api_key: str,
    model_name: str,
    message: str,
    role: str
    #check_order_status: callable
) -> str:
    """Processes order queries
    
    Args:
        url: API base URL
        api_key: API key
        model_name: Model name
        message: User message
        role: Customer service role description
        check_order_status: Function that checks order status
        
    Returns:
        str: Processing result
    """
    try:
        # Preparing the initial message
        messages = [
            {
                "role": "system",
                "content": role
            },
            {
                "role": "user",
                "content": message
            }
        ]
        
        # Get tool definition
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "check_order_status",
                    "description": "特定の注文のステータスを照会する",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {
                                "type": "string",
                                "description": "注文番号"
                            }
                        },
                        "required": ["order_id"]
                    }
                }
            }
        ]
        
        # The first call gets the possible tool calls
        response = call_llm_tools(
            url=url,
            api_key=api_key,
            model_name=model_name,
            messages=messages,
            tools=tools
        )
        
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        
        # Check if there is a tool call
        if tool_calls:
            messages.append(response_message)
            
            # Process each tool call
            for tool_call in tool_calls:
                if tool_call.function.name == "check_order_status":
                    function_args = json.loads(tool_call.function.arguments)
                    order_id = function_args.get("order_id")
                    
                    # Calling the function
                    order_info = check_order_status(order_id)
                    
                    #The result returned by the function
                    messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": "check_order_status",
                            "content": json.dumps(order_info)
                        }
                    )
            
            # The second call gets the final response
            second_response = call_llm_tools(
                url=url,
                api_key=api_key,
                model_name=model_name,
                messages=messages,
                tools=tools
            )
            
            return second_response.choices[0].message.content
        
        # If there is no tool call, return the response directly
        return response_message.content
    
    except Exception as e:
        print(f"注文クエリエラー: {str(e)}")
        return "注文追跡サービスは一時的に利用できません。しばらくしてからもう一度お試しください。"

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

def call_llm_tools(
    url: str,
    api_key: str,
    model_name: str,
    messages: List[Dict[str, str]],
    tools: List[Dict[str, Any]],
    tool_choice: str = "auto"
) -> Any:
    """
    LLM API that supports tool calls
    
    Args:
        url: API base URL
        api_key: API key
        model_name: Model name
        messages: Message list
        tools: Tool definition list
        tool_choice: Tool selection mode
        
    Returns:
        Any: Model response object
    """
    try:
        client = OpenAI(api_key=api_key, base_url=url)
        
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice
        )
        
        #Tell the application which function to call
        return response
        
    except Exception as e:
        print(f"ツール呼び出しエラー: {str(e)}")
        raise