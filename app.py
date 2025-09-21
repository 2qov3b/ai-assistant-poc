import streamlit as st
from llm_api import call_llm

# Custom CSS styles
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stSidebar {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        margin-bottom: 10px;
    }
    .message-container {
        display: flex;
        margin-bottom: 10px;
    }
    .user-message {
        background-color: #e1f5fe;
        padding: 10px;
        border-radius: 10px;
        margin-left: auto;
        max-width: 80%;
    }
    .bot-message {
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 10px;
        margin-right: auto;
        max-width: 80%;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize variables"""
    if 'current_page' not in st.session_state:
        st.session_state.current_page = None
    
    #LLM Configuration
    if 'llm_config' not in st.session_state:
        st.session_state.llm_config ={
            'url':'https://api.deepseek.com',
            'api_key':'',
            'model':'deepseek-chat'
        }
    #Customer Assistant Configuration
    if 'bot_config' not in st.session_state:
        st.session_state.bot_config= {
            'name':'AIアシスタント',
            'description' : '私はAIチャット君です！お客様からの質問に丁寧にお答えします！',
            'model':'deepseek-chat'
        }

    #Chat Message Information
    if 'messages' not in st.session_state:
        st.session_state.messages = []

def show_sidebar():
    """Show left sidebar"""
    with st.sidebar:
        st.title("AIアシスタントシステム")
        st.markdown("---")

        #Menu button
        if st.button("会話チャット"):
            st.session_state.current_page = 'chat'

        #LLM Configuration
        if st.button("言語モデル設定"):
            st.session_state.current_page = 'model_config'

        #Customer Assistant Configuration
        if st.button("AIアシスタント設定"):
            st.session_state.current_page = 'bot_config'

def show_main_content():
    """Show main window"""

    if st.session_state.current_page == 'chat':
        show_chat()
    elif st.session_state.current_page == 'model_config':
        show_model_config()
    elif st.session_state.current_page == 'bot_config':
        show_bot_config()
    else:
        st.title("AIアシスタントへようこそ")
        st.write("左側のメニューから機能を選択してください")

def show_chat():
    """Show chat interface"""
    st.title("AIチャット君")

    #Displays information about the message
    if not st.session_state.messages:
        st.session_state.messages.append({
            "role":"asssitant",
            #LLM Customer Service Assistant
            "content":st.session_state.bot_config['description']
        })
    
    #Create a dialog box to save the content of the question and answer, including the user's questions and the customer service assistant's answers
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                #Set the user-related message style, you can use markdown format
                st.markdown(f'<div class="message-container"><div class="user-message">{message["content"]}</div></div>', 
                           unsafe_allow_html=True)
            else:
                #Set the AI assistant related message style, you can use markdown format
                st.markdown(f'<div class="message-container"><div class="bot-message">{message["content"]}</div></div>', 
                           unsafe_allow_html=True)
    
    message = st.chat_input("質問を入力してください")
    if message :
        #Here you need to call the LLM to process the message
        process_message(message)
        st.rerun()

def process_message(message: str):
    """Processing new messages"""
    if message.strip():
        #Add User Message
        st.session_state.messages.append({
            "role":"user",
            "content": message
        })
        #System prompt words
        system_prompt =f"{st.session_state.bot_config['description']}"
        #Call the LLM to respond
        bot_response = call_llm(
            url= st.session_state.llm_config['url'],
            api_key=st.session_state.llm_config['api_key'],
            model_name=st.session_state.llm_config['model'],
            prompt=message,
            system_prompt=system_prompt
        )

        st.session_state.messages.append({
            "role":"assistant",
            "content":bot_response
        })

def show_model_config():
    """Display LLM configuration interface"""
    st.title("言語モデル設定")

    #url setting
    st.text_input(
        "API URL",
        value = st.session_state.llm_config['url'],
        key="llm_url",
        help="言語モデルAPIのURLアドレス"
    )

    #api key setting
    st.text_input(
        "API Key",
        value= st.session_state.llm_config['api_key'],
        type="password",
        key="llm_key",
        help="言語モデルにアクセスAPIキー"
    )

    #Model name
    st.text_input(
        "言語モデル名",
        value=st.session_state.llm_config['model'],
        key = "llm_model",
        help = "使用する言語モデルの名前を入力します。例: deepseek-chat"
    )

    #Save button
    if st.button("保存", type="primary"):
        save_model_config()

    
def save_model_config():
    """言語モデルの設定"""
    st.session_state.llm_config ={
        'url':st.session_state.llm_url,
        'api_key':st.session_state.llm_key,
        'model':st.session_state.llm_model
    }
    st.success("設定が保存されました!")


def show_bot_config():
    """Display the customer service configuration interface"""
    st.title("AIアシスタント設定")

    #Customer service name
    st.text_input(
        "AIアシスタント名前",
        value = st.session_state.bot_config['name'],
        key="bot_name",
        help="AIアシスタントの名前を設定する"
    )

    #Customer service description
    st.text_area(
        "AIアシスタント説明",
        value=st.session_state.bot_config['description'],
        key="bot_description",
        help="AIアシスタントの説明"
    )

    #Model
    st.text_input(
        "利用モデル",
        value=st.session_state.llm_config['model'],
        disabled=True,
        help="現在のシステム構成の言語モデル"
    )

    if st.button("保存", type="primary"):
        save_bot_config()
        st.success("AIアシスタントの設定が保存されました")

def save_bot_config():
    st.session_state.bot_config={
        'name': st.session_state.bot_name,
        'description':st.session_state.bot_description
    }


def main():
    #1 Configuring Variables
    init_session_state()
    #2 Show left sidebar
    show_sidebar()
    #3 Show main window
    show_main_content()

if __name__ == "__main__":
    main()