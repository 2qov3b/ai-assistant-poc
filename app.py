import streamlit as st
from llm_api import call_llm, call_llm_docs, classify_intent,handle_order_query
from vector_store import process_document_deepseek
from function import check_order_status

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

    #Knowledge base setting
    if 'knowledge_base' not in st.session_state:
        st.session_state.knowledge_base = {
            'chunks':[],
            'vector_store':None
        }    

    #Define orders
    if 'orders' not in st.session_state:
        st.session_state.orders = []    

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

        #Knowledge base configuration
        if st.button("ナレッジベース設定"):
            st.session_state.current_page = 'knowledge_cofig'    

        #Loading order information
        if st.button("注文設定"):
            st.session_state.current_page = "order_config"

def show_main_content():
    """Show main window"""

    if st.session_state.current_page == 'chat':
        show_chat()
    elif st.session_state.current_page == 'model_config':
        show_model_config()
    elif st.session_state.current_page == 'bot_config':
        show_bot_config()
    elif st.session_state.current_page == 'knowledge_cofig':
        show_knowledge_config()    
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
                if message.get("is_knowledge"):
                    st.markdown(f'<div class="message-container"><div class="knowledge-message">{message["content"]}</div></div>', 
                               unsafe_allow_html=True)
                    with st.expander("一致するナレッジブロックを表示"):
                        #Put the matching documents into docs
                        for i, doc in enumerate(message["docs"],1 ):
                            st.write(f"### ナレッジブロック{i}")
                            st.text(doc["page_content"])
                else:
                    #Set the AI assistant-related message style, you can use markdown format
                    st.markdown(f'<div class="message-container"><div class="bot-message">{message["content"]}</div></div>', 
                            unsafe_allow_html=True)
                    if message.get("sources"):
                        with st.expander("ナレッジベース元"):
                            #Put the knowledge base source in sources
                            for source in message["sources"]:
                                st.write(f"{source}")
    
    message = st.chat_input("質問を入力してください")
    if message :
        #Here you need to call the LLM to process the message
        process_message(message)
        st.rerun()

def process_message(message: str):
    """Processing new messages"""
    if message.strip():
        intent = classify_intent(
            url=st.session_state.llm_config['url'],
            api_key=st.session_state.llm_config['api_key'],
            model_name=st.session_state.llm_config['model'],
            message= message,
            role=st.session_state.bot_config['description']
        )

        st.session_state.messages.append({
            "role":"assistant",
            "content":message,
            "sources": intent
        })

        # Routing to different processing flows based on intent type
        if intent['intent_type'] == 'order':
            response = handle_order_query(
                url=st.session_state.llm_config['url'],
                api_key=st.session_state.llm_config['api_key'],
                model_name=st.session_state.llm_config['model'],
                message=message,
                role=st.session_state.bot_config['description']
            )
        elif intent['intent_type'] == 'knowledge':
            response = handle_knowledge_query(message)
        else:  # Others, it will be handled by human
            response = handle_human_transfer(intent)

        
        # Add assistant reply
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "intent": intent  # Save intent information for debugging
        })

def handle_knowledge_query(message: str) -> str:
    """Handling knowledge base related issues"""
    context = ""
    sources = []
    
    # Check if the knowledge base is loaded
    if st.session_state.knowledge_base['vector_store'] is None:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "⚠️ ナレッジベースはまだ読み込まれていません。まずはナレッジベース設定ページでドキュメントをアップロードしてください。",
            "is_knowledge": False
        })
        return "ナレッジベースはまだ読み込まれていません。まずはナレッジベース設定ページでドキュメントをアップロードしてください。"
            
    retriever = st.session_state.knowledge_base['vector_store'].as_retriever()
    docs = retriever.invoke(message)
    context = "\n".join([doc.page_content for doc in docs])
    sources = [f"ナレッジブロック #{i+1} " 
                for i, doc in enumerate(docs)]
    
    # Display knowledge base search results
    if docs:
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "🔍 ナレッジベースから以下の情報を見つけてください:",
            "sources": sources,
            "is_knowledge": True,
            "docs": [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]
        })
    else:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "ℹ️ ナレッジベースで関連する一致が見つかりません",
            "is_knowledge": False
        })

    if docs:  # If the knowledge base is hit
        bot_response = call_llm_docs(
            docs, 
            message,
            url=st.session_state.llm_config['url'],
            api_key=st.session_state.llm_config['api_key'],
            model_name=st.session_state.llm_config['model']
        )
    else:  # If the knowledge base is not hit
        system_prompt = f"{st.session_state.bot_config['description']}"
        if context:
            system_prompt += f"\n\n現在のナレッジベースのコンテキスト:\n{context}"
        bot_response = call_llm(
            url=st.session_state.llm_config['url'],
            api_key=st.session_state.llm_config['api_key'],
            model_name=st.session_state.llm_config['model'],
            prompt=message,
            system_prompt=system_prompt
        )
    return bot_response

def handle_human_transfer(intent: dict) -> str:
    """Route to human customer service"""
    try:
        confidence = intent.get('confidence', 0.0)
        if confidence > 0.6:
            return "ご質問には手動でのサポートが必要です。カスタマーサービス担当者に転送いたします..."
        return "お客様の問題は手動での処理が必要です。カスタマーサービスチケットを送信いたしましたので、1時間以内にご連絡いたします。"
    except Exception as e:
        print(f"手動エラー処理: {str(e)}")
        return "カスタマーサービス転送サービスは一時的にご利用いただけません。しばらくしてからもう一度お試しください。"

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

def show_order_config():
    """Display the order configuration interface"""
    st.title("注文管理")
    
    # Add an order form
    with st.form("add_order_form"):
        st.subheader("新しい注文を追加")
        
        col1, col2 = st.columns(2)
        with col1:
            username = st.text_input("ユーザー名", key="order_username")
            product = st.text_input("製品名", key="order_product")
        with col2:
            order_id = st.text_input("注文番号", key="order_id")
            status = st.selectbox(
                "注文状況",
                ["支払い待ち", "支払い済み", "発送済み", "完了", "キャンセル"],
                key="order_status"
            )
            date = st.date_input("日付", key="order_date")
        
        if st.form_submit_button("注文を追加する"):
            if not all([username, product, order_id]):
                st.error("必須項目をすべて入力してください")
            else:
                # Check if the order number already exists
                if any(order["order_id"] == order_id for order in st.session_state.orders):
                    st.error("注文番号は既に存在します")
                else:
                    # Add New Order
                    st.session_state.orders.append({
                        "username": username,
                        "product": product,
                        "order_id": order_id,
                        "status": status,
                        "date": str(date)
                    })
                    st.success("注文が正常に追加されました")
    
    # Display order list
    st.subheader("注文リスト")
    if st.session_state.orders:
        # Convert to DataFrame for better display
        orders_df = st.session_state.orders
        st.dataframe(orders_df)
    else:
        st.info("注文データはまだありません")

def show_knowledge_config():
    """Display the knowledge base configuration interface"""
    st.title("ナレッジベースの構成")

    #Upload a TXT file
    st.title("ナレッジベースにアップロード")
    upload_file = st.file_uploader("ドキュメント（TXT形式）をアップロードしてください", type="txt")

    #Parameter configuration
    st.title("ファイルブロック構成")
    col1,col2 = st.columns(2)
    with col1:
        chunk_size = st.number_input("ブロックサイズ", min_value=100, max_value=2000, value=100, step=50, help="各テキストブロックの最大長を指定します")
    with col2:
        chunk_overlap = st.number_input("オーバーラップサイズ", min_value=0, max_value=500, value=20, step=10, help="各テキストブロックのオーバーラップ長を指定します")

    #Delimiter configuration
    st.title("セパレーターの設定")
    use_custom_separators = st.radio("カスタム区切り文字の使用",["いいえ","はい"],index=0,help="区切り文字に従ってテキストを正確に区切るには、「はい」を選択します")

    separators = None
    if use_custom_separators == "はい":
        separators_input = st.text_input("区切り文字入力", value="###", help="テキストを区切るためのカスタム区切り文字を入力します")
    
        separators = [ s.strip() for s in separators_input.split(",") if s.strip()]

    #Save uploaded files to session_state
    if upload_file is not None:
        st.session_state.knowledge_base['upload_file'] = upload_file
    #Process Document Button
    if st.button("文書処理", type="primary") and 'upload_file' in st.session_state.knowledge_base:
        with  st.spinner("文書処理中....."):
            #Process the document and embed it into the vector database. 
            #Return the 1st vector store: vector_store; the 2nd chunks: file chunks
            result= process_document_deepseek(
                st.session_state.knowledge_base['upload_file'],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                custom_separators=(use_custom_separators == 'はい'),
                separators= separators
            )
            
            st.session_state.knowledge_base['vector_store'] = result[0]
            st.session_state.knowledge_base['chunks'] = result[1]

    #Display file block
    if 'chunks' in st.session_state.knowledge_base:
        chunks = st.session_state.knowledge_base['chunks']
        st.subheader("文書セグメンテーション結果")
        st.write(f"{len(chunks)} つのテキストブロックに分かれています")

        #Show first 5 blocks
        for i, chunk in enumerate(chunks[:5],1):
            with st.expander(f"テキストブロック #{i} (長さ：{len(chunk)}文字)"):
                st.text(chunk)
        st.success("ナレッジベースドキュメントの処理が完了しました！")

    if st.session_state.knowledge_base['vector_store']:
        st.success("ナレッジベースが読み込まれました。")    


def main():
    #1 Configuring Variables
    init_session_state()
    #2 Show left sidebar
    show_sidebar()
    #3 Show main window
    show_main_content()

if __name__ == "__main__":
    main()