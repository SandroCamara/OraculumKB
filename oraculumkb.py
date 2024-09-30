import streamlit as st
import os
from llm_rag import store_in_vectordb, load_collections, display_collection_items, split_text_into_chunks, get_pdf_text, start_chat_with_collection
from llm_engine import llm_config, create_chain
from dotenv import load_dotenv

load_dotenv()

llm = llm_config()
chain = create_chain(llm)

CHUNK_SIZE = os.getenv("CHUNK_SIZE")
ANSWERS = os.getenv("ANSWERS")
VECTORSIZE = os.getenv("VECTORSIZE")

def write_message(role, content, save=True):
    if save:
        st.session_state.messages.append({"role": role, "content": content})

    with st.chat_message(role, avatar="avatarrobo.png" if role == 'assistant' else None):
        st.markdown(content)


def handle_submit(message,vectordb):
    with st.spinner('Pesquisando...'):
        docs = vectordb.similarity_search(message, k=int(ANSWERS))
        context = "\n\n".join([doc.page_content for doc in docs])
        resposta = chain.run(context=context, question=message)
        write_message('assistant',resposta)


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Olá! Eu sou o Oraculum, como posso te ajudar?"},
    ]        

if 'vectors' not in st.session_state:
    st.session_state['vectors'] = None

if 'selected_collection' not in st.session_state:
    st.session_state['selected_collection'] = '' 

st.set_page_config("Oraculo IA+RAG", page_icon=':heartbeat:')
st.title('Bem vindo ao Oraculum Knowledge Base')    
st.header(f'Base Selecionada: {st.session_state["selected_collection"]}')    
with st.sidebar:
    st.subheader("Documento")
    pdf_docs = st.file_uploader("Carregue seus documentos aqui e clique em 'Processar'", type="pdf", accept_multiple_files=True)
    collection_name = st.text_input("Nome da nova Base de Conhecimento:")

    if st.button("Processar"):
        if pdf_docs:
            if collection_name:
                with st.spinner("Processando"):
                    extracted_text = get_pdf_text(pdf_docs)
                    text_chunks = split_text_into_chunks(extracted_text,int(CHUNK_SIZE))
                    store_in_vectordb(text_chunks, collection_name,int(VECTORSIZE))
                    st.success(f"Base de Conhecimento '{collection_name}' criada")
            else:
                st.error("Por favor, forneça um nome para a nova Base de Conhecimento.")
        else:
            st.error("Por favor, forneça ao menos um arquivo para a nova Base de Conhecimento.")

    collections = load_collections()
    if collections:
        selected_collection = st.selectbox("Base de Conhecimento:", collections)
        st.session_state['selected_collection'] = selected_collection if selected_collection else ''
        if st.button("Iniciar Chat com a Base Selecionada"):
            if selected_collection:
                st.session_state['vectors'] = start_chat_with_collection(selected_collection)
                st.session_state.messages = [
                    {"role": "assistant", "content": "Olá! Eu sou o Oraculo, como posso te ajudar?"},
                ]


    else:
        st.warning("Nenhuma Base de Conhecimento encontrada. Carregue uma base de conhecimento primeiro.")    

for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

if prompt := st.chat_input("Faça sua pergunta"):
    if st.session_state['selected_collection'] == '':
        st.error("Selecione uma Base de Conhecimento !")
    else:        
        write_message('user', prompt)
        handle_submit(prompt,st.session_state['vectors'])