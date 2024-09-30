import os
import ollama
import pdfplumber
import streamlit as st
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from langchain_community.vectorstores import Qdrant
#from langchain.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
#from openai import OpenAI

load_dotenv()

# Variáveis de ambiente
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")  # Alterado para o nome do serviço Docker
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "ollama")  # Novo: host do Ollama
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))  # Novo: porta do Ollama
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.1")



def get_pdf_text(pdf_files):
    """Extrai texto de arquivos PDF."""
    text = ""
    for pdf_file in pdf_files:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:  
                    text += page_text + " "
    return text

def split_text_into_chunks(text, chunksize):
    """Divide o texto em chunks menores de um tamanho específico."""
    words = text.split()
    chunks = [' '.join(words[i:i + chunksize]) for i in range(0, len(words), chunksize)]
    return chunks 

def store_in_vectordb(chunks, collection_name,vectosize):
    """Armazena chunks de texto no Qdrant com embeddings."""
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    #clientollama = OpenAI(
    #    base_url='http://localhost:11434/v1/',
    #    api_key='ollama'
    #)
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vectosize, distance=Distance.COSINE),
        )

    points = []
    for i, d in enumerate(chunks):
        if d.strip():  # Verifica se o chunk não é vazio
            #embed = OllamaEmbeddings(model=OLLAMA_MODEL)
            response = ollama.embeddings(model=OLLAMA_MODEL, prompt=d)
            #embedding = embed.embed_query(d)
            embedding = response.get("embedding")

            if embedding:  # Verifica se o embedding foi gerado corretamente
                points.append(PointStruct(id=i, vector=embedding, payload={"document": d}))

    if points:
        client.upsert(collection_name=collection_name, points=points)
    else:
        st.error("Nenhum chunk válido foi encontrado para armazenar.")


def load_collections():
    """Carrega coleções existentes do Qdrant."""
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    collections = client.get_collections()
    collection_names = [collection.name for collection in collections.collections]
    return collection_names

def display_collection_items(collection_name):
    """Exibe itens de uma coleção específica do Qdrant."""
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    response, _ = client.scroll(
        collection_name=collection_name,
        limit=100,
        with_payload=True,
        with_vectors=False
    )
    
    points = response
    for point in points:
        document_content = point.payload.get("document")
        if document_content and document_content.strip():  
            st.write(document_content)
        else:
            st.warning("Encontrado documento com conteúdo vazio ou inválido.")        


def start_chat_with_collection(collection_name):
    """Inicia um chat utilizando uma coleção de Qdrant."""
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    # Inicialize os embeddings
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)  # Certifique-se de que este seja o modelo correto
    
    # Inicialize o Qdrant com os embeddings
    vectorstore = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings,
        content_payload_key="document",  # Especifica a chave do payload onde o conteúdo do documento está armazenado
        )

    return vectorstore
