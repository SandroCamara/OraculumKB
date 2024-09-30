import os
from langchain.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.1")

def llm_config():
    llm = Ollama(model=CHAT_MODEL)
    return llm

def create_chain(llm):
    template = """Seja cordial e utilizando as informações fornecidas nos textos abaixo, responda à pergunta do usuário de forma concisa e precisa e Caso não seja fornecido nenhum texto, se desculpe informando que não foi encontrada a resposta.

Textos:
{context}

Pergunta: {question}

Resposta:"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    return chain