from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
import asyncio

OLLAMA_URL = "http://localhost:11434"

OLLAMA_HEADERS = {"Host": "localhost:11434"}

llm = ChatOllama(model="granite3.3:8b", base_url=OLLAMA_URL, base_headers=OLLAMA_HEADERS)
print("aqui rodou")

embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_URL)
print("foi")

vector_store = Chroma(
    persist_directory="./chroma_langchain_db",
    embedding_function=embeddings
)

retriever = vector_store.as_retriever(search_kargs={"k": 3})

template = """
Você é um Conselheiro Espiritual Bíblico, amável, compassivo e sábio. Sua missão é fornecer respostas de aconselhamento teológico baseadas exclusivamente na Bíblia e no contexto fornecido.

**REGRAS:**
1. **Tom:** Use um tom de voz empático, encorajador e acolhedor.
2. **Conteúdo:** A resposta DEVE ser fundamentada nos trechos bíblicos/contexto fornecidos.
3. **Estrutura:** Divida a resposta em três partes caso for uma explicação completa:
   a. Reconheça a dificuldade do usuário.
   b. Use o CONTEXTO para construir o conselho.
   c. **Versículos (Referência):** Cite as referências bíblicas mais relevantes do contexto (Ex: Romanos 8:28, Salmos 23:1).
4. **Estrutura:** Divida a resposta em duas partes caso a pergunta for algo relacionado ao estudo literal da bíblia (Exemplo:"Qual o nome da mãe de Jesus"):
   a. Responda a pergunta 
   b. Passe uma explicação ou reflexão sobre a resposta do contexto
5. **Segurança:** JAMAIS use linguagem que incentive julgamento, culpa ou que promova qualquer forma de extremismo.
6. Sempre lembre da pergunta anterior caso a pergunta do usuário esteja no mesmo contexto, mas se não estiver mude o contexto junto com o usuário.
7. Caso o usuário pergunte um versículo muito específico (exemplo: "o que está escrito em Gênesis 26:10?") divida a resposta em duas partes:
   a. Diga o texto do versículo de forma direta.
   b. Pergunte se o usuário quer se aprofundar em algo sobre aquele versículo.
8. Caso pergunte o que você é, diga resumidamente que é um Conselheiro Espiritual Bíblico, e pergunte se o usuário quer se aprofundar em algo.

Contexto (Versículos):
{context}

Pergunta:
{question}

Versículos Relevantes:
"""

prompt_rag = ChatPromptTemplate.from_template(template)

chain_rag = (
    RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )
    | prompt_rag
    | llm
    | StrOutputParser()
)

def recusa(_):
    return (
        "Sinto muito, mas como assistente focado em te assistenciar com a bíblia, não posso fornecer conselhos sobre questões médicas, financeiras ou psicológicas. Para essas questões, é fundamental buscar a orientação de um profissional qualificado"
    )

template_cassifier = """
Dada a pergunta do usuário, classifique-a em uma destas três categorias:
'teologia_biblica', 'aconselhamento_perigoso' ou 'aconselhamento_seguro'

- 'teologia_biblica': Perguntas sobre histórias da bíblia, personagens, versículos, doutrinas, situações onde versículos da bíblia pode ajudar ou saudações simples (olá, tudo bem).
- 'aconselhamento_seguro': Para perguntas sobre dificuldades espirituais, fé, oração, dúvidas e questões da vida cotidiana que não envolvam risco de vida ou crime.
- 'aconselhamento_perigoso': Perguntas pedindo conselhos médicos (como remédios), conselhos sobre relacionamenstos (namoro, casamento), financeiros (investimentos, dívidas), ou decisões profissionalizantes (carreira profissional).

Pergunta: {question}

**FORMATO OBRIGATÓRIO:** Forneça APENAS o nome da categoria, em minúsculas, sem aspas, e nada mais.

Classificação:
"""

prompt_classifier = ChatPromptTemplate.from_template(template_cassifier)

chain_classifier = (
    prompt_classifier
    | llm
    | StrOutputParser()
)

chain_rag_input = itemgetter("question") | chain_rag

full_chain = (
    RunnableParallel(
        classificacao=chain_classifier,
        question=RunnablePassthrough()
    )
    | RunnableBranch(

        (
            lambda x: "aconselhamento_perigoso" in x["classificacao"].lower().strip(),
            recusa
        ),
        
        chain_rag_input
    )
)

#Run the AI without interruptions
#lembra de rodar uvicorn RAG:app --reload
from fastapi import FastAPI
from pydantic import BaseModel
import json

def rag_function(query: str) -> str:
    oke = full_chain.invoke(query)
    return oke

app = FastAPI(title="WorshipSoulGPT")

class QueryModel(BaseModel):
    question_text: str

class ResponseModel(BaseModel):
    ai_answer: str

@app.post("/perguntar", response_model=ResponseModel)
def question_the_ai(query: QueryModel):
    answer = rag_function(query.question_text)
    return {"ai_answer": answer}