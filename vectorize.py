from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import os
import json

embeddings = OllamaEmbeddings(model="nomic-embed-text")

docments = []

for archive in os.listdir('versiculos'):
    if archive.endswith('.json'):
        path = os.path.join('versiculos', archive)

        with open(path, 'r', encoding='utf-8') as f:

            v = json.load(f)

            doc = Document(
                page_content=v['texto'],
                metadata={
                    'livro': v['livro'],
                    'capitulo': str(v['capitulo']),
                    'versiculo': str(v['versiculo']),
                    'abreviacao': v['abreviacao']
                }
            )
            docments.append(doc)

vector_store = Chroma.from_documents(
    documents=docments,
    embedding=embeddings,
    persist_directory="./chroma_langchain_db"
    
)

print("banco criado")