import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb

from dotenv import load_dotenv
import os


def load_and_split(splitter, path):
    
    loader = PyPDFLoader(path)
    doc_array = loader.load_and_split(text_splitter=splitter)

    return doc_array

def init_splitter(chunk, overlap):
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk,
        chunk_overlap=overlap,
        length_function=len,
        is_separator_regex=False
    )

    return splitter

def init_embedding_model(model_name):
    embedding_model = OpenAIEmbeddings(model=model_name)
    return embedding_model

def init_connection_vectordb(host, port, collection, model):
    c_client = chromadb.HttpClient(host, port)
    
    if collection in c_client.list_collections():
        raise ValueError(f"Collection {collection} already exists. Please delete it or use a different name.")

    c_client.get_or_create_collection(collection, metadata={"hnsw:space":"cosine"})
    db = Chroma(collection_name=collection, embedding_function=model, client=c_client)

    return db
    
if __name__ == "__main__":
    load_dotenv()

    splitter = init_splitter(1000, 100)
    
    embedding_model = init_embedding_model(os.getenv("MODEL_NAME"))
    
    db = init_connection_vectordb(os.getenv("CHROMA_HOST"), 
                                  os.getenv("CHROMA_PORT"), 
                                  os.getenv("COLLECTION_NAME"), 
                                  model=embedding_model)
    
    for doc in glob.glob("docs/*.pdf"):
        
        splitted_docs = load_and_split(splitter, doc)
        
        _ = db.add_documents(splitted_docs)
        print('acabei')

    