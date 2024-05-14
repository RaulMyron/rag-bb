import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

import chromadb
from dotenv import load_dotenv
import os
import logging

logging.basicConfig(level=logging.INFO)


def load_and_split(splitter, path):
    """
    Load a PDF document from the given path and split it into an array of documents using the specified splitter.

    Args:
        splitter (TextSplitter): The splitter object used to split the text content of the PDF document.
        path (str): The path to the PDF document.

    Returns:
        list: An array of chunk objects, each containing a chunk of the original document.

    """
    loader = PyPDFLoader(path)
    doc_array = loader.load_and_split(text_splitter=splitter)

    return doc_array

def init_splitter(chunk, overlap):
    """
    Initialize and return a RecursiveCharacterTextSplitter object.

    Parameters:
    - chunk (int): The size of each chunk.
    - overlap (int): The overlap between consecutive chunks.

    Returns:
    - splitter (RecursiveCharacterTextSplitter): The initialized splitter object.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk,
        chunk_overlap=overlap,
        length_function=len,
        is_separator_regex=False
    )

    return splitter

def init_embedding_model(model_name):
    """
    Initializes an embedding model using the specified model name.

    Args:
        model_name (str): The name of the embedding model to initialize.

    Returns:
        embedding_model: The initialized embedding model.

    """
    embedding_model = OpenAIEmbeddings(model=model_name)    
    return embedding_model

def init_connection_vectordb(host, port, collection, model):
    """
    Initializes a connection to VectorDB and creates a collection if it doesn't exist.

    Args:
        host (str): The host address of the VectorDB server.
        port (int): The port number of the VectorDB server.
        collection (str): The name of the collection to be created.
        model: The embedding function to be used for the collection.

    Returns:
        Chroma: An instance of the Chroma class representing the connection to VectorDB.

    Raises:
        ValueError: If the specified collection already exists.

    """
    c_client = chromadb.HttpClient(host=host, port=port)
    
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
        
        logging.info("Iniciando o processamento do documento: %s", doc)
        splitted_docs = load_and_split(splitter, doc)
        
        _ = db.add_documents(splitted_docs)
        logging.info("Finalizado o processamento do documento: %s", doc)

    