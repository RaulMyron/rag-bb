# This is RAG LMM generator


chromadb running in docker


1. Definir uma network
> docker network create bb

2. Subir um servidor do chroma_db
> docker run --name chroma-server -p 8000:8000 --network bb -d chromadb/chroma

3. Buildar a imagem do pipeline de ingestão
> docker build -t rag-bb/data-ingestion src/data_ingestion

4. Rodar a imagem do pipeline de ingestão
> docker run --network bb rag-bb/data-ingestion
