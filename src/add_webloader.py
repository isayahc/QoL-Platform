
import chromadb
import os
from typing import List, Optional
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters.sentence_transformers import SentenceTransformersTokenTextSplitter
from langchain_core.documents import Document
import uuid
import dotenv
import os

dotenv.load_dotenv()
PERSIST_DIRECTORY = os.getenv('VECTOR_DATABASE_LOCATION')
COLLECTION_NAME="SampleCollection"

def generate_uuid() -> str:
    """
    Generate a UUID (Universally Unique Identifier) and return it as a string.

    Returns:
        str: A UUID string.
    """
    return str(uuid.uuid4())


def chunk_web_data(
    urls: List[str],
    chunk_overlap: Optional[int] = 50,
    tokens_per_chunk: Optional[int] = None,
    model_name: str = os.getenv("EMBEDDING_MODEL"),
    chunk_size: int = 1000,
    ) -> List[Document]:
    """
    ## Summary
    This function is used to chunk webpages
    
    ## Arguments
    urls list[str] : a list of urls  to be chunks
    chunk_size int : the chunking size
    model_name str : the embedding model used to chunk will use the environment default unless overwritten
    tokens_per_chunk int | None : the amount of chunks per token paramter inhereted from `SentenceTransformersTokenTextSplitter`
    chunk_size int : the size of chunks per `Document`

    ## Return
    it may be a List[Document] or None if it is a List[Document] then these chunks will be 
    embedded wiht a different function or method
    """
    
    text_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=chunk_overlap,
        tokens_per_chunk=tokens_per_chunk,
        model_name=model_name,
        chunk_size=chunk_size
    )
    
    loader = WebBaseLoader(urls)

    data = loader.load_and_split(
        text_splitter=text_splitter
    )
    
    return data



if __name__ == '__main__':
    
    
    client = chromadb.PersistentClient(
    #  path=persist_directory, # this value is optional
    )
    
    collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    )
    
    # example_site = "https://www.hsph.harvard.edu/"
    example_site = "https://www.hsph.harvard.edu/news/hsph-in-the-news/using-cleaning-products-may-raise-womens-risk-of-asthma-respiratory-conditions/"
    
    web_chunks = chunk_web_data(
        [example_site]
    )
    
    documents_page_content:list = [i.page_content for i in web_chunks]
    
    embedding_function = SentenceTransformerEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL"),
        )
    
    model_name=os.getenv("EMBEDDING_MODEL")
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    
    
    
for i in web_chunks:
        
        collection.add(
            ids=[generate_uuid() for i in web_chunks], # give each document a uuid
            documents=[i.page_content for i in web_chunks], # contents of document
            embeddings=hf.embed_documents([i.page_content for i in web_chunks]),
            metadatas=[i.metadata for i in web_chunks],  # type: ignore
        )

