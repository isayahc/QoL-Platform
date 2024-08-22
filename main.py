import chromadb
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
import os
from langchain_chroma import Chroma

import chromadb.utils.embedding_functions as embedding_functions

COLLECTION_NAME = "SampleCollection"


huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
    api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


persistent_client = chromadb.PersistentClient()
collection = persistent_client.get_or_create_collection(COLLECTION_NAME)


results = collection.query(
    query_texts=["This is a query document"],
    n_results=2,
    # where={"metadata_field": "is_equal_to_this"}, # optional filter
    # where_document={"$contains":"search_string"}  # optional filter
)

# embedding_function = SentenceTransformerEmbeddings(
#         model_name=os.getenv("EMBEDDING_MODEL"),
#         )

vector_store_from_client = Chroma(
    client=persistent_client,
    collection_name=COLLECTION_NAME,
    embedding_function=huggingface_ef,
)
x = 0