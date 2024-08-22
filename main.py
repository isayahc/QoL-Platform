import chromadb
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
import os
from langchain_chroma import Chroma

import chromadb.utils.embedding_functions as embedding_functions
from langchain.chains import VectorDBQA
from langchain_community.llms import HuggingFaceHub
from langchain.chains import VectorDBQA
from langchain.prompts import PromptTemplate

from langchain_community.embeddings import HuggingFaceEmbeddings

COLLECTION_NAME = "SampleCollection"


PERSIST_DIRECTORY = os.getenv('VECTOR_DATABASE_LOCATION')

persistent_client = chromadb.PersistentClient()
collection = persistent_client.get_or_create_collection(COLLECTION_NAME)


template = """Context: {context}

Question: {question}

Answer the question based on the context provided. If the answer cannot be found in the context, say "I don't have enough information to answer that question."

Answer:"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)


llm = HuggingFaceHub(
    # repo_id="google/flan-t5-xl",
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    model_kwargs={"temperature": 0.5, "max_length": 512}
)


model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

hf_embeddings = HuggingFaceEmbeddings(
    model_name=os.getenv("EMBEDDING_MODEL"),
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


results = collection.query(
    query_texts=["tell me about asthma "],
    n_results=2,
    # where={"metadata_field": "is_equal_to_this"}, # optional filter
    # where_document={"$contains":"search_string"}  # optional filter
)

vectordb = Chroma(
    persist_directory=PERSIST_DIRECTORY, 
    embedding_function=hf_embeddings,
    )


# Create the VectorDBQA chain
qa = VectorDBQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    vectorstore=vectordb,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

if __name__ == '__main__':
    # Example usage
    query = "What is the capital of France?"
    result = qa({"query": query})
    print(result['result'])

