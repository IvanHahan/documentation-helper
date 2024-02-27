from dotenv import load_dotenv

load_dotenv()

import os

from langchain_community.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain.embeddings.huggingface_hub import HuggingFaceHubEmbeddings
from tqdm import tqdm

def batchify(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


INDEX_NAME = "langchain-doc-index"


def ingest_docs():
    loader = ReadTheDocsLoader(
        "langchain-docs/langchain.readthedocs.io/en/latest/modules/chains"
    )

    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    print(f'Splitted into {len(documents)} chunks')
    
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    embeddings = HuggingFaceHubEmbeddings(
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    print(f"Going to add {len(documents)} to Pinecone")
    
    for i, doc_batch in tqdm(enumerate(batchify(documents, 50))):
        if i == 0:
            db = FAISS.from_documents(doc_batch, embeddings)
        else:
            db.add_documents(doc_batch)
    db.save_local('langchain-docs')
    print("****Loading to vectorstore done ***")


if __name__ == "__main__":
    ingest_docs()
