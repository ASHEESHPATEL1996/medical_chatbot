from typing import List
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


def filter_to_minimal_docs(documents: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list where each Document
    containing only source in metadata and original page_content.
    """
    minimal_docs = []
    for doc in documents:
        src = doc.metadata.get("source")
        minimal_doc = Document(page_content=doc.page_content, metadata={"source": src})
        minimal_docs.append(minimal_doc)
    return minimal_docs


def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len,
    )
    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk


# vector embedd

def download_embeddings():
    """
    Download and return the HuggingFace embedding model.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    )
    return embeddings