from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Tuple, List
import torch
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


def process_document_deepseek(file, chunk_size: int=100, chunk_overlap: int=20, custom_separators:bool = False, separators: list=None)-> Tuple[FAISS, List[str]]:
    """Process uploaded documents (deepseek-only)
    Args:
        file: Uploaded file
        chunk_size: Chunk size
        chunk_overlap: Chunk overlap
        custom_separators: Whether to use custom separators
        separators: Custom separators
    Returns:
        FAISS: Vector database
        List: Text chunks after segmentation
    """
    #Read file
    text = file.read().decode("utf-8")

    #Text separation
    if custom_separators and separators:
        text_splitter = RecursiveCharacterTextSplitter(
            separators= separators,
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            )
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,)
    
    #Document chunks
    chunks = text_splitter.split_text(text)

    #Embed the segmented documents into the vector database
    #The ability of LLM affects search capabilities
    model_name = "BAAI/bge-large-zh-v1.5"
    model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
    encode_kwargs = {"normalize_embeddings": True}

    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs = model_kwargs,
        encode_kwargs = encode_kwargs
    )

    #Embed the cut file blocks into the vector database
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store, chunks