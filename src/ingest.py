import shutil
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.config import (DATA_DIR, CHROMA_DB_DIR, EMBEDDINGS_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP)

def load_and_split_pdfs():
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {DATA_DIR}")

    pages = []
    for pdf_path in pdf_files:
        loader = PyMuPDFLoader(file_path=str(pdf_path))
        pdf_pages = loader.load()
        print(f"Loaded {pdf_path.name}: {len(pdf_pages)} pages")
        pages.extend(pdf_pages)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""])

    docs = text_splitter.split_documents(pages)
    print(f"Total pages: {len(pages)}, created chunks: {len(docs)}")
    return docs


def load_vectorstore(force_rebuild: bool = False) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)

    if force_rebuild and CHROMA_DB_DIR.exists():
        shutil.rmtree(CHROMA_DB_DIR)

    if CHROMA_DB_DIR.exists() and any(CHROMA_DB_DIR.iterdir()):
        return Chroma(
            persist_directory=str(CHROMA_DB_DIR),
            embedding_function=embeddings)

    docs = load_and_split_pdfs()
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(CHROMA_DB_DIR))
    print(f"Vector store created at {CHROMA_DB_DIR}")
    return vectorstore


if __name__ == "__main__":
    load_vectorstore(force_rebuild=True)